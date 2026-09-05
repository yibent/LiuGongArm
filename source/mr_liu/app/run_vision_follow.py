"""SO-101 follow-target driven by dual-camera Florence FIND + YOLOE TRACK."""

from __future__ import annotations

import numpy as np
import time
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, XformPrim
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
from isaacsim.core.utils.viewports import set_camera_view

from mr_liu.config import motion_config, scene_config
from mr_liu.motion.follow import FollowTargetController
from mr_liu.paths import repo_root
from mr_liu.perception.camera import SceneCamera, spawn_configured_cameras
from mr_liu.robot.so101 import So101Arm
from mr_liu.sim.spawn import spawn_table_and_so101
from mr_liu.vision import MultiViewFindTrackPipeline, RuntimeConfig
from mr_liu.vision.control import VisionControlServer, VisionRuntimeControl
from mr_liu.vision.worker import VisionWorker


def _bbox_to_table_xy(xyxy: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Map image bbox center to a point on the table plane (heuristic)."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = ((x1 + x2) * 0.5) / max(width, 1)
    cy = ((y1 + y2) * 0.5) / max(height, 1)
    scene = scene_config()
    table = scene["table_pos"]
    world_x = float(table[0]) - 0.15 + (cx - 0.5) * 0.45
    world_y = float(table[1]) - (cy - 0.5) * 0.40
    return world_x, world_y


def setup(
    app,
    *,
    physics_dt: float,
    device: str,
) -> tuple[FollowTargetController, So101Arm, GeomPrim, dict[str, SceneCamera]]:
    scene = scene_config()
    stage_utils.create_new_stage()
    # A physics scene must be created after the new USD stage. Creating it
    # before create_new_stage leaves SimulationManager holding a stale scene.
    SimulationManager.setup_simulation(dt=physics_dt, device=device)
    from mr_liu.sim.timing import configure_render_timing
    configure_render_timing(physics_dt, float(motion_config().get("render_hz", 30)))
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(float(scene["distant_intensity"]))
    spawn_table_and_so101()

    target_cfg = scene["target"]
    Cube(
        paths=target_cfg["prim_path"],
        positions=[target_cfg["position"]],
        sizes=float(target_cfg["size"]),
        colors=(0.1, 0.85, 0.35),
    )
    target = GeomPrim(paths=target_cfg["prim_path"])
    from pxr import Semantics
    target_semantics = Semantics.SemanticsAPI.Apply(target.prims[0], 'Semantics')
    target_semantics.CreateSemanticTypeAttr().Set('class')
    target_semantics.CreateSemanticDataAttr().Set('block')
    app.update()
    set_camera_view(eye=[1.6, 1.4, 1.7], target=[0.15, 0.0, 1.15], camera_prim_path="/OmniverseKit_Persp")

    cameras = spawn_configured_cameras()
    if set(cameras) != {"scene", "wrist"}:
        raise RuntimeError(f"Dual-view vision requires scene and wrist cameras, got {sorted(cameras)}")
    app.update()

    arm = So101Arm()
    app.update()
    arm.validate_joint_mapping()
    print(f"[mr_liu] Articulation: {arm.prim_path}")
    print(f"[mr_liu] USD DOFs: {arm.dof_names}")

    controller = FollowTargetController(arm.articulation, target)
    return controller, arm, target, cameras


def _try_load_pipelines(
    cfg: RuntimeConfig,
    *,
    device: str | None,
) -> MultiViewFindTrackPipeline | None:
    weights = repo_root() / "yoloe-26x-seg.pt"
    if not weights.is_file():
        print(f"[mr_liu] YOLOE weights missing ({weights}); vision loop disabled, cube is still draggable.")
        return None
    pipe = MultiViewFindTrackPipeline(yoloe_weights=weights, device=device)
    try:
        pipe.load(cfg.fast_backend)
    except Exception as exc:  # noqa: BLE001
        print(f"[mr_liu] Vision models failed to load: {exc}")
        return None
    print(
        f"[mr_liu] Dual-view vision ready prompt={cfg.prompt!r} "
        f"fast={cfg.fast_backend} views={sorted(pipe.pipelines)}"
    )
    return pipe


def main(
    app,
    *,
    test_frames: int | None = None,
    device: str | None = None,
    prompt: str = "cube",
    fast: str = "yoloe",
    slow_interval: int = 45,
    control_host: str = "127.0.0.1",
    control_port: int = 7861,
    follow_target: bool = True,
    grasp_backend: str = "graspgenx",
) -> None:
    motion = motion_config()
    physics_dt = float(motion["physics_dt"])
    physics_device = device or str(motion["device"])
    controller, arm, target, cameras = setup(
        app,
        physics_dt=physics_dt,
        device=physics_device,
    )
    app.update()
    app_utils.play()
    app.update()
    if follow_target:
        arm.apply_configured_pose()
        app.update()
    else:
        print("[mr_liu] Vision-only mode keeps the USD initial arm pose for tabletop coverage.")
    drive_info = arm.configure_drives()

    vis_cfg = RuntimeConfig(
        prompt=prompt,
        fast_backend=fast,  # type: ignore[arg-type]
        slow_interval=max(0, int(slow_interval)),
    )
    control = VisionRuntimeControl(vis_cfg, follow_enabled=follow_target)
    from mr_liu.vision.grounding import SceneGrounding, surface_point, confirmed_mask, instance_association
    control.grounding = SceneGrounding()
    from mr_liu.config import robot_config
    from mr_liu.motion.commands import MotionCommands
    from scipy.spatial.transform import Rotation
    robot_cfg = robot_config()
    control.motion = MotionCommands(repo_root() / robot_cfg["robot_dir"] / robot_cfg["urdf"], robot_cfg["default_joint_positions"], config=motion.get("commands"))
    control.motion.drive_info = drive_info
    robot_base = XformPrim(robot_cfg["usd_prim_path"])
    if grasp_backend != "disabled":
        from mr_liu.grasp.runtime import GraspRuntime
        from mr_liu.grasp.isaac_session import IsaacGraspSession
        cameras["scene"].enable_instance_segmentation()
        for camera in (cameras["scene"], cameras["wrist"]):
            camera.enable_instance_segmentation(leaf_paths=True)
        control.grasp = GraspRuntime(control, lambda grounded, selected, guard, progress: IsaacGraspSession(
            arm, cameras["wrist"], app, 1/float(motion.get("render_hz", 30)), grounded, selected, guard, progress,
            backend=grasp_backend, output_root=repo_root() / "output/busagent_grasp",
            scene=cameras["scene"],
        ))

    def update_motion():
        q = arm.articulation.get_dof_positions().numpy().reshape(-1)
        qd = arm.articulation.get_dof_velocities().numpy().reshape(-1)
        base_position, base_orientation = robot_base.get_world_poses()
        base = np.eye(4)
        base[:3, 3] = base_position.numpy()[0]
        quat = base_orientation.numpy()[0]
        base[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        def apply(values):
            arm.articulation.set_dof_position_targets([[values[name] for name in arm.dof_names]])
        def apply_velocity(values):
            arm.articulation.set_dof_velocity_targets([[values[name] for name in arm.dof_names]])
        return control.motion.tick(dict(zip(arm.dof_names, q.tolist())), base, apply, app_utils.is_playing(),
                                   sim_time=SimulationManager.get_simulation_time(),
                                   velocities=dict(zip(arm.dof_names, qd.tolist())),
                                   apply_velocity=apply_velocity)
    server = VisionControlServer(control, host=control_host, port=control_port)
    host, port = server.start()
    print(f"[mr_liu] Runtime vision control: http://{host}:{port}")

    pipe = _try_load_pipelines(vis_cfg, device=device)
    if pipe is None:
        control.set_error("Vision models failed to load or weights are missing")
    else:
        control.set_model_status(pipe.status())
    target_xform = XformPrim(scene_config()["target"]["prim_path"])
    table_z = float(scene_config()["target"]["position"][2])

    print("SO-101 vision follow-target (cuMotion RMPflow)")
    print("  Top + wrist RGB → shared Florence FIND → independent YOLOE TRACK.")
    if follow_target:
        print("  Top-view detection → /World/TargetCube → arm.")
    else:
        print("  Automatic follow disabled; manual motion commands remain available through the control API.")
    print(f"  Change targets while running: POST http://{host}:{port}/api/prompt")

    reset_needed = True
    frames = 0
    vision_cycles = 0
    motion_owns_arm = False
    def physics_step(dt, context):
        nonlocal motion_owns_arm, reset_needed
        motion_owns_arm = update_motion()
        sim_t = SimulationManager.get_simulation_time()
        if reset_needed:
            controller.reset(sim_t)
            reset_needed = False
        if control.follow_enabled() and not motion_owns_arm:
            controller.step(sim_t)

    # All joint reads/writes remain on Kit's physics thread; inference cannot
    # make trajectory time jump ahead of actual simulated physics.
    callback = SimulationManager.register_callback(physics_step, SimulationEvent.PHYSICS_POST_STEP)

    def process_vision(packet):
        results = pipe.process_frames(packet["frames"], packet["config"])
        # JPEG encoding is also off the physics/render thread.
        if control.snapshot().prompt_version == packet["config"].prompt_version:
            for view, result in results.items():
                control.publish(view, result)
        return packet, results

    worker = VisionWorker(process_vision) if pipe is not None else None
    next_capture = 0.
    try:
        while app.is_running():
            app.update()
            if control.grasp is not None:
                control.grasp.poll()
            if app_utils.is_playing() and SimulationManager.is_simulating():
                if worker is not None:
                    ready = worker.poll()
                    if ready is not None:
                        packet, results = ready
                        current_cfg = packet["config"]
                        if current_cfg.prompt_version == control.snapshot().prompt_version:
                            camera_frames = packet["frames"]
                            observed_at = packet["observed_at"]
                            scene_depth, scene_semantics = packet["depth"], packet["semantics"]
                            for view, result in results.items():
                                stats = result.stats
                                if stats.path == "slow" or vision_cycles % 30 == 0:
                                    print(f"DUAL_TRACK view={view} frame={stats.frame_idx} path={stats.path} "
                                          f"slow_ms={stats.slow_ms:.1f} fast_ms={stats.fast_ms:.1f}")
                            scene_result = results["scene"]
                            detections = scene_result.fast or scene_result.slow
                            grounded_detections, rejections = [], []
                            for det in detections:
                                mask, rejection = confirmed_mask(camera_frames["scene"], scene_semantics, det.xyxy, current_cfg.prompt)
                                if rejection:
                                    rejections.append(rejection)
                                    continue
                                point = surface_point(det.xyxy, scene_depth, packet["unproject"], mask)
                                association = instance_association(packet.get("instances"), mask,
                                                                   packet.get("leaf_instances"))
                                grounded_detections.append(dict(xyxy=det.xyxy.tolist(), label=det.label,
                                    score=float(det.score), world_position_m=point,
                                    object_id=association["object_id"], association=association,
                                    acquisition_id=packet.get("acquisition_id")))
                            control.grounding.publish(current_cfg.prompt_version, current_cfg.prompt,
                                                      grounded_detections, observed_at, rejections)
                            if control.follow_enabled() and not motion_owns_arm and detections and time.monotonic()-observed_at < 1:
                                height, width = camera_frames["scene"].shape[:2]
                                x, y = _bbox_to_table_xy(detections[0].xyxy, width, height)
                                target_xform.set_world_poses(positions=[[x, y, table_z]])
                            vision_cycles += 1
                    if worker.available and time.monotonic() >= next_capture:
                        next_capture = time.monotonic() + 1/float(motion.get("vision_capture_hz", 10))
                        scene_packet = cameras["scene"].grounding_frame()
                        wrist_bgr = cameras["wrist"].rgb_bgr()
                        camera_frames = {}
                        if scene_packet is not None and wrist_bgr is not None:
                            camera_frames = {"scene": scene_packet["bgr"], "wrist": wrist_bgr}
                        if set(camera_frames) == set(pipe.pipelines):
                            observed_at = time.monotonic()
                            worker.submit(dict(frames=camera_frames, observed_at=observed_at,
                                config=control.snapshot(), depth=scene_packet["depth"],
                                semantics=scene_packet["semantics"], instances=scene_packet["instances"],
                                leaf_instances=scene_packet["leaf_instances"],
                                acquisition_id=scene_packet["acquisition_id"],
                                unproject=scene_packet["unproject"]))
                frames += 1
                if test_frames is not None and frames >= test_frames:
                    break
            elif not app_utils.is_playing():
                reset_needed = True
                update_motion()  # paused simulation has no physics callbacks
    except Exception as exc:
        control.set_error(str(exc))
        raise
    finally:
        if control.grasp is not None:
            control.grasp.close()
        SimulationManager.deregister_callback(callback)
        if worker is not None:
            worker.close()
        server.stop()
