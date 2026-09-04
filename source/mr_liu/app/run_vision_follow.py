"""SO-101 follow-target driven by dual-camera Florence FIND + YOLOE TRACK."""

from __future__ import annotations

import numpy as np
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, XformPrim
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view

from mr_liu.config import motion_config, scene_config
from mr_liu.motion.follow import FollowTargetController
from mr_liu.paths import repo_root
from mr_liu.perception.camera import SceneCamera, spawn_configured_cameras
from mr_liu.robot.so101 import So101Arm
from mr_liu.sim.spawn import spawn_table_and_so101
from mr_liu.vision import MultiViewFindTrackPipeline, RuntimeConfig
from mr_liu.vision.control import VisionControlServer, VisionRuntimeControl


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

    vis_cfg = RuntimeConfig(
        prompt=prompt,
        fast_backend=fast,  # type: ignore[arg-type]
        slow_interval=max(0, int(slow_interval)),
    )
    control = VisionRuntimeControl(vis_cfg)
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
        print("  Vision-only mode: arm motion disabled; both camera views remain stable.")
    print(f"  Change targets while running: POST http://{host}:{port}/api/prompt")

    t = 0.0
    reset_needed = True
    frames = 0
    vision_cycles = 0
    try:
        while app.is_running():
            app.update()
            if app_utils.is_playing() and SimulationManager.is_simulating():
                if reset_needed:
                    t = 0.0
                    controller.reset(t)
                    reset_needed = False
                else:
                    t += physics_dt
                    if pipe is not None:
                        camera_frames = {
                            name: frame
                            for name, camera in cameras.items()
                            if (frame := camera.rgb_bgr()) is not None
                        }
                        if set(camera_frames) == set(pipe.pipelines):
                            current_cfg = control.snapshot()
                            results = pipe.process_frames(camera_frames, current_cfg)
                            for view, result in results.items():
                                control.publish(view, result)
                                stats = result.stats
                                if stats.path == "slow" or vision_cycles % 30 == 0:
                                    print(
                                        f"DUAL_TRACK view={view} frame={stats.frame_idx} "
                                        f"path={stats.path} prompt={stats.prompt!r} "
                                        f"find={stats.n_slow} track={stats.n_fast} "
                                        f"slow_ms={stats.slow_ms:.1f} fast_ms={stats.fast_ms:.1f}"
                                    )

                            scene_result = results["scene"]
                            detections = scene_result.fast or scene_result.slow
                            if follow_target and detections:
                                height, width = camera_frames["scene"].shape[:2]
                                x, y = _bbox_to_table_xy(detections[0].xyxy, width, height)
                                target_xform.set_world_poses(positions=[[x, y, table_z]])
                            vision_cycles += 1
                    if follow_target:
                        controller.step(t)
                frames += 1
                if test_frames is not None and frames >= test_frames:
                    break
            elif not app_utils.is_playing():
                reset_needed = True
    except Exception as exc:
        control.set_error(str(exc))
        raise
    finally:
        server.stop()
