"""SO-101 follow-target driven by YOLOE / Florence FIND+TRACK on the scene camera."""

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
from mr_liu.vision import FindTrackPipeline, RuntimeConfig


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


def setup(app) -> tuple[FollowTargetController, So101Arm, GeomPrim, SceneCamera]:
    scene = scene_config()
    stage_utils.create_new_stage()
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
    camera = cameras["scene"]
    app.update()

    arm = So101Arm()
    app.update()
    arm.validate_joint_mapping()
    print(f"[mr_liu] Articulation: {arm.prim_path}")
    print(f"[mr_liu] USD DOFs: {arm.dof_names}")

    controller = FollowTargetController(arm.articulation, target)
    return controller, arm, target, camera


def _try_load_pipeline(prompt: str, fast: str) -> tuple[FindTrackPipeline | None, RuntimeConfig]:
    cfg = RuntimeConfig(prompt=prompt, fast_backend=fast)  # type: ignore[arg-type]
    from find_and_track.settings import default_yoloe

    weights = default_yoloe()
    if fast == "yoloe" and not weights.is_file():
        print(f"[mr_liu] YOLOE weights missing ({weights}); vision loop disabled, cube is still draggable.")
        return None, cfg
    pipe = FindTrackPipeline(yoloe_weights=weights)
    try:
        pipe.load(cfg.fast_backend)
    except Exception as exc:  # noqa: BLE001
        print(f"[mr_liu] Vision models failed to load: {exc}")
        return None, cfg
    print(f"[mr_liu] Vision ready prompt={prompt!r} fast={fast}")
    return pipe, cfg


def main(
    app,
    *,
    test_frames: int | None = None,
    device: str | None = None,
    prompt: str = "cube",
    fast: str = "yoloe",
) -> None:
    motion = motion_config()
    SimulationManager.setup_simulation(dt=float(motion["physics_dt"]), device=device or motion["device"])
    controller, arm, target, camera = setup(app)
    app.update()
    app_utils.play()
    app.update()
    arm.apply_configured_pose()
    app.update()

    pipe, vis_cfg = _try_load_pipeline(prompt, fast)
    target_xform = XformPrim(scene_config()["target"]["prim_path"])
    table_z = float(scene_config()["target"]["position"][2])

    print("SO-101 vision follow-target (cuMotion RMPflow)")
    print("  Scene camera → FIND/TRACK → /World/TargetCube → arm.")
    print("  Drag the cube manually if vision is off. Stop/Play resets the controller.")

    physics_dt = float(motion["physics_dt"])
    t = 0.0
    reset_needed = True
    frames = 0
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
                    frame = camera.rgb_bgr()
                    if frame is not None:
                        result = pipe.process_frame(frame, vis_cfg)
                        dets = result.fast or result.slow
                        if dets:
                            h, w = frame.shape[:2]
                            x, y = _bbox_to_table_xy(dets[0].xyxy, w, h)
                            target_xform.set_world_poses(positions=[[x, y, table_z]])
                controller.step(t)
            frames += 1
            if test_frames is not None and frames >= test_frames:
                break
        elif not app_utils.is_playing():
            reset_needed = True
