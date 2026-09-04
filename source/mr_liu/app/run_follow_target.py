"""Phase 1: SO-101 follows a draggable cube with cuMotion RMPflow."""

from __future__ import annotations

from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view

from mr_liu.config import motion_config, scene_config
from mr_liu.motion.follow import FollowTargetController
from mr_liu.robot.so101 import So101Arm
from mr_liu.sim.spawn import spawn_table_and_so101


def setup(app) -> tuple[FollowTargetController, So101Arm]:
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

    arm = So101Arm()
    app.update()
    arm.validate_joint_mapping()
    print(f"[mr_liu] Articulation: {arm.prim_path}")
    print(f"[mr_liu] USD DOFs: {arm.dof_names}")

    controller = FollowTargetController(arm.articulation, target)
    return controller, arm


def main(app, *, test_frames: int | None = None, device: str | None = None) -> None:
    motion = motion_config()
    SimulationManager.setup_simulation(dt=float(motion["physics_dt"]), device=device or motion["device"])
    controller, _arm = setup(app)
    app.update()
    app_utils.play()
    app.update()
    _arm.apply_configured_pose()
    app.update()

    print("SO-101 follow-target (cuMotion RMPflow, position only)")
    print("  Drag /World/TargetCube in XY/Z. Keep it within ~0.15–0.35 m of the base.")
    print("  Tune configs/motion.yaml (track_orientation, rmp_overrides). Stop/Play resets.")

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
                controller.step(t)
            frames += 1
            if test_frames is not None and frames >= test_frames:
                break
        elif not app_utils.is_playing():
            reset_needed = True
