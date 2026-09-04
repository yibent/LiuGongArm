"""One-process Isaac Sim eye-in-hand FineGrasp demo and regression runner."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=("geometric",), default="geometric")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


ARGS = _parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless})

import cv2
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import fine_grasp_config, motion_config, scene_config
from mr_liu.grasp.adapters.isaac_camera import IsaacWristCamera, T_world_opencv_from_ros_pose
from mr_liu.grasp.adapters.isaac_gripper import IsaacGripperController
from mr_liu.grasp.adapters.isaac_motion import IsaacCumotionExecutor
from mr_liu.grasp.backends.geometric import GeometricAntipodalBackend
from mr_liu.grasp.contracts import FineGraspRequest, ObjectProperties, TargetSpec
from mr_liu.grasp.debug import DebugSegmenter, FineGraspDebugRecorder
from mr_liu.grasp.node import GeneralGraspNode
from mr_liu.grasp.segmentation import AppearanceDepthTrackerSegmenter, SeededDepthSegmenter
from mr_liu.grasp.settings import FineGraspSettings
from mr_liu.grasp.transforms import invert_transform, transform_points
from mr_liu.grasp.verification import VisionGripperVerifier
from mr_liu.perception.camera import spawn_configured_cameras
from mr_liu.robot.so101 import So101Arm
from mr_liu.sim.spawn import spawn_table_and_so101


TARGET_PATH = "/World/FineGraspTarget"
OUTPUT = ROOT / "output" / "fine_grasp_demo"
TARGET_POSITION = (0.324, -0.198, 1.068)
# A collision-free handoff pose found from the SO-101 URDF kinematics. It puts
# the wrist optical axis about 10 cm above the target and nearly normal to the
# tabletop, matching the intended fast-loop -> fine-loop handoff condition.
FINE_ENTRY_JOINTS = (-0.5408, -0.5510, 0.8902, 0.8826, -0.0982, 0.0)


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _tag_target() -> None:
    import Semantics

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(TARGET_PATH)
    api = Semantics.SemanticsAPI.Apply(prim, "class")
    api.CreateSemanticTypeAttr().Set("class")
    api.CreateSemanticDataAttr().Set("fine_grasp_target")


def _spawn_target() -> RigidPrim:
    # Default wrist optical axis meets the tabletop near this point, keeping
    # the initial observation inside the eye-in-hand camera's close-range FOV.
    Cube(
        TARGET_PATH,
        positions=[list(TARGET_POSITION)],
        sizes=0.036,
        colors=(0.85, 0.12, 0.08),
    )
    GeomPrim(TARGET_PATH, apply_collision_apis=True)
    rigid = RigidPrim(TARGET_PATH, masses=[0.035])
    _tag_target()
    return rigid


def _pose(prim: XformPrim) -> list[float]:
    positions, _orientations = prim.get_world_poses()
    if hasattr(positions, "numpy"):
        positions = positions.numpy()
    return np.asarray(positions, dtype=np.float64).reshape(-1, 3)[0].tolist()


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _arm_drive_report(arm: So101Arm) -> dict[str, object]:
    """Capture the active PhysX drive configuration for reproducible tuning."""
    stiffness, damping = arm.articulation.get_dof_gains()
    return {
        "dof_names": list(arm.articulation.dof_names),
        "stiffness": _numpy(stiffness).reshape(-1).tolist(),
        "damping": _numpy(damping).reshape(-1).tolist(),
        "max_effort": _numpy(arm.articulation.get_dof_max_efforts()).reshape(-1).tolist(),
        "max_velocity": _numpy(arm.articulation.get_dof_max_velocities()).reshape(-1).tolist(),
        "drive_type": _jsonable(arm.articulation.get_dof_drive_types()),
    }


def _configure_arm_drives(arm: So101Arm, config: dict[str, object]) -> None:
    """Tune only the five arm drives; the gripper keeps its force controller."""
    names = list(arm.articulation.dof_names)
    arm_indices = [index for index, name in enumerate(names) if name != "gripper"]
    count = len(arm_indices)
    arm.articulation.set_dof_gains(
        stiffnesses=[float(config["arm_drive_stiffness_nm_per_rad"])] * count,
        dampings=[float(config["arm_drive_damping_nms_per_rad"])] * count,
        dof_indices=arm_indices,
    )
    arm.articulation.set_dof_max_efforts(
        [float(config["arm_drive_max_effort_nm"])] * count,
        dof_indices=arm_indices,
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    error_path = OUTPUT / "error.txt"
    error_path.unlink(missing_ok=True)
    scene = scene_config()
    motion_cfg = motion_config()
    stage_utils.create_new_stage()
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(float(scene["distant_intensity"]))
    spawn_table_and_so101()
    target_rigid = _spawn_target()
    simulation_app.update()
    cameras = spawn_configured_cameras()
    wrist_scene_camera = cameras["wrist"]
    SimulationManager.setup_simulation(
        # CPU PhysX keeps articulated USD transforms synchronized with RTX
        # sensors. cuMotion itself still runs on CUDA. GPU/Fabric mode requires
        # an explicit fabric-to-USD bridge or the eye-in-hand pose can be stale.
        dt=float(motion_cfg["physics_dt"]),
        device="cpu",
    )
    arm = So101Arm()
    simulation_app.update()
    app_utils.play()
    for _ in range(30):
        simulation_app.update()
    imported_arm_drive = _arm_drive_report(arm)
    sim_drive_config = fine_grasp_config()["isaac_sim"]
    _configure_arm_drives(arm, sim_drive_config)
    arm_drive = _arm_drive_report(arm)
    print(f"[mr_liu] Imported SO-101 drive configuration: {json.dumps(imported_arm_drive)}")
    print(f"[mr_liu] Tuned SO-101 drive configuration: {json.dumps(arm_drive)}")
    arm.articulation.set_dof_positions(list(FINE_ENTRY_JOINTS))
    arm.articulation.set_dof_position_targets(list(FINE_ENTRY_JOINTS))
    for _ in range(45):
        simulation_app.update()

    preopen_rgb = wrist_scene_camera.rgb_rgba()
    if preopen_rgb is not None:
        cv2.imwrite(str(OUTPUT / "preopen_wrist.png"), preopen_rgb[:, :, :3][:, :, ::-1])
    preopen_position, preopen_orientation = wrist_scene_camera.world_pose(camera_axes="ros")
    preopen_T_base_camera = T_world_opencv_from_ros_pose(
        np.asarray(preopen_position), np.asarray(preopen_orientation)
    )

    physics_dt = float(motion_cfg["physics_dt"])
    gripper = IsaacGripperController(
        arm.articulation,
        advance_frame=simulation_app.update,
        physics_dt_s=physics_dt,
    )
    if not gripper.open(0.075, speed_mps=0.04):
        raise RuntimeError("Failed to place gripper in observation/open state")
    for _ in range(15):
        simulation_app.update()

    target = TargetSpec(
        object_id="fine_grasp_target",
        semantic_label="unseen red cube",
        coarse_position_base_m=TARGET_POSITION,
        properties=ObjectProperties(),
    )
    camera = IsaacWristCamera(
        wrist_scene_camera,
        ee_pose_provider=arm.T_base_ee,
        advance_frame=simulation_app.update,
    )
    initial_observation = camera.capture(target)
    if initial_observation is None:
        raise RuntimeError("Wrist RGB-D camera did not produce an initial frame")
    cv2.imwrite(
        str(OUTPUT / "initial_wrist.png"), initial_observation.rgb[:, :, ::-1]
    )
    seed_segmenter = SeededDepthSegmenter(depth_tolerance_m=0.018, max_radius_px=180)
    initial_mask = seed_segmenter.segment(initial_observation, target)
    if initial_mask is None or not initial_mask.any():
        actual_target = _pose(XformPrim(TARGET_PATH))
        point_camera = transform_points(
            invert_transform(initial_observation.T_base_camera),
            np.asarray(actual_target, dtype=np.float64).reshape(1, 3),
        )[0]
        joint_positions = arm.articulation.get_dof_positions()
        if hasattr(joint_positions, "numpy"):
            joint_positions = joint_positions.numpy()
        raise RuntimeError(
            "Seeded local RGB-D segmentation could not find the target; "
            f"actual_target={actual_target}, point_camera={point_camera.tolist()}, "
            f"T_base_camera={initial_observation.T_base_camera.round(5).tolist()}, "
            f"preopen_T_base_camera={preopen_T_base_camera.round(5).tolist()}, "
            f"T_base_ee={arm.T_base_ee().round(5).tolist()}, "
            f"joints={np.asarray(joint_positions).reshape(-1).round(5).tolist()}, "
            f"depth_finite={int(np.isfinite(initial_observation.depth_m).sum())}"
        )
    cv2.imwrite(
        str(OUTPUT / "initial_mask.png"), initial_mask.astype(np.uint8) * 255
    )

    target_xform = XformPrim(TARGET_PATH)
    initial_target_position = _pose(target_xform)
    segmenter = AppearanceDepthTrackerSegmenter(seed_segmenter)
    recorder = FineGraspDebugRecorder(OUTPUT / "debug")
    segmenter = DebugSegmenter(segmenter, recorder)
    executor = IsaacCumotionExecutor(
        arm,
        advance_frame=simulation_app.update,
        physics_dt_s=physics_dt,
        target_object_paths=[TARGET_PATH],
        track_orientation=True,
    )
    node = GeneralGraspNode(
        camera=camera,
        segmenter=segmenter,
        backend=GeometricAntipodalBackend(),
        motion=executor,
        gripper=gripper,
        verifier=VisionGripperVerifier(),
        settings=FineGraspSettings.from_mapping(fine_grasp_config()),
        trace=recorder.trace,
    )
    result = node.execute(
        FineGraspRequest(target=target, request_id="isaac-demo", dry_run=ARGS.dry_run)
    )
    final_observation = camera.capture(target)
    if final_observation is not None:
        cv2.imwrite(str(OUTPUT / "final_wrist.png"), final_observation.rgb[:, :, ::-1])
    final_target_position = _pose(target_xform)
    final_joint_positions = _numpy(arm.articulation.get_dof_positions()).reshape(-1)
    masses = target_rigid.get_masses()
    if hasattr(masses, "numpy"):
        masses = masses.numpy()
    report = {
        "result": _jsonable(result),
        "initial_target_position_m": initial_target_position,
        "final_target_position_m": final_target_position,
        "actual_target_lift_m": final_target_position[2] - initial_target_position[2],
        "final_gripper": _jsonable(gripper.state()),
        "final_T_base_ee": arm.T_base_ee().tolist(),
        "final_joint_positions_rad": final_joint_positions.tolist(),
        "final_joint_efforts": _numpy(arm.articulation.get_dof_efforts()).reshape(-1).tolist(),
        "kinematic_diagnostics": executor.kinematic_diagnostics(),
        "trace": _jsonable(recorder.events),
        "target_mass_kg": float(np.asarray(masses).reshape(-1)[0]),
        "arm_drive": arm_drive,
        "imported_arm_drive": imported_arm_drive,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["result"], indent=2))
    return 0 if result.success else 2


exit_code = 0
try:
    exit_code = main()
except Exception:  # persist all Kit/Python failures for headless regressions
    OUTPUT.mkdir(parents=True, exist_ok=True)
    error = traceback.format_exc()
    (OUTPUT / "error.txt").write_text(error, encoding="utf-8")
    print(error, file=sys.stderr)
    raise
finally:
    simulation_app.close()

raise SystemExit(exit_code)
