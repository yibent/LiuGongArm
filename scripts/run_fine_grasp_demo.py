"""One-process Isaac Sim eye-in-hand FineGrasp demo and regression runner."""

from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
import sys
import traceback
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--backend", choices=("geometric", "graspgenx"), default="geometric"
    )
    parser.add_argument("--graspgenx-port", type=int, default=5556)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--perception", choices=("single", "multiview"), default="single")
    parser.add_argument("--planner", choices=("graspmoe", "diffusion"), default="graspmoe")
    parser.add_argument("--segmenter", choices=("depth", "sam2"), default="depth")
    parser.add_argument("--recovery", choices=("off", "assisted", "active"), default="off")
    parser.add_argument("--drop-initial-wrist-frames", type=int, default=0,
                        help="Explicit test-only transient camera fault after the initial diagnostic capture")
    parser.add_argument(
        "--case-json",
        type=Path,
        help="Optional parameterized physical benchmark case (legacy cube by default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Per-run output directory (default: output/fine_grasp_demo)",
    )
    return parser.parse_args()


ARGS = _parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless})

import cv2
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import fine_grasp_config, motion_config, scene_config
from mr_liu.grasp.adapters.isaac_camera import IsaacWristCamera, IsaacSceneCamera, T_world_opencv_from_ros_pose
from mr_liu.grasp.adapters.isaac_gripper import IsaacGripperController
from mr_liu.grasp.adapters.isaac_motion import IsaacCumotionExecutor
from mr_liu.grasp.backends.factory import create_grasp_backend
from mr_liu.grasp.benchmark import (
    BenchmarkCase,
    default_demo_case,
    load_case,
    spawn_benchmark_target,
)
from mr_liu.grasp.contracts import FineGraspRequest, ObjectProperties, TargetSpec
from mr_liu.grasp.debug import DebugSegmenter, FineGraspDebugRecorder
from mr_liu.grasp.node import GeneralGraspNode
from mr_liu.grasp.segmentation import AppearanceDepthTrackerSegmenter, SeededDepthSegmenter
from mr_liu.grasp.identity import GeometricIdentitySegmenter, RemotePromptSegmenter, RobotMaskedSegmenter
from mr_liu.grasp.recovery import RecoveringGraspNode, RecoveryConfig
from mr_liu.grasp.scene_observer import SceneTargetObserver, SceneAssistedVerifier
from mr_liu.grasp.backends.graspgenx import ZmqGraspGenXTransport
from mr_liu.grasp.settings import FineGraspSettings
from mr_liu.grasp.transforms import invert_transform, transform_points
from mr_liu.grasp.verification import VisionGripperVerifier
from mr_liu.perception.camera import spawn_configured_cameras
from mr_liu.robot.so101 import So101Arm
from mr_liu.sim.spawn import spawn_table_and_so101


TARGET_PATH = "/World/FineGraspTarget"
OUTPUT = ARGS.output.resolve() if ARGS.output else ROOT / "output" / "fine_grasp_demo"
CASE: BenchmarkCase = load_case(ARGS.case_json) if ARGS.case_json else default_demo_case()
TARGET_POSITION = CASE.position_m
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
    rigid = spawn_benchmark_target(CASE, TARGET_PATH)
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
    if ARGS.case_json:
        np.random.seed(CASE.seed)
    grasp_cfg = fine_grasp_config().copy()
    grasp_cfg["backend"] = ARGS.backend
    grasp_cfg["backends"]["graspgenx"]["planner"] = ARGS.planner
    if ARGS.perception == "multiview":
        grasp_cfg["fusion"] = {"enabled": True}
        grasp_cfg["active_views"] = {"enabled": True}
        grasp_cfg["selection"]["max_elongated_axis_error_deg"] = 180.0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    error_path = OUTPUT / "error.txt"
    error_path.unlink(missing_ok=True)
    scene = scene_config()
    motion_cfg = motion_config()
    stage_utils.create_new_stage()
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(float(scene["distant_intensity"]) * CASE.light_scale)
    spawn_table_and_so101()
    target_rigid = _spawn_target()
    simulation_app.update()
    # Keep large/tall benchmark objects out of the arm's startup sweep.  They
    # are restored only after the wrist reaches the fine-loop handoff pose.
    target_rigid.set_world_poses(positions=[[0.0, 0.0, 2.0]])
    cameras = spawn_configured_cameras()
    wrist_scene_camera = cameras["wrist"]
    if ARGS.recovery != "off":
        for scene_camera in cameras.values():
            scene_camera.enable_instance_segmentation(leaf_paths=True)
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
    sim_drive_config = grasp_cfg["isaac_sim"]
    _configure_arm_drives(arm, sim_drive_config)
    arm_drive = _arm_drive_report(arm)
    print(f"[mr_liu] Imported SO-101 drive configuration: {json.dumps(imported_arm_drive)}")
    print(f"[mr_liu] Tuned SO-101 drive configuration: {json.dumps(arm_drive)}")
    arm.articulation.set_dof_positions(list(FINE_ENTRY_JOINTS))
    arm.articulation.set_dof_position_targets(list(FINE_ENTRY_JOINTS))
    for _ in range(45):
        simulation_app.update()

    physics_dt = float(motion_cfg["physics_dt"])
    # Motion/gripper control does not need to re-render both RTX cameras for
    # every 60 Hz physics tick.  Fresh images are rendered by camera.capture()
    # at the closed-loop boundary; using physics-only ticks here preserves the
    # eye-in-hand semantics while avoiding hundreds of redundant renders.
    def advance_control_physics() -> None:
        SimulationManager.step(steps=1)

    executor = IsaacCumotionExecutor(
        arm,
        advance_frame=advance_control_physics,
        physics_dt_s=physics_dt,
        target_object_paths=[TARGET_PATH],
        track_orientation=True,
    )
    # The upstream fast loop is expected to hand off 5--8 cm above the target,
    # not at one fixed world Z.  Raise the observation pose for tall objects so
    # an upright bottle or mug is not struck before the first wrist frame.
    extra_entry_height_m = max(0.0, float(CASE.dimensions_m[2]) - 0.040)
    if extra_entry_height_m > 0.0:
        safe_entry = arm.T_base_ee().copy()
        safe_entry[2, 3] += extra_entry_height_m
        if not executor.move_to(safe_entry, speed_scale=0.4):
            raise RuntimeError(
                f"Failed to establish height-adaptive fine-loop entry pose: "
                f"extra_z={extra_entry_height_m:.3f} m"
            )

    yaw_rad = float(np.deg2rad(CASE.yaw_deg))
    target_rigid.set_world_poses(
        positions=[list(CASE.position_m)],
        orientations=[[
            float(np.cos(yaw_rad * 0.5)),
            0.0,
            0.0,
            float(np.sin(yaw_rad * 0.5)),
        ]],
    )
    target_rigid.set_velocities(
        linear_velocities=[[0.0, 0.0, 0.0]],
        angular_velocities=[[0.0, 0.0, 0.0]],
    )
    for _ in range(30):
        simulation_app.update()

    preopen_rgb = wrist_scene_camera.rgb_rgba()
    if preopen_rgb is not None:
        cv2.imwrite(str(OUTPUT / "preopen_wrist.png"), preopen_rgb[:, :, :3][:, :, ::-1])
    preopen_position, preopen_orientation = wrist_scene_camera.world_pose(camera_axes="ros")
    preopen_T_base_camera = T_world_opencv_from_ros_pose(
        np.asarray(preopen_position), np.asarray(preopen_orientation)
    )

    gripper = IsaacGripperController(
        arm.articulation,
        advance_frame=advance_control_physics,
        physics_dt_s=physics_dt,
    )
    if not gripper.open(0.075, speed_mps=0.04):
        raise RuntimeError("Failed to place gripper in observation/open state")
    for _ in range(15):
        simulation_app.update()

    target = TargetSpec(
        object_id="fine_grasp_target",
        semantic_label=CASE.semantic_label if ARGS.case_json else "unseen red cube",
        coarse_position_base_m=TARGET_POSITION,
        properties=ObjectProperties(
            fragile=CASE.fragile,
            thin=CASE.thin,
            reflective=CASE.reflective,
            metadata={"benchmark_case": CASE.name, **dict(CASE.metadata)},
        ),
    )
    camera = IsaacWristCamera(
        wrist_scene_camera,
        ee_pose_provider=arm.T_base_ee,
        advance_frame=simulation_app.update,
        model_seed=CASE.seed,
        robot_mask_provider=(lambda: wrist_scene_camera.instance_mask("/World/SO101", leaf_paths=True))
                            if ARGS.recovery != "off" else None,
    )
    initial_observation = camera.capture(target)
    if initial_observation is None:
        raise RuntimeError("Wrist RGB-D camera did not produce an initial frame")
    cv2.imwrite(
        str(OUTPUT / "initial_wrist.png"), initial_observation.rgb[:, :, ::-1]
    )
    np.savez_compressed(
        OUTPUT / "initial_rgbd.npz",
        depth_m=initial_observation.depth_m,
        rgb=initial_observation.rgb,
        intrinsics=initial_observation.intrinsics.matrix,
        T_base_camera=initial_observation.T_base_camera,
    )
    seed_segmenter = SeededDepthSegmenter(depth_tolerance_m=0.010, max_radius_px=180)
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
    attempt_physics = []
    segmenter = AppearanceDepthTrackerSegmenter(seed_segmenter)
    if ARGS.segmenter == "sam2":
        segmenter = RemotePromptSegmenter(ZmqGraspGenXTransport("127.0.0.1", ARGS.graspgenx_port, 60000),
                                           seed_segmenter)
    if ARGS.perception == "multiview":
        segmenter = GeometricIdentitySegmenter(segmenter)
    recorder = FineGraspDebugRecorder(OUTPUT / "debug")
    segmenter = DebugSegmenter(segmenter, recorder)
    grasp_cfg["backends"]["graspgenx"]["port"] = ARGS.graspgenx_port
    node = GeneralGraspNode(
        camera=camera,
        segmenter=segmenter,
        backend=create_grasp_backend(grasp_cfg),
        motion=executor,
        gripper=gripper,
        verifier=VisionGripperVerifier(),
        settings=FineGraspSettings.from_mapping(grasp_cfg),
        trace=recorder.trace,
    )
    if ARGS.drop_initial_wrist_frames:
        class TransientCameraFault:
            def __init__(self, source, count):
                self.source, self.remaining = source, count
            def capture(self, target):
                if self.remaining > 0:
                    self.remaining -= 1
                    return None
                return self.source.capture(target)
        camera = TransientCameraFault(camera, ARGS.drop_initial_wrist_frames)
        node.camera = camera
    if ARGS.recovery != "off":
        import copy
        external_camera = IsaacSceneCamera(
            cameras["scene"], T_base_world=np.eye(4), advance_frame=simulation_app.update,
            robot_mask_provider=lambda: cameras["scene"].instance_mask("/World/SO101", leaf_paths=True),
        )
        observer = SceneTargetObserver(external_camera,
            table_height_m=grasp_cfg["selection"]["table_height_m"],
            require_robot_mask=True,
            recorder=FineGraspDebugRecorder(OUTPUT / "overhead"))

        def attempt_factory(index, failed, current_target):
            config = copy.deepcopy(grasp_cfg)
            active = index > 0 or ARGS.perception == "multiview"
            config["fusion"] = {"enabled": active}
            config["active_views"] = {"enabled": active}
            if active:
                config["selection"]["max_elongated_axis_error_deg"] = 180.0
            inner = AppearanceDepthTrackerSegmenter(
                SeededDepthSegmenter(depth_tolerance_m=0.010, max_radius_px=180))
            if ARGS.segmenter == "sam2":
                inner = RemotePromptSegmenter(ZmqGraspGenXTransport("127.0.0.1", ARGS.graspgenx_port, 60000),
                                              SeededDepthSegmenter(depth_tolerance_m=0.010, max_radius_px=180))
            if active:
                inner = GeometricIdentitySegmenter(inner)
            inner = RobotMaskedSegmenter(inner, require_mask=True)
            attempt_recorder = FineGraspDebugRecorder(OUTPUT / f"attempt_{index + 1}" / "debug")
            def trace_attempt(event):
                attempt_recorder.trace(event)
                recorder.trace({**event, "attempt": index + 1})
            return GeneralGraspNode(camera=camera, segmenter=DebugSegmenter(inner, attempt_recorder),
                backend=create_grasp_backend(config), motion=executor, gripper=gripper,
                verifier=SceneAssistedVerifier(observer, executor, current_target),
                settings=FineGraspSettings.from_mapping(config), trace=trace_attempt, failed_grasps=failed)

        def recovery_trace(event):
            recorder.trace(event)
            if event.get("event") == "attempt_end":
                # Independent benchmark instrumentation only; never fed to the
                # observer, controller, verifier or retry policy.
                attempt_physics.append({"attempt": event["attempt"],
                    "actual_target_lift_m": _pose(target_xform)[2] - initial_target_position[2]})
        node = RecoveringGraspNode(attempt_factory=attempt_factory, observer=observer,
                                  motion=executor, gripper=gripper, trace=recovery_trace,
                                  config=RecoveryConfig(max_attempts=2 if ARGS.recovery == "active" else 1))
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
        "provenance": {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip(),
            "source_sha256": hashlib.sha256(b"".join(
                p.relative_to(ROOT).as_posix().encode() + p.read_bytes()
                for p in sorted((ROOT / "source/mr_liu/grasp").rglob("*.py")))).hexdigest(),
            "config_sha256": hashlib.sha256(json.dumps(grasp_cfg, sort_keys=True).encode()).hexdigest(),
            "model_seed": CASE.seed,
            "segmenter": ARGS.segmenter,
        },
        "perception_mode": ARGS.perception,
        "recovery_mode": ARGS.recovery,
        "fault_initial_wrist_frames": ARGS.drop_initial_wrist_frames,
        "attempt_results": _jsonable(getattr(node, "attempt_results", [])),
        "attempt_physics": attempt_physics,
        "effective_config": grasp_cfg,
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
    if ARGS.case_json:
        report["benchmark_case"] = CASE.to_dict()
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["result"], indent=2))
    return 0 if result.success else 2


exit_code = 0
try:
    exit_code = main()
except Exception:  # persist all Kit/Python failures for headless regressions
    exit_code = 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    error = traceback.format_exc()
    (OUTPUT / "error.txt").write_text(error, encoding="utf-8")
    print(error, file=sys.stderr, flush=True)
finally:
    # Isaac 6 fast shutdown exits here, before the SystemExit below.
    simulation_app.close(exit_code=exit_code)

raise SystemExit(exit_code)
