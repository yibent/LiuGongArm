"""One-process Isaac Sim eye-in-hand FineGrasp demo and regression runner."""

from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-open", action="store_true",
                        help="Keep the GUI open with physics paused after the test")
    parser.add_argument("--start-delay-s", type=float, default=0.,
                        help="Pause before the grasp test so the viewer can get ready")
    parser.add_argument("--record-video", action="store_true",
                        help="Record overview + two RGB-D RGB panels to demo.mp4 (adds render cost)")
    parser.add_argument(
        "--backend", choices=("geometric", "graspgenx", "m2t2"), default="geometric"
    )
    parser.add_argument("--graspgenx-port", type=int, default=5556)
    parser.add_argument("--m2t2-port", type=int, default=5562)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--label", help="Opt-in label -> external RGB-D/LK coarse approach -> wrist FineGrasp")
    parser.add_argument("--locator-port", type=int, default=5570)
    parser.add_argument("--localization-mode", choices=("florence", "florence_yoloe"), default="florence_yoloe")
    parser.add_argument("--coarse-only", action="store_true", help="Test label approach and wrist handoff without closing")
    parser.add_argument("--wrist-camera-profile", choices=("fine_grasp", "tabletop_wide"), default="fine_grasp")
    parser.add_argument("--test-coarse-shift-m", type=float, default=0.,
                        help="Simulation-only world-X perturbation during label coarse transit (<=5 cm)")
    parser.add_argument("--perception", choices=("single", "multiview"), default="single")
    parser.add_argument("--planner", choices=("graspmoe", "diffusion"), default="graspmoe")
    parser.add_argument("--segmenter", choices=("depth", "sam2"), default="depth")
    parser.add_argument("--recovery", choices=("off", "assisted", "active"), default="off")
    parser.add_argument("--scene-view", choices=("overhead", "oblique"), default="overhead")
    parser.add_argument("--drop-initial-wrist-frames", type=int, default=0,
                        help="Explicit test-only transient camera fault after the initial diagnostic capture")
    parser.add_argument("--test-target-shift-m", type=float, default=0.,
                        help="Simulation-only world-X target perturbation before first close (<= 5 cm)")
    parser.add_argument(
        "--case-json",
        type=Path,
        help="Optional parameterized physical benchmark case (legacy cube by default)",
    )
    parser.add_argument(
        "--clutter-case-json",
        type=Path,
        help="Optional deterministic clutter/collision scene JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Per-run output directory (default: output/fine_grasp_demo)",
    )
    args = parser.parse_args()
    if args.coarse_only and not args.label:
        parser.error("--coarse-only requires --label")
    if args.test_coarse_shift_m and (not args.label or not abs(args.test_coarse_shift_m) <= .05):
        parser.error("--test-coarse-shift-m requires --label and magnitude <=5 cm")
    if (args.keep_open or args.start_delay_s) and args.headless:
        parser.error("--keep-open and --start-delay-s require --no-headless")
    if not 0. <= args.start_delay_s <= 120.:
        parser.error("--start-delay-s must be between 0 and 120")
    return args


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
from mr_liu.grasp.contracts import FineGraspRequest, ObjectProperties, PlaceRequest, TargetSpec
from mr_liu.grasp.debug import DebugSegmenter, FineGraspDebugRecorder
from mr_liu.grasp.node import GeneralGraspNode
from mr_liu.grasp.segmentation import AppearanceDepthTrackerSegmenter, SeededDepthSegmenter
from mr_liu.grasp.identity import GeometricIdentitySegmenter, RemotePromptSegmenter, RobotMaskedSegmenter
from mr_liu.grasp.recovery import RecoveringGraspNode, RecoveryConfig
from mr_liu.grasp.scene_observer import SceneTargetObserver, SceneAssistedVerifier
from mr_liu.grasp.backends.graspgenx import ZmqGraspGenXTransport
from mr_liu.grasp.backends.m2t2 import M2T2Config, M2T2PlacementPlanner
from mr_liu.grasp.settings import FineGraspSettings
from mr_liu.grasp.transforms import invert_transform, transform_points, look_at_opencv, matrix_to_quaternion_wxyz
from mr_liu.grasp.verification import VisionGripperVerifier
from mr_liu.grasp.place import PickPlaceController
from mr_liu.perception.camera import spawn_configured_cameras
from mr_liu.robot.so101 import So101Arm
from mr_liu.sim.spawn import spawn_table_and_so101
from mr_liu.sim.clutter import load_scene, spawn_clutter_scene
from mr_liu.sim.grasp_faults import OneShotPreCloseShift, OneShotCoarseShift


TARGET_PATH = "/World/FineGraspTarget"
OUTPUT = ARGS.output.resolve() if ARGS.output else ROOT / "output" / "fine_grasp_demo"
CASE: BenchmarkCase = load_case(ARGS.case_json) if ARGS.case_json else default_demo_case()
CLUTTER_SCENE = load_scene(ARGS.clutter_case_json) if ARGS.clutter_case_json else None
TARGET_POSITION = CASE.position_m
# A collision-free handoff pose found from the SO-101 URDF kinematics. It puts
# the wrist optical axis about 10 cm above the target and nearly normal to the
# tabletop, matching the intended fast-loop -> fine-loop handoff condition.
FINE_ENTRY_JOINTS = (-0.5408, -0.5510, 0.8902, 0.8826, -0.0982, 0.0)
VIDEO = None


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
    global VIDEO
    if ARGS.case_json:
        np.random.seed(CASE.seed)
    grasp_cfg = fine_grasp_config().copy()
    grasp_cfg["backend"] = ARGS.backend
    grasp_cfg["backends"]["graspgenx"]["planner"] = ARGS.planner
    grasp_cfg["backends"]["m2t2"]["port"] = ARGS.m2t2_port
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
    # Isolated development fixture, optionally augmented with a deterministic
    # clutter layout whose bodies are included in the cuMotion world.
    spawn_table_and_so101(include_tabletop_props=False)
    clutter_paths = spawn_clutter_scene(CLUTTER_SCENE) if CLUTTER_SCENE is not None else []
    target_rigid = _spawn_target()
    simulation_app.update()
    # Keep large/tall benchmark objects out of the arm's startup sweep.  They
    # are restored only after the wrist reaches the fine-loop handoff pose.
    target_rigid.set_world_poses(positions=[[0.0, 0.0, 2.0]])
    cameras = spawn_configured_cameras(profiles={"wrist": ARGS.wrist_camera_profile})
    if ARGS.scene_view == "oblique":
        mount = cameras["scene"].config["recovery_oblique"]
        pose = look_at_opencv(mount["position"], mount["target"])
        cameras["scene"]._cam.set_world_pose(position=pose[:3, 3],
            orientation=matrix_to_quaternion_wxyz(pose[:3, :3]), camera_axes="ros")
    wrist_scene_camera = cameras["wrist"]
    overview_camera = None
    if ARGS.record_video:
        from isaacsim.sensors.camera import Camera
        from mr_liu.sim.demo_video import DemoVideoRecorder
        # Recording-only camera, deliberately excluded from the sensor registry
        # passed to the controller. Never used for segmentation or verification.
        overview_camera = Camera(prim_path="/World/DemoOnly/Overview", frequency=60,
                                 resolution=(832, 624))
        overview_camera.initialize()
        overview_camera.set_focal_length(1.8)
        overview_camera.set_clipping_range(0.01, 10.)
        overview_pose = look_at_opencv([0.65, -0.90, 1.50], [0.22, -0.10, 1.18])
        overview_camera.set_world_pose(position=overview_pose[:3, 3],
            orientation=matrix_to_quaternion_wxyz(overview_pose[:3, :3]), camera_axes="ros")
        VIDEO = DemoVideoRecorder(OUTPUT / "demo.mp4", CASE.name, ARGS.test_target_shift_m,
                                  coarse_fault_m=ARGS.test_coarse_shift_m)
    recording_active = False
    control_ticks = 0
    semantic_tracker = None
    coarse_report = None

    def capture_video():
        if VIDEO is not None and recording_active:
            scene_rgb = cameras["scene"].rgb_rgba()
            if semantic_tracker is not None and semantic_tracker.latest is not None and scene_rgb is not None:
                scene_rgb = scene_rgb.copy()
                ys, xs = np.nonzero(semantic_tracker.latest.mask)
                if len(xs):
                    cv2.rectangle(scene_rgb, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 100, 255), 2)
            VIDEO.capture(overview_camera.get_rgba(), wrist_scene_camera.rgb_rgba(),
                          scene_rgb)

    def advance_observation_frame():
        simulation_app.update()
        capture_video()
    if ARGS.recovery != "off" or ARGS.label:
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
    if ARGS.label:
        # Scene initialization only: a generic parked pose, independent of
        # target pose/shape. All subsequent transit uses measured RGB-D + IK.
        arm.apply_configured_pose()
    else:
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
        nonlocal control_ticks
        control_ticks += 1
        if recording_active and control_ticks % 4 == 0:
            # Replace (not add to) this physics tick with a rendered app tick,
            # so closing/lifting/sideways motion is visible between observations.
            advance_observation_frame()
        else:
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
    if extra_entry_height_m > 0.0 and not ARGS.label:
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
        semantic_label=ARGS.label or (CASE.semantic_label if ARGS.case_json else "unseen red cube"),
        coarse_position_base_m=None if ARGS.label else TARGET_POSITION,
        properties=ObjectProperties(
            fragile=CASE.fragile,
            thin=CASE.thin,
            reflective=CASE.reflective,
            metadata={"benchmark_case": CASE.name, **dict(CASE.metadata)},
        ),
    )
    if ARGS.label:
        from mr_liu.control.coarse_approach import CoarseApproach
        from mr_liu.perception.semantic_target import LocalSemanticLocator, SemanticFlowTarget
        coarse_recorder = FineGraspDebugRecorder(OUTPUT / "coarse" / "debug")
        def coarse_physical_shift(delta):
            # Test instrument only. No truth/hint updates to the controller.
            position = np.asarray(_pose(XformPrim(TARGET_PATH)), float) + np.asarray(delta)
            target_rigid.set_world_poses(positions=[position])
            target_rigid.set_velocities(linear_velocities=[[0., 0., 0.]], angular_velocities=[[0., 0., 0.]])
        coarse_fault = OneShotCoarseShift(ARGS.test_coarse_shift_m, coarse_physical_shift)
        def coarse_trace(event):
            coarse_recorder.trace(event)
            print(f"[LabelGrasp] {json.dumps(_jsonable(event))}", flush=True)
            if VIDEO is not None:
                VIDEO.trace(event)
            injected = coarse_fault.on_event(event)
            if injected is not None:
                coarse_recorder.trace(injected)
        external = IsaacSceneCamera(cameras["scene"], T_base_world=np.eye(4),
            advance_frame=advance_observation_frame,
            robot_mask_provider=lambda: cameras["scene"].instance_mask("/World/SO101", leaf_paths=True))
        semantic_tracker = SemanticFlowTarget(external,
            LocalSemanticLocator(ARGS.locator_port, ARGS.localization_mode), target,
            grasp_cfg["selection"]["table_height_m"], trace=coarse_trace, recorder=coarse_recorder)
        coarse = CoarseApproach(semantic_tracker, executor, trace=coarse_trace)
        recording_active = ARGS.record_video
        capture_video()
        coarse_report = {"label": ARGS.label, "localization_mode": ARGS.localization_mode,
            "wrist_camera_profile": ARGS.wrist_camera_profile,
            "camera_configs": {name: _jsonable(cam.config) for name, cam in cameras.items()},
            "target_pose_source": "RGB-D pixels only; no benchmark target pose supplied to controller",
            "tracker": "pyramidal LK with forward/backward check; no MIL fallback",
            "passed": False}
        try:
            target = coarse.run()
            coarse_report.update(motion_passed=True, target=_jsonable(target))
        except Exception as exc:
            coarse_report["failure"] = str(exc)
            coarse_trace({"phase": "failed", "event": "coarse_failed", "reason": str(exc)})
            raise
        finally:
            executor.clear_grasp_plan()
            coarse_report.update(find_count=semantic_tracker.find_count, flow_count=semantic_tracker.flow_count,
                                 moves=coarse.move_count, guard_checks=coarse.guard_count,
                                 test_coarse_shift_m=ARGS.test_coarse_shift_m,
                                 test_coarse_shift_applied=coarse_fault.applied,
                                 trace=_jsonable(coarse_recorder.events))
            (OUTPUT / "coarse_report.json").write_text(json.dumps(coarse_report, indent=2), encoding="utf-8")
    camera = IsaacWristCamera(
        wrist_scene_camera,
        ee_pose_provider=arm.T_base_ee,
        advance_frame=advance_observation_frame,
        model_seed=CASE.seed,
        robot_mask_provider=(lambda: wrist_scene_camera.instance_mask("/World/SO101", leaf_paths=True))
                            if ARGS.recovery != "off" or ARGS.label else None,
    )
    initial_observation = camera.capture(target)
    if initial_observation is None:
        raise RuntimeError("Wrist RGB-D camera did not produce an initial frame")
    place_scene_camera = None
    place_observer = None
    if ARGS.backend == "m2t2":
        place_scene_camera = IsaacSceneCamera(
            cameras["scene"], T_base_world=np.eye(4),
            advance_frame=advance_observation_frame,
            robot_mask_provider=lambda: cameras["scene"].instance_mask("/World/SO101", leaf_paths=True),
        )
        place_observer = SceneTargetObserver(
            place_scene_camera,
            table_height_m=grasp_cfg["selection"]["table_height_m"],
            require_robot_mask=True,
            recorder=FineGraspDebugRecorder(OUTPUT / "placement" / "overhead"),
        )
        # Establish an appearance anchor before the object is lifted.  This
        # observation is evidence only and is never used as a pose hint.
        place_observer.observe(target)
    cv2.imwrite(
        str(OUTPUT / "initial_wrist.png"), initial_observation.rgb[:, :, ::-1]
    )
    np.savez_compressed(
        OUTPUT / "initial_rgbd.npz",
        depth_m=initial_observation.depth_m,
        rgb=initial_observation.rgb,
        intrinsics=initial_observation.intrinsics.matrix,
        T_base_camera=initial_observation.T_base_camera,
        T_base_ee=initial_observation.T_base_ee,
        T_ee_camera=initial_observation.T_ee_camera,
    )
    seed_segmenter = SeededDepthSegmenter(depth_tolerance_m=0.010, max_radius_px=180)
    initial_mask = seed_segmenter.segment(initial_observation, target)
    if (initial_mask is None or not initial_mask.any()) and ARGS.recovery == "off":
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
        str(OUTPUT / "initial_mask.png"), (np.zeros_like(initial_observation.depth_m, np.uint8)
                                         if initial_mask is None else initial_mask.astype(np.uint8) * 255)
    )
    if ARGS.label:
        coarse_report["wrist_handoff_mask_pixels"] = 0 if initial_mask is None else int(initial_mask.sum())
        coarse_report["wrist_handoff_sequence"] = initial_observation.sequence
        coarse_report["wrist_optical_axis_base"] = initial_observation.T_base_camera[:3, 2].tolist()
        coarse_report["wrist_angle_from_base_down_deg"] = float(np.degrees(np.arccos(
            np.clip(-initial_observation.T_base_camera[2, 2], -1., 1.))))
        try:
            coarse_report["wrist_handoff"] = semantic_tracker.validate_wrist_handoff(initial_observation, initial_mask)
            coarse_report["passed"] = True
        except Exception as exc:
            coarse_report.update(passed=False, failure=str(exc))
            raise
        finally:
            (OUTPUT / "coarse_report.json").write_text(json.dumps(coarse_report, indent=2), encoding="utf-8")
        if ARGS.coarse_only:
            if VIDEO is not None:
                VIDEO.finish(overview_camera.get_rgba(), wrist_scene_camera.rgb_rgba(),
                             cameras["scene"].rgb_rgba(), "标签接近与腕部交接通过（未执行抓取）")
            return 0

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
    def shift_sim_target(delta):
        # Test instrument changes the physical scene only, never target hints,
        # candidates, camera evidence, verifier output or recovery decisions.
        position = np.asarray(_pose(target_xform), dtype=float) + np.asarray(delta)
        target_rigid.set_world_poses(positions=[position])
        target_rigid.set_velocities(linear_velocities=[[0., 0., 0.]], angular_velocities=[[0., 0., 0.]])
    fault = OneShotPreCloseShift(ARGS.test_target_shift_m, shift_sim_target)
    def trace_event(event):
        recorder.trace(event)
        if VIDEO is not None:
            VIDEO.trace(event)
        injected = fault.on_event(event)
        if injected is not None:
            recorder.trace(injected)
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
        trace=trace_event,
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
            cameras["scene"], T_base_world=np.eye(4), advance_frame=advance_observation_frame,
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
                trace_event({**event, "attempt": index + 1})
            return GeneralGraspNode(camera=camera, segmenter=DebugSegmenter(inner, attempt_recorder),
                backend=create_grasp_backend(config), motion=executor, gripper=gripper,
                verifier=SceneAssistedVerifier(observer, executor, current_target),
                settings=FineGraspSettings.from_mapping(config), trace=trace_attempt, failed_grasps=failed)

        def recovery_trace(event):
            recorder.trace(event)
            if VIDEO is not None:
                VIDEO.trace(event)
            if event.get("event") == "attempt_end":
                # Independent benchmark instrumentation only; never fed to the
                # observer, controller, verifier or retry policy.
                attempt_physics.append({"attempt": event["attempt"],
                    "actual_target_lift_m": _pose(target_xform)[2] - initial_target_position[2]})
        node = RecoveringGraspNode(attempt_factory=attempt_factory, observer=observer,
                                  motion=executor, gripper=gripper, trace=recovery_trace,
                                  config=RecoveryConfig(max_attempts=2 if ARGS.recovery == "active" else 1))
    if not ARGS.headless:
        from isaacsim.core.utils.viewports import set_camera_view
        # Presentation camera only: never used as a perception input.
        set_camera_view(eye=[0.95, -0.95, 1.65], target=[0.22, -0.10, 1.18],
                        camera_prim_path="/OmniverseKit_Persp")
    if ARGS.start_delay_s:
        app_utils.pause()
        print(f"[BusAgent] Grasp starts in {ARGS.start_delay_s:g} seconds", flush=True)
        deadline = time.monotonic() + ARGS.start_delay_s
        while simulation_app.is_running() and time.monotonic() < deadline:
            simulation_app.update()
            time.sleep(0.01)
        if not simulation_app.is_running():
            return 1
        app_utils.play()
    print("[BusAgent] Starting closed-loop grasp test", flush=True)
    recording_active = ARGS.record_video
    capture_video()
    result = node.execute(
        FineGraspRequest(target=target, request_id="isaac-demo", dry_run=ARGS.dry_run)
    )
    place_result = None
    if result.success and ARGS.backend == "m2t2" and not ARGS.dry_run:
        from dataclasses import replace
        from mr_liu.grasp.contracts import FailureCode, FineGraspPhase, PickPlacePhase, PlaceResult
        observer = getattr(node, "observer", None) or place_observer
        scene_camera = place_scene_camera
        if observer is None or scene_camera is None:
            place_result = PlaceResult(False, PickPlacePhase.FAILED,
                                       FailureCode.PLACE_VERIFICATION_FAILED,
                                       "放置验证相机未初始化。")
        else:
            evidence = observer.observe(target, expected_center=executor.robot_state().T_base_ee[:3, 3])
            if evidence is None:
                evidence = observer.latest
            if evidence is None:
                place_result = PlaceResult(False, PickPlacePhase.FAILED,
                                           FailureCode.PLACE_VERIFICATION_FAILED,
                                           "抬升后没有获得固定相机持物证据。")
            else:
                section = grasp_cfg["backends"]["m2t2"]
                backend_for_place = getattr(node, "backend", None)
                while hasattr(backend_for_place, "primary"):
                    backend_for_place = backend_for_place.primary
                planner = M2T2PlacementPlanner(
                    M2T2Config.from_mapping(section),
                    transport=getattr(backend_for_place, "transport", None),
                )
                place_cfg = grasp_cfg.get("place", {})
                place_request = PlaceRequest(
                    center_base_m=tuple(place_cfg.get("target_center_base_m", [0.68, 0., 1.05])),
                    size_xy_m=tuple(place_cfg.get("target_size_xy_m", [0.14, 0.14])),
                    surface_z_m=place_cfg.get("surface_z_m"), object_id=target.object_id,
                )
                def placement_center(current_target, req):
                    found = observer.observe(current_target, expected_center=np.asarray(req.center_base_m, dtype=float))
                    return None if found is None else found.center_base_m
                controller = PickPlaceController(
                    motion=executor, gripper=gripper, planner=planner.generate,
                    camera=scene_camera, target=target, observer=placement_center,
                    max_width_m=float(place_cfg.get("max_width_m", 0.075)),
                    speed_scale=float(place_cfg.get("speed_scale", 0.35)),
                    release_speed_mps=float(place_cfg.get("release_speed_mps", 0.025)),
                )
                place_result = controller.execute(
                    place_request, held_points_base=evidence.points_base,
                    before_observation=evidence.observation,
                )
        if place_result is not None and not place_result.success:
            result = replace(result, success=False, phase=FineGraspPhase.FAILED,
                             failure=place_result.failure or FailureCode.PLACE_VERIFICATION_FAILED,
                             message=f"抓取抬升成功，但放置未验证：{place_result.message}",
                             metrics={**dict(result.metrics), "place_success": 0, **dict(place_result.metrics)})
        elif place_result is not None:
            result = replace(result, message="抓取、搬运、放置及落点验证成功",
                             metrics={**dict(result.metrics), "place_success": 1, **dict(place_result.metrics)})
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
        "wrist_camera_profile": ARGS.wrist_camera_profile,
        "camera_configs": {name: _jsonable(cam.config) for name, cam in cameras.items()},
        "motion_config": motion_cfg,
        "coarse_approach": coarse_report,
        "recovery_mode": ARGS.recovery,
        "scene_view": ARGS.scene_view,
        "clutter_scene": None if CLUTTER_SCENE is None else CLUTTER_SCENE.to_dict(),
        "clutter_prim_paths": clutter_paths,
        "fault_initial_wrist_frames": ARGS.drop_initial_wrist_frames,
        "test_target_shift_m": ARGS.test_target_shift_m,
        "test_target_shift_applied": fault.applied,
        "attempt_results": _jsonable(getattr(node, "attempt_results", [])),
        "attempt_physics": attempt_physics,
        "effective_config": grasp_cfg,
        "result": _jsonable(result),
        "place": _jsonable(place_result),
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
    if VIDEO is not None:
        report["video"] = {"path": "demo.mp4", "recording_overhead": True,
                           "overview_is_perception_input": False}
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        outcome = (f"成功：实际抬升 {report['actual_target_lift_m']*100:.1f} cm" if result.success
                   else f"未成功 / 安全停止：{getattr(result.failure, 'value', result.failure)}")
        VIDEO.finish(overview_camera.get_rgba(), wrist_scene_camera.rgb_rgba(),
                     cameras["scene"].rgb_rgba(), outcome)
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
    if VIDEO is not None:
        try:
            VIDEO.close()
        except Exception:
            exit_code = 1
            print(traceback.format_exc(), file=sys.stderr, flush=True)
    if ARGS.keep_open and simulation_app.is_running():
        app_utils.pause()
        print(f"[BusAgent] Test finished (exit {exit_code}); GUI kept open, physics paused. "
              f"Report: {OUTPUT / 'report.json'}", flush=True)
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.01)
    # Isaac 6 fast shutdown exits here, before the SystemExit below.
    simulation_app.close(exit_code=exit_code)

raise SystemExit(exit_code)
