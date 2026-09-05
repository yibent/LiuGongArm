"""Hardware-bridge tests that never connect to a physical SO-101."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.adapters.lerobot import (  # noqa: E402
    LeRobotGripperController,
    LeRobotWristCamera,
    PlannedLeRobotMotionExecutor,
    SynchronizedLeRobotRGBD,
)
from mr_liu.grasp.contracts import CameraIntrinsics, TargetSpec  # noqa: E402
from mr_liu.grasp.transforms import make_transform  # noqa: E402


class _Camera:
    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.latest_color_frame = np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8)
        self.latest_depth_frame = np.full((3, 4, 1), 125, dtype=np.uint16)
        self.latest_timestamp = 9.95


class _Robot:
    def __init__(self, *, contact_position: float | None = None) -> None:
        self.cameras = {"wrist": _Camera()}
        self.position = 100.0
        self.target = 100.0
        self.contact_position = contact_position
        self.force_commands: list[float] = []

    def get_observation(self):
        return {
            "shoulder_pan.pos": 1.0,
            "gripper.pos": self.position,
            "wrist": self.cameras["wrist"].latest_color_frame,
            "wrist_depth": self.cameras["wrist"].latest_depth_frame,
            "wrist_timestamp": self.cameras["wrist"].latest_timestamp,
        }

    def send_action(self, action):
        self.target = float(action["gripper.pos"])
        return action

    def advance(self, _duration: float) -> None:
        new_position = self.position + float(np.clip(self.target - self.position, -2.0, 2.0))
        if self.contact_position is not None:
            new_position = max(new_position, self.contact_position)
        self.position = new_position


class _ArmRobot:
    def __init__(self) -> None:
        self.q_deg = np.zeros(5)
        self.target_deg = self.q_deg.copy()

    def get_observation(self):
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        return {f"{name}.pos": value for name, value in zip(names, self.q_deg, strict=True)}

    def send_action(self, action):
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        self.target_deg = np.asarray([action[f"{name}.pos"] for name in names])
        return action

    def advance(self, _duration: float) -> None:
        self.q_deg += np.clip(self.target_deg - self.q_deg, -1.0, 1.0)


class _Planner:
    def forward_kinematics(self, q):
        return make_transform(translation=np.asarray([q[0], q[1], 1.0 + q[2]]))

    def solve_ik(self, q, target):
        result = np.asarray(q).copy()
        result[:3] = [target[0, 3], target[1, 3], target[2, 3] - 1.0]
        return result

    def is_path_collision_free(self, start, goal, *, margin_m, allow_target_contact):
        del start, margin_m, allow_target_contact
        return bool(np.max(np.abs(goal)) < 1.0)


class LeRobotAdapterTests(unittest.TestCase):
    def test_wrist_camera_uses_same_observation_fk_and_fixed_extrinsic(self) -> None:
        robot = _Robot()
        camera = LeRobotWristCamera(
            robot,
            camera_key="wrist",
            intrinsics=CameraIntrinsics(4, 3, 100.0, 100.0, 2.0, 1.5),
            T_ee_camera=make_transform(translation=np.asarray([0.02, 0.0, 0.01])),
            ee_pose_from_observation=lambda obs: make_transform(
                translation=np.asarray([float(obs["shoulder_pan.pos"]), 0.0, 1.0])
            ),
            clock=lambda: 10.0,
        )
        frame = camera.capture(TargetSpec("part"))
        assert frame is not None
        np.testing.assert_allclose(frame.T_base_camera[:3, 3], [1.02, 0.0, 1.01])
        np.testing.assert_allclose(frame.depth_m, 0.125)
        self.assertAlmostEqual(frame.metadata["capture_age_ms"], 50.0)

    def test_wrist_camera_rejects_unsynchronized_joint_and_frame_sources(self) -> None:
        robot = _Robot()
        camera = LeRobotWristCamera(
            robot,
            camera_key="wrist",
            intrinsics=CameraIntrinsics(4, 3, 100.0, 100.0, 2.0, 1.5),
            T_ee_camera=np.eye(4),
            ee_pose_from_observation=lambda _obs: np.eye(4),
        )
        original = robot.get_observation
        robot.get_observation = lambda: {
            key: value for key, value in original().items() if key != "wrist_timestamp"
        }
        with self.assertRaisesRegex(RuntimeError, "Strict eye-in-hand synchronization"):
            camera.capture(TargetSpec("part"))

    def test_driver_owned_snapshot_keeps_camera_and_fk_sample_together(self) -> None:
        robot = _Robot()
        synchronized = SynchronizedLeRobotRGBD(
            observation={"shoulder_pan.pos": 2.0},
            rgb=np.zeros((3, 4, 3), dtype=np.uint8),
            depth=np.full((3, 4), 200, dtype=np.uint16),
            timestamp_s=9.98,
        )
        camera = LeRobotWristCamera(
            robot,
            camera_key="wrist",
            intrinsics=CameraIntrinsics(4, 3, 100.0, 100.0, 2.0, 1.5),
            T_ee_camera=np.eye(4),
            ee_pose_from_observation=lambda obs: make_transform(
                translation=np.asarray([float(obs["shoulder_pan.pos"]), 0.0, 0.0])
            ),
            synchronized_snapshot=lambda: synchronized,
            clock=lambda: 10.0,
        )
        frame = camera.capture(TargetSpec("part"))
        assert frame is not None
        np.testing.assert_allclose(frame.T_base_camera[:3, 3], [2.0, 0.0, 0.0])
        np.testing.assert_allclose(frame.depth_m, 0.2)

    def test_gripper_requires_explicit_force_limit(self) -> None:
        robot = _Robot()
        controller = LeRobotGripperController(
            robot,
            set_force_limit_n=None,
            sleep=robot.advance,
        )
        self.assertFalse(controller.close(0.0, force_n=8.0, speed_mps=0.02))

    def test_gripper_accepts_contact_stall_and_records_force(self) -> None:
        robot = _Robot(contact_position=45.0)
        forces: list[float] = []
        controller = LeRobotGripperController(
            robot,
            set_force_limit_n=forces.append,
            sleep=robot.advance,
        )
        self.assertTrue(controller.close(0.0, force_n=6.0, speed_mps=0.02))
        self.assertEqual(forces, [6.0])
        self.assertTrue(controller.state().contact)

    def test_planned_motion_uses_ik_collision_and_feedback(self) -> None:
        robot = _ArmRobot()
        motion = PlannedLeRobotMotionExecutor(robot, _Planner(), sleep=robot.advance)
        target = make_transform(translation=np.asarray([0.12, -0.08, 1.06]))
        self.assertTrue(motion.is_reachable(target))
        self.assertTrue(motion.is_collision_free(target, margin_m=0.006))
        self.assertTrue(motion.move_to(target, speed_scale=0.5))
        np.testing.assert_allclose(motion.robot_state().T_base_ee[:3, 3], target[:3, 3], atol=0.004)


if __name__ == "__main__":
    unittest.main()
