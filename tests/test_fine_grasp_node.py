"""Closed-loop FineGrasp node tests with deterministic hardware fakes."""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import fine_grasp_config  # noqa: E402
from mr_liu.grasp.backends.geometric import GeometricAntipodalBackend  # noqa: E402
from mr_liu.grasp.contracts import (  # noqa: E402
    CameraIntrinsics,
    FailureCode,
    FineGraspRequest,
    FineGraspPhase,
    GraspCandidate,
    GraspBackendError,
    GripperState,
    RGBDObservation,
    RobotState,
    TargetSpec,
    VerificationResult,
)
from mr_liu.grasp.node import GeneralGraspNode  # noqa: E402
from mr_liu.grasp.segmentation import ObservationMaskSegmenter  # noqa: E402
from mr_liu.grasp.settings import FineGraspSettings  # noqa: E402
from mr_liu.grasp.transforms import make_transform  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class FakeMotion:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.pose = make_transform(translation=np.asarray([0.0, 0.0, 1.1]))
        self.stopped = False
        self.servo_command_distances: list[float] = []

    def robot_state(self) -> RobotState:
        return RobotState(self.clock(), self.pose)

    def is_reachable(self, pose: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(pose)))

    def is_collision_free(self, pose: np.ndarray, *, margin_m: float, allow_target_contact: bool = False) -> bool:
        del margin_m, allow_target_contact
        return pose[2, 3] >= 1.05

    def move_to(self, pose: np.ndarray, *, speed_scale: float) -> bool:
        del speed_scale
        self.pose = np.asarray(pose).copy()
        return True

    def servo_to(self, pose: np.ndarray, *, speed_scale: float) -> bool:
        self.servo_command_distances.append(
            float(np.linalg.norm(np.asarray(pose)[:3, 3] - self.pose[:3, 3]))
        )
        return self.move_to(pose, speed_scale=speed_scale)

    def stop(self) -> None:
        self.stopped = True


class FakeCamera:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sequence = 0

    def capture(self, target: TargetSpec) -> RGBDObservation:
        del target
        self.sequence += 1
        width, height = 80, 60
        mask = np.zeros((height, width), dtype=bool)
        mask[10:26, 28:52] = True
        depth = np.full((height, width), np.inf, dtype=np.float32)
        depth[mask] = 0.2
        return RGBDObservation(
            sequence=self.sequence,
            timestamp_s=self.clock(),
            rgb=np.zeros((height, width, 3), dtype=np.uint8),
            depth_m=depth,
            intrinsics=CameraIntrinsics(width, height, 120, 120, 40, 30),
            T_base_camera=make_transform(translation=np.asarray([0.0, 0.0, 1.0])),
            target_mask=mask,
        )


class FakeGripper:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.closed = False

    def open(self, width_m: float, *, speed_mps: float) -> bool:
        del width_m, speed_mps
        self.closed = False
        return True

    def close(self, width_m: float, *, force_n: float, speed_mps: float) -> bool:
        del width_m, force_n, speed_mps
        self.closed = True
        return True

    def state(self) -> GripperState:
        return GripperState(self.clock(), width_m=0.02, contact=self.closed)


class AlwaysVerifier:
    def verify_closed(self, before, after, gripper) -> VerificationResult:
        del before, after, gripper
        return VerificationResult(True, 1.0, "test")

    def verify_lifted(self, before, after, gripper) -> VerificationResult:
        del before, after, gripper
        return VerificationResult(True, 1.0, "test", {"target_lift_m": 0.08})


class TimeoutBackend:
    name = "timeout_model"

    def generate(self, observation, object_points_camera, target):
        del observation, object_points_camera, target
        raise GraspBackendError(
            FailureCode.MODEL_TIMEOUT,
            "model request deadline expired",
            retryable=True,
        )


class TopSurfaceBackend:
    name = "top_surface_model"

    def generate(self, observation, object_points_camera, target):
        del object_points_camera, target
        return [
            GraspCandidate(
                T_camera_grasp=make_transform(
                    rotation=np.diag([1.0, -1.0, -1.0]),
                    translation=np.asarray([0.0, 0.0, 0.2]),
                ),
                width_m=0.03,
                score=0.9,
                observation_sequence=observation.sequence,
                backend=self.name,
            )
        ]


class FineGraspNodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.motion = FakeMotion(self.clock)
        settings = FineGraspSettings.from_mapping(fine_grasp_config())
        self.settings = replace(
            settings,
            servo=replace(
                settings.servo,
                rotation_tolerance_rad=math.pi,
                max_rotation_step_rad=math.pi,
                stable_frames=1,
            ),
            loop=replace(settings.loop, model_period_s=100.0, servo_max_steps=20),
        )

    def node(self) -> GeneralGraspNode:
        return GeneralGraspNode(
            camera=FakeCamera(self.clock),
            segmenter=ObservationMaskSegmenter(),
            backend=GeometricAntipodalBackend(),
            motion=self.motion,
            gripper=FakeGripper(self.clock),
            verifier=AlwaysVerifier(),
            settings=self.settings,
            clock=self.clock,
        )

    def test_dry_run_returns_feasible_grasp_without_motion(self) -> None:
        original = self.motion.pose.copy()
        result = self.node().execute(
            FineGraspRequest(TargetSpec("object-1", "unseen part"), dry_run=True)
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.phase, FineGraspPhase.SUCCEEDED)
        self.assertIsNotNone(result.selected_grasp)
        np.testing.assert_allclose(self.motion.pose, original)

    def test_complete_loop_reobserves_and_lifts(self) -> None:
        result = self.node().execute(FineGraspRequest(TargetSpec("object-2", "box")))
        self.assertTrue(result.success, result.message)
        self.assertGreaterEqual(int(result.metrics["observations"]), 4)
        self.assertGreaterEqual(int(result.metrics["servo_steps"]), 1)
        self.assertGreater(self.motion.pose[2, 3], 1.2)

    def test_stable_frames_do_not_issue_zero_length_ik_commands(self) -> None:
        self.settings = replace(
            self.settings,
            servo=replace(self.settings.servo, stable_frames=3),
        )
        result = self.node().execute(FineGraspRequest(TargetSpec("object-stable", "box")))
        self.assertTrue(result.success, result.message)
        self.assertTrue(self.motion.servo_command_distances)
        self.assertTrue(all(distance > 1e-6 for distance in self.motion.servo_command_distances))

    def test_backend_timeout_is_not_reported_as_internal_error(self) -> None:
        node = self.node()
        node.backend = TimeoutBackend()
        result = node.execute(FineGraspRequest(TargetSpec("object-3", "part")))
        self.assertFalse(result.success)
        self.assertEqual(result.failure, FailureCode.MODEL_TIMEOUT)
        self.assertEqual(result.metrics["backend_error"], "model_timeout")
        self.assertEqual(result.metrics["backend_error_retryable"], "true")

    def test_top_down_model_grasp_uses_support_completed_contact_height(self) -> None:
        node = self.node()
        node.backend = TopSurfaceBackend()
        result = node.execute(
            FineGraspRequest(TargetSpec("object-supported", "unseen part"), dry_run=True)
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.selected_grasp.metadata["support_height_constrained"])
        self.assertAlmostEqual(result.selected_grasp.T_base_grasp[2, 3], 1.125, places=3)


if __name__ == "__main__":
    unittest.main()
