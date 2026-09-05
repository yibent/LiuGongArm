"""External-camera evidence must be fresh, target-specific and metric."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation, TargetSpec, GripperState, VerificationResult
from mr_liu.grasp.scene_observer import SceneTargetObserver, SceneAssistedVerifier


class SceneObserverTests(unittest.TestCase):
    def setUp(self):
        pose = np.diag([1., -1., -1., 1.])
        pose[2, 3] = 2.
        rgb = np.full((50, 50, 3), 120, np.uint8)
        depth = np.full((50, 50), .95, np.float32)
        rgb[22:29, 22:29] = [220, 30, 30]
        depth[22:29, 22:29] = .9
        self.frame = RGBDObservation(1, 10., rgb, depth,
            CameraIntrinsics(50, 50, 100, 100, 25, 25), pose, camera_frame="overhead")
        self.camera = Mock()
        self.camera.capture.return_value = self.frame
        self.observer = SceneTargetObserver(self.camera, table_height_m=1.05, clock=lambda: 10.)
        self.target = TargetSpec("part", coarse_position_base_m=(0., 0., 1.1))

    def test_metric_component_excludes_support_plane(self):
        evidence = self.observer.observe(self.target)
        self.assertIsNotNone(evidence)
        np.testing.assert_allclose(evidence.center_base_m, [0., 0., 1.1], atol=1e-6)
        self.assertEqual(int(evidence.observation.target_mask.sum()), 49)

    def test_stale_or_repeated_frames_cannot_be_verification_evidence(self):
        self.assertIsNotNone(self.observer.observe(self.target))
        self.assertIsNone(self.observer.observe(self.target))
        self.camera.capture.return_value = replace(self.frame, sequence=2, timestamp_s=8.)
        self.assertIsNone(self.observer.observe(self.target))

    def test_robot_silhouette_and_depth_holes_are_not_target_evidence(self):
        robot_mask = np.zeros((50, 50), bool)
        robot_mask[22:29, 22:29] = True
        self.camera.capture.return_value = replace(self.frame, metadata={"robot_self_mask": robot_mask})
        self.assertIsNone(self.observer.observe(self.target))
        missing = self.frame.depth_m.copy()
        missing[22:29, 22:29] = np.nan
        self.camera.capture.return_value = replace(self.frame, sequence=2, depth_m=missing)
        self.assertIsNone(self.observer.observe(self.target))

    def test_identity_does_not_learn_replacement_finger_color(self):
        self.assertIsNotNone(self.observer.observe(self.target))
        rgb = self.frame.rgb.copy()
        rgb[22:29, 22:29] = [240, 230, 10]
        self.camera.capture.return_value = replace(self.frame, sequence=2, rgb=rgb)
        self.assertIsNone(self.observer.observe(self.target))


class SceneVerificationTests(unittest.TestCase):
    def setUp(self):
        self.observer, self.motion, self.wrist = Mock(), Mock(), Mock()
        self.pose = np.eye(4)
        self.motion.robot_state.side_effect = lambda: SimpleNamespace(T_base_ee=self.pose.copy())
        self.start = np.array([.3, -.2, 1.08])
        self.observer.observe.return_value = SimpleNamespace(center_base_m=self.start)
        self.verifier = SceneAssistedVerifier(self.observer, self.motion, TargetSpec("part"), wrist=self.wrist)
        self.wrist.min_held_width_m = .002
        self.wrist.min_lift_m = .035
        self.wrist.verify_lifted.return_value = VerificationResult(False, 0., "occluded", {})
        self.verifier.before_close(None)
        self.pose[2, 3] += .08
        self.gripper = GripperState(0., width_m=.015, stalled=True)

    def test_independent_measured_lift_can_resolve_wrist_occlusion(self):
        self.observer.observe.return_value = SimpleNamespace(center_base_m=self.start + [0., 0., .08])
        result = self.verifier.verify_lifted(None, None, self.gripper)
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "external_target_follows_lift")

    def test_stationary_target_vetoes_a_wrist_false_positive(self):
        self.wrist.verify_lifted.return_value = VerificationResult(True, .9, "wrist", {})
        result = self.verifier.verify_lifted(None, None, self.gripper)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "external_target_stayed_on_support")

    def test_no_target_or_empty_gripper_cannot_confirm_success(self):
        self.observer.observe.return_value = None
        self.assertFalse(self.verifier.verify_lifted(None, None, self.gripper).success)
        self.observer.observe.return_value = SimpleNamespace(center_base_m=self.start + [0., 0., .08])
        self.assertFalse(self.verifier.verify_lifted(None, None, replace(self.gripper, width_m=0.)).success)


if __name__ == "__main__":
    unittest.main()
