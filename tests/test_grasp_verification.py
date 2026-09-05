from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.contracts import CameraIntrinsics, GripperState, RGBDObservation  # noqa: E402
from mr_liu.grasp.transforms import make_transform  # noqa: E402
from mr_liu.grasp.verification import VisionGripperVerifier  # noqa: E402


def observation(sequence: int, camera_z: float) -> RGBDObservation:
    depth = np.full((5, 5), 0.20, dtype=np.float32)
    pose = make_transform(translation=np.asarray([0.0, 0.0, camera_z]))
    return RGBDObservation(
        sequence=sequence,
        timestamp_s=float(sequence),
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=CameraIntrinsics(5, 5, 100.0, 100.0, 2.0, 2.0),
        T_base_camera=pose,
        T_base_ee=pose,
        T_ee_camera=np.eye(4),
        target_mask=np.ones((5, 5), dtype=bool),
    )


class VisionGripperVerifierTests(unittest.TestCase):
    def test_accepts_target_that_rises_with_loaded_wrist(self) -> None:
        result = VisionGripperVerifier().verify_lifted(
            observation(1, 0.0),
            observation(2, 0.08),
            GripperState(2.0, width_m=0.012, contact=True, stalled=True),
        )
        self.assertTrue(result.success)
        self.assertLess(result.metrics["target_wrist_follow_error_m"], 1e-6)

    def test_rejects_zero_width_even_if_finger_mask_follows_wrist(self) -> None:
        result = VisionGripperVerifier().verify_lifted(
            observation(1, 0.0),
            observation(2, 0.08),
            GripperState(2.0, width_m=0.0, contact=True, stalled=True),
        )
        self.assertFalse(result.success)

    def test_rejects_fully_closed_gripper_when_lift_target_is_occluded(self) -> None:
        after = observation(2, 0.08)
        after = RGBDObservation(**{**after.__dict__, "target_mask": None})
        result = VisionGripperVerifier().verify_lifted(
            observation(1, 0.0),
            after,
            GripperState(2.0, width_m=0.0, contact=True, stalled=True),
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.reason, "target_missing_or_gripper_fully_closed_after_lift"
        )

    def test_rejects_target_motion_that_does_not_follow_wrist(self) -> None:
        before = observation(1, 0.0)
        after = observation(2, 0.08)
        # Visual target appears to rise only 40 mm while the wrist rises 80 mm.
        after = RGBDObservation(
            **{
                **after.__dict__,
                "T_base_camera": make_transform(
                    translation=np.asarray([0.0, 0.0, 0.04])
                ),
                "T_base_ee": make_transform(
                    translation=np.asarray([0.0, 0.0, 0.08])
                ),
                "T_ee_camera": make_transform(
                    translation=np.asarray([0.0, 0.0, -0.04])
                ),
            }
        )
        result = VisionGripperVerifier(max_follow_error_m=0.025).verify_lifted(
            before,
            after,
            GripperState(2.0, width_m=0.012, contact=True, stalled=True),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "target_did_not_follow_wrist")


if __name__ == "__main__":
    unittest.main()
