"""Isaac/OpenCV camera-axis and calibration-chain tests without launching Kit."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.adapters.isaac_camera import T_world_opencv_from_ros_pose  # noqa: E402
from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation  # noqa: E402
from mr_liu.grasp.transforms import make_transform  # noqa: E402


class IsaacCameraAdapterTests(unittest.TestCase):
    def test_ros_to_opencv_flips_image_x_and_y_only(self) -> None:
        T = T_world_opencv_from_ros_pose(np.asarray([1, 2, 3]), np.asarray([1, 0, 0, 0]))
        np.testing.assert_allclose(T[:3, :3], np.diag([-1.0, -1.0, 1.0]))
        self.assertAlmostEqual(float(np.linalg.det(T[:3, :3])), 1.0)

    def test_observation_rejects_inconsistent_hand_eye_chain(self) -> None:
        intrinsics = CameraIntrinsics(2, 2, 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            RGBDObservation(
                sequence=1,
                timestamp_s=0.0,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                depth_m=np.ones((2, 2), dtype=np.float32),
                intrinsics=intrinsics,
                T_base_camera=make_transform(translation=np.asarray([1.0, 0.0, 0.0])),
                T_base_ee=make_transform(),
                T_ee_camera=make_transform(),
            )

    def test_observation_accepts_exact_hand_eye_chain(self) -> None:
        intrinsics = CameraIntrinsics(2, 2, 1, 1, 1, 1)
        T_base_ee = make_transform(translation=np.asarray([0.2, 0.1, 1.0]))
        T_ee_camera = make_transform(translation=np.asarray([0.05, 0.0, 0.02]))
        observation = RGBDObservation(
            sequence=1,
            timestamp_s=0.0,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_m=np.ones((2, 2), dtype=np.float32),
            intrinsics=intrinsics,
            T_base_camera=T_base_ee @ T_ee_camera,
            T_base_ee=T_base_ee,
            T_ee_camera=T_ee_camera,
        )
        np.testing.assert_allclose(observation.T_base_camera[:3, 3], [0.25, 0.1, 1.02])


if __name__ == "__main__":
    unittest.main()
