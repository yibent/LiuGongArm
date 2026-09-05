"""RGB-D acquisition contract without loading the simulator SDK."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.perception.camera import SceneCamera


class CameraRGBDFrameTests(unittest.TestCase):
    def setUp(self):
        self.camera = SceneCamera("scene")
        self.camera.resolution = (3, 2)
        self.pose = np.eye(4)
        self.pose[:3, 3] = [0.5, 0.0, 2.3]
        self.pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        usd_pose = self.pose @ np.diag([1., -1., -1., 1.])
        self.frame = {
            "rgb": np.full((2, 3, 4), 42, dtype=np.uint8),
            "distance_to_image_plane": np.full((2, 3), 1.25, dtype=np.float32),
            "camera_params": {"cameraViewTransform": np.linalg.inv(usd_pose).T},
            "rendering_time": 1.5, "rendering_frame": 90,
        }
        self.camera._cam = Mock()
        self.camera._cam.get_current_frame.return_value = self.frame
        self.camera._cam.get_intrinsics_matrix.return_value = np.eye(3)

    def test_metric_depth_and_pose_come_from_same_render_frame(self):
        sample = self.camera.rgbd_frame()
        self.assertIsNotNone(sample)
        self.assertEqual(sample.rendering_frame, 90)
        self.assertEqual(sample.rendering_time_s, 1.5)
        np.testing.assert_allclose(sample.T_world_camera, self.pose)
        np.testing.assert_allclose(sample.depth_m, 1.25)
        self.camera._cam.get_current_frame.assert_called_once()
        self.camera._cam.get_world_pose.assert_not_called()
        self.camera._cam.get_rgba.assert_not_called()
        self.camera._cam.get_depth.assert_not_called()

    def test_samples_do_not_alias_mutable_sdk_buffers(self):
        sample = self.camera.rgbd_frame()
        self.frame["rgb"][:] = 0
        self.frame["distance_to_image_plane"][:] = 0
        self.frame["camera_params"]["cameraViewTransform"][:] = 0
        np.testing.assert_array_equal(sample.rgba, 42)
        np.testing.assert_allclose(sample.depth_m, 1.25)
        np.testing.assert_allclose(sample.T_world_camera, self.pose)

    def test_isaac6_rational_render_reference_is_preserved(self):
        self.frame["rendering_frame"] = {
            "referenceTimeNumerator": 90, "referenceTimeDenominator": 60,
        }
        sample = self.camera.rgbd_frame()
        self.assertIsNone(sample.rendering_frame)
        self.assertEqual(sample.acquisition_id, (90, 60))
        self.assertEqual(sample.rendering_time_s, 1.5)

    def test_invalid_render_reference_is_rejected(self):
        self.frame["rendering_frame"] = {
            "referenceTimeNumerator": 90, "referenceTimeDenominator": 0,
        }
        self.assertIsNone(self.camera.rgbd_frame())

    def test_partial_acquisitions_are_not_filled_from_live_state(self):
        for key in self.frame:
            with self.subTest(missing=key):
                self.camera._cam.get_current_frame.return_value = {
                    k: v for k, v in self.frame.items() if k != key
                }
                self.assertIsNone(self.camera.rgbd_frame())

    def test_mismatched_depth_resolution_is_rejected(self):
        self.frame["distance_to_image_plane"] = np.ones((1, 3))
        self.assertIsNone(self.camera.rgbd_frame())

    def test_rgb_only_camera_cannot_supply_metric_depth(self):
        self.camera.annotators = ("rgb",)
        self.assertIsNone(self.camera.rgbd_frame())
        self.camera._cam.get_current_frame.assert_not_called()

    def test_robot_mask_uses_leaf_paths_and_normalized_isaac6_frame_key(self):
        self.frame["instance_id_segmentation"] = {
            "data": np.array([[1, 2, 0], [2, 0, 1]], dtype=np.uint32),
            "info": {"idToLabels": {"1": "/World/SO101/gripper/mesh",
                                     "2": "/World/UnseenObject/mesh"}},
        }
        mask = self.camera.instance_mask("/World/SO101", leaf_paths=True)
        np.testing.assert_array_equal(mask, [[True, False, False], [False, False, True]])


if __name__ == "__main__":
    unittest.main()
