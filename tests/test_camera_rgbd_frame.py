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
    def test_wrist_pose_uses_live_fabric_not_authored_usd(self):
        from mr_liu.grasp.transforms import quaternion_wxyz_to_matrix
        camera = SceneCamera("wrist")
        camera._cam = Mock()
        camera._fabric_world_pose = Mock(return_value=(np.array([.4, -.2, 1.1]), np.array([1., 0., 0., 0.])))
        position, orientation = camera.world_pose(camera_axes="ros")
        np.testing.assert_allclose(position, [.4, -.2, 1.1])
        np.testing.assert_allclose(quaternion_wxyz_to_matrix(orientation), np.diag([1., -1., -1.]), atol=1e-12)
        camera._cam.get_world_pose.assert_not_called()
        _, usd = camera.world_pose(camera_axes="usd")
        np.testing.assert_allclose(quaternion_wxyz_to_matrix(usd), np.eye(3))
        _, world = camera.world_pose(camera_axes="world")
        np.testing.assert_allclose(quaternion_wxyz_to_matrix(world), [[0,-1,0],[0,0,1],[-1,0,0]], atol=1e-12)

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

    def test_grounding_copies_one_acquisition_not_direct_annotator_reads(self):
        for key in ('semantic_segmentation', 'instance_segmentation', 'instance_id_segmentation'):
            self.frame[key] = {'data': np.ones((2, 3), dtype=np.uint32),
                               'info': {'idToLabels': {'1': '/World/TabletopProps/Block'}}}
        packet = self.camera.grounding_frame()
        self.camera._cam.get_current_frame.assert_called_once()
        self.camera._cam.get_rgba.assert_not_called()
        self.camera._cam.get_depth.assert_not_called()
        self.camera._cam.get_world_pose.assert_not_called()
        self.frame['rgb'][:] = 0
        self.frame['instance_id_segmentation']['data'][:] = 0
        self.frame['instance_segmentation']['info']['idToLabels'].clear()
        np.testing.assert_array_equal(packet['bgr'], 42)
        np.testing.assert_array_equal(packet['leaf_instances']['data'], 1)
        self.assertIn('1', packet['instances']['info']['idToLabels'])
        np.testing.assert_allclose(packet['unproject'](np.array([[0, 0]]), np.array([1.25])),
                                   self.pose[:3, 3][None] + self.pose[:3, 2][None] * 1.25)

    def test_grounding_does_not_fill_missing_buffered_rgb_from_live_image(self):
        del self.frame['rgb']
        self.assertIsNone(self.camera.grounding_frame())
        self.camera._cam.get_rgba.assert_not_called()

    def test_grounding_accepts_legacy_leaf_key(self):
        payload = {'data': np.ones((2, 3)), 'info': {'idToLabels': {}}}
        self.frame['instance_id_segmentation_fast'] = payload
        packet = self.camera.grounding_frame()
        np.testing.assert_array_equal(packet['leaf_instances']['data'], 1)

    def test_instance_frame_accepts_isaac6_normalized_and_legacy_keys(self):
        from mr_liu.vision.grounding import instance_object_path
        for key in ("instance_segmentation", "instance_segmentation_fast"):
            payload = {"data": np.full((2, 3), 5, dtype=np.uint32),
                       "info": {"idToLabels": {"5": "/World/TabletopProps/NutM20/mesh"}}}
            self.camera._cam.get_current_frame.return_value = {key: payload}
            self.assertIs(self.camera.instance_frame(), payload)
            mask = self.camera.instance_mask("/World/TabletopProps/NutM20")
            self.assertTrue(mask.all())
            self.assertEqual(instance_object_path(self.camera.instance_frame(), mask),
                             "/World/TabletopProps/NutM20")

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

    def test_object_and_robot_masks_keep_separate_annotators_after_merge(self):
        for suffix in ("", "_fast"):
            with self.subTest(suffix=suffix):
                objects = {"data": np.array([[7, 0, 0], [0, 0, 7]]),
                           "info": {"idToLabels": {"7": "/World/TabletopProps/NutM20"}}}
                robot = {"data": np.array([[0, 3, 0], [0, 3, 0]]),
                         "info": {"idToLabels": {"3": "/World/SO101/gripper/mesh"}}}
                self.camera._cam.get_current_frame.return_value = {
                    "instance_segmentation" + suffix: objects,
                    "instance_id_segmentation" + suffix: robot,
                }
                self.assertIs(self.camera.instance_frame(), objects)
                self.assertIs(self.camera.instance_frame(leaf_paths=True), robot)
                np.testing.assert_array_equal(self.camera.instance_mask("NutM20"), objects["data"] == 7)
                np.testing.assert_array_equal(self.camera.instance_mask("/World/SO101", leaf_paths=True), robot["data"] == 3)

    def test_empty_robot_mask_differs_from_missing_annotator(self):
        self.camera._cam.get_current_frame.return_value = {
            "instance_id_segmentation": {
                "data": np.zeros((2, 3), dtype=np.uint32), "info": {"idToLabels": {}},
            },
        }
        np.testing.assert_array_equal(self.camera.instance_mask("/World/SO101", leaf_paths=True),
                                      np.zeros((2, 3), dtype=bool))
        self.assertIsNone(self.camera.instance_mask("NutM20"))
        self.camera._cam.get_current_frame.return_value = {}
        self.assertIsNone(self.camera.instance_mask("/World/SO101", leaf_paths=True))


if __name__ == "__main__":
    unittest.main()
