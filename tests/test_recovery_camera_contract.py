"""The fixed camera must not invent an eye-in-hand pose or recycle frames."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from mr_liu.grasp.adapters.isaac_camera import IsaacSceneCamera, IsaacWristCamera
from mr_liu.grasp.contracts import TargetSpec
from mr_liu.perception.camera import CameraRGBDFrame
from mr_liu.grasp.transforms import look_at_opencv, invert_transform, transform_points


class RecoveryCameraTests(unittest.TestCase):
    def test_oblique_camera_projects_workspace_center_on_optical_axis(self):
        position, target = np.array([.65, -.65, 1.75]), np.array([.35, -.15, 1.08])
        pose = look_at_opencv(position, target)
        point = transform_points(invert_transform(pose), target[None])[0]
        np.testing.assert_allclose(point[:2], 0., atol=1e-10)
        self.assertGreater(point[2], 0.)
        with self.assertRaises(ValueError):
            look_at_opencv(target, target)

    def setUp(self):
        self.scene = Mock(which='scene', has_depth=True, resolution=(3, 2))
        self.pose = np.eye(4)
        self.pose[2, 3] = 2.
        self.sample = CameraRGBDFrame(np.zeros((2, 3, 4), np.uint8), np.ones((2, 3)),
            np.array([[100., 0., 1.], [0., 100., 1.], [0., 0., 1.]]), self.pose, 1., 1)
        self.scene.rgbd_frame.return_value = self.sample
        self.world_to_base = np.eye(4)
        self.world_to_base[0, 3] = .4
        self.camera = IsaacSceneCamera(self.scene, T_base_world=self.world_to_base,
                                       advance_frame=Mock(), clock=lambda: 10.)

    def test_fixed_camera_uses_calibrated_base_chain_not_wrist_chain(self):
        result = self.camera.capture(TargetSpec('object'))
        np.testing.assert_allclose(result.T_base_camera, self.world_to_base @ self.pose)
        self.assertIsNone(result.T_base_ee)
        self.assertIsNone(result.T_ee_camera)
        self.assertEqual(result.metadata['mount_type'], 'fixed')

    def test_same_acquisition_is_not_republished_as_new_evidence(self):
        self.assertIsNotNone(self.camera.capture(TargetSpec('object')))
        self.assertIsNone(self.camera.capture(TargetSpec('object')))
        self.assertEqual(self.camera._sequence, 1)

    def test_moving_external_camera_requires_recalibration(self):
        self.camera.capture(TargetSpec('object'))
        moved = self.pose.copy()
        moved[0, 3] += .01
        self.scene.rgbd_frame.return_value = replace(self.sample, rendering_frame=2, rendering_time_s=2., T_world_camera=moved)
        with self.assertRaisesRegex(RuntimeError, 'recalibration'):
            self.camera.capture(TargetSpec('object'))

    def test_render_time_regression_cannot_be_hidden_by_a_new_counter(self):
        self.camera.capture(TargetSpec('object'))
        self.scene.rgbd_frame.return_value = replace(self.sample, rendering_frame=2, rendering_time_s=.5)
        self.assertIsNone(self.camera.capture(TargetSpec('object')))

    def test_invalid_external_calibration_is_rejected_before_any_capture(self):
        with self.assertRaises(ValueError):
            IsaacSceneCamera(self.scene, T_base_world=np.zeros((4, 4)), advance_frame=Mock())

    def test_wrist_camera_does_not_restamp_frozen_rendering_as_new(self):
        self.scene.which = 'wrist'
        self.scene.world_pose.return_value = (self.pose[:3, 3], [1., 0., 0., 0.])
        camera = IsaacWristCamera(self.scene, ee_pose_provider=lambda: np.eye(4), advance_frame=Mock())
        target = TargetSpec('object')
        first = camera.capture(target)
        self.assertIsNotNone(first)
        np.testing.assert_allclose(first.T_base_camera, first.T_base_ee @ first.T_ee_camera)
        self.assertIsNone(camera.capture(target))
        self.scene.rgbd_frame.return_value = replace(self.sample, rendering_frame=2, rendering_time_s=2.)
        self.assertEqual(camera.capture(target).sequence, 2)


if __name__ == '__main__':
    unittest.main()
