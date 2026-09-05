import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.grasp.observation import current_view_usable, observation_views


class ObservationTests(unittest.TestCase):
    def test_current_view_requires_measured_unoccluded_depth_and_margin(self):
        sample = SimpleNamespace(T_world_camera=np.eye(4),
            intrinsics=np.array([[100.,0,50],[0,100,50],[0,0,1]]), depth_m=np.full((100,100),.2))
        self.assertTrue(current_view_usable(sample, [0,0,.2]))
        self.assertFalse(current_view_usable(sample, [.098,0,.2]))
        self.assertFalse(current_view_usable(sample, [0,0,-.2]))
        sample.depth_m[:] = .1
        self.assertFalse(current_view_usable(sample, [0,0,.2]))
        sample.depth_m[:] = np.nan
        self.assertFalse(current_view_usable(sample, [0,0,.2]))
        self.assertFalse(current_view_usable(None, [0,0,.2]))

    def test_bounded_views_include_height_offset_and_tilt_without_moving(self):
        tool, camera = np.eye(4), np.eye(4)
        camera[:3,3] = [.15,0,-.1]
        point = np.array([.45,-.2,1.08])
        poses = observation_views(tool, camera, point)
        self.assertEqual(len(poses), 40)
        heights, offsets = set(), set()
        for pose in poses:
            optical = pose @ camera
            delta = point-optical[:3,3]
            np.testing.assert_allclose(optical[:3,2], delta/np.linalg.norm(delta),atol=1e-10)
            np.testing.assert_allclose(optical[:3,:3].T@optical[:3,:3], np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(optical[:3,:3]),1)
            heights.add(round(-delta[2],2)); offsets.add(round(np.linalg.norm(delta[:2]),2))
        self.assertGreaterEqual(len(heights),4)
        self.assertEqual(offsets,{0.,.08,.12})
