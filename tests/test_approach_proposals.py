import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.grasp.approach_proposals import inclined_proposals
from mr_liu.grasp.backends.graspgenx import ZmqGraspGenXTransport


class ApproachProposalTests(unittest.TestCase):
    def test_pose_frames_and_contact_heights(self):
        points = np.array([[.46,-.24,1.09], [.50,-.20,1.09], [.48,-.22,1.09]])
        poses = inclined_proposals(points, [.2,0,1.02], 1.05, .045)
        self.assertEqual(len(poses), 210)
        centers = poses[:, :3, 3] + poses[:, :3, 2] * .045
        self.assertTrue(np.allclose(centers[:, :2], [.48,-.22]))
        self.assertGreaterEqual(float(centers[:, 2].min()), 1.0659)
        self.assertLessEqual(float(centers[:, 2].max()), 1.0701)
        np.testing.assert_allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-6)
        self.assertTrue((poses[:, 2, 2] < 0).all())

    def test_diversity_does_not_copy_or_boost_scores(self):
        poses = np.repeat(np.eye(4)[None], 300, axis=0)
        scores = np.linspace(1, .1, 300)
        tags = ["obb"] * 150 + ["tilt_scored"] * 150
        _, confidence, branches = ZmqGraspGenXTransport._filter(poses, scores, tags, 0, 128)
        self.assertEqual(branches.count("tilt_scored"), 96)
        self.assertEqual(branches.count("obb"), 32)
        self.assertTrue(all(any(abs(float(v)-s)<1e-7 for s in scores) for v in confidence))

    def test_missing_support_or_robot_relation_does_not_make_poses(self):
        self.assertEqual(len(inclined_proposals(np.array([[.2,0,1.09]]), [.2,0,1.02],1.05,.045)),0)
        self.assertEqual(len(inclined_proposals(np.array([[.4,0,1.01]]), [.2,0,1.02],1.05,.045)),0)
