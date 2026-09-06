import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.arena.contracts import ManipulationRequest, model_grasp_to_tcp, placement_to_tcp


class ArenaContractTests(unittest.TestCase):
    def test_difficult_task_routes_both_models(self):
        self.assertEqual(ManipulationRequest("part", "tray", precise=True).route()["placement"], "anyplace")
        self.assertEqual(ManipulationRequest("part", "tray").route()["grasp"], "arena_basic")
        self.assertIsNone(ManipulationRequest("part", mode="enhanced").route()["placement"])

    def test_placement_keeps_rotation_and_measured_grasp_offset(self):
        relative = np.array([[0., -1, 0, .3], [1, 0, 0, .2], [0, 0, 1, .1], [0, 0, 0, 1]])
        initial = np.eye(4); initial[:3, 3] = [.4, -.2, .1]
        held = np.eye(4); held[:3, 3] = [.02, .01, -.04]
        goal = placement_to_tcp(relative, initial, held)
        np.testing.assert_allclose(goal @ held, relative @ initial)
        np.testing.assert_allclose(goal[:3, :3], relative[:3, :3])

    def test_panda_axes_and_tcp(self):
        tcp = model_grasp_to_tcp(np.eye(4))
        np.testing.assert_allclose(tcp[:3, 1], [-1, 0, 0])
        self.assertAlmostEqual(tcp[2, 3], .1034)

    def test_no_implicit_contact_skill_or_invalid_pose(self):
        with self.assertRaises(ValueError):
            ManipulationRequest("cup", "hook", relation="hang")
        with self.assertRaises(ValueError):
            placement_to_tcp(np.zeros((4, 4)), np.eye(4), np.eye(4))


if __name__ == "__main__":
    unittest.main()
