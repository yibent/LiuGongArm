import unittest
import numpy as np

from mr_liu.arena.truth_grounding import IsaacWorldTruthProvider


class Body:
    class Data:
        root_pos_w = np.array([[1.25, -0.4, 0.83]])
        root_quat_w = np.array([[0.0, 0.0, 0.70710678, 0.70710678]])
    data = Data()


class TruthGroundingTests(unittest.TestCase):
    def test_reads_exact_root_pose(self):
        target = IsaacWorldTruthProvider(lambda name: Body()).locate("cube")
        self.assertEqual(target.entity_name, "cube")
        self.assertEqual(target.position_world_m, (1.25, -0.4, 0.83))
        self.assertEqual(target.source, "isaac_world_truth")


if __name__ == "__main__":
    unittest.main()
