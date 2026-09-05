import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.grasp.motion_diagnostics import endpoint_error_chain


class MotionDiagnosticsTests(unittest.TestCase):
    def test_error_vectors_add_in_one_world_frame(self):
        chain = endpoint_error_chain([1,2,3], [1.001,2,3], [1.004,2,3], [1.0041,2,3])
        self.assertAlmostEqual(chain["ik_residual"]["norm_mm"], 1)
        self.assertAlmostEqual(chain["joint_tracking"]["norm_mm"], 3)
        self.assertAlmostEqual(chain["fk_usd_discrepancy"]["norm_mm"], .1)
        np.testing.assert_allclose(
            sum(np.array(chain[k]["vector_mm"]) for k in ("ik_residual", "joint_tracking", "fk_usd_discrepancy")),
            chain["total"]["vector_mm"])

    def test_scalar_lengths_are_not_summed_when_errors_cancel(self):
        chain = endpoint_error_chain([0,0,0], [.003,0,0], [.001,0,0], [.001,0,0])
        self.assertAlmostEqual(chain["total"]["norm_mm"], 1)
        self.assertAlmostEqual(chain["joint_tracking"]["vector_mm"][0], -2)
