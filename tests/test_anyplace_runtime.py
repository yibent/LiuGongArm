"""Input-contract tests, not learned success or physical placement tests."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.place.anyplace_runtime import AnyPlaceRuntime


class AnyPlaceInputTests(unittest.TestCase):
    def runtime(self):
        instance=object.__new__(AnyPlaceRuntime)
        instance.torch=None
        return instance

    def test_invalid_cloud_rejected_before_model(self):
        with self.assertRaisesRegex(ValueError,'finite Nx3'):
            self.runtime().infer(np.zeros((8192,2)),np.zeros((1024,3)))
        with self.assertRaisesRegex(ValueError,'finite Nx3'):
            self.runtime().infer(np.full((8192,3),np.nan),np.zeros((1024,3)))

    def test_sparse_child_is_not_padded_with_invented_geometry(self):
        with self.assertRaisesRegex(ValueError,'child: insufficient'):
            self.runtime().infer(np.zeros((8192,3)),np.zeros((128,3)))

    def test_sparse_parent_rejected(self):
        with self.assertRaisesRegex(ValueError,'parent: insufficient'):
            self.runtime().infer(np.zeros((1023,3)),np.zeros((1024,3)))

    def test_unsupported_budget_rejected_before_model(self):
        with self.assertRaisesRegex(ValueError,'budget'):
            self.runtime().infer(np.zeros((8192,3)),np.zeros((1024,3)),candidates=1)


if __name__=='__main__':unittest.main()
