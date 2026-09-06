"""Input-contract tests, not learned success or physical placement tests."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.place.anyplace_runtime import AnyPlaceRuntime, model_cloud


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

    def test_sparse_cloud_repeats_only_observed_points_with_honest_counts(self):
        points = np.random.default_rng(4).normal(size=(406,3)).astype(np.float32)
        sampled, counts = model_cloud(points, 'child')
        assert sampled.shape == (1024,3)
        assert {tuple(p) for p in sampled} == {tuple(p) for p in points}
        assert counts == {'observed_points':406, 'model_input_points':1024, 'repeated_samples':618}
        _, frequency = np.unique(sampled, axis=0, return_counts=True)
        assert frequency.max() - frequency.min() <= 1

    def test_empty_cloud_has_no_measurements_to_sample(self):
        with self.assertRaisesRegex(ValueError,'parent: empty'):
            self.runtime().infer(np.empty((0,3)),np.zeros((1024,3)))

    def test_dense_cloud_preserves_all_samples_for_upstream_fps(self):
        points = np.ones((4096,3), np.float32)
        sampled, counts = model_cloud(points, 'parent')
        assert sampled is points and counts['repeated_samples'] == 0

    def test_unsupported_budget_rejected_before_model(self):
        with self.assertRaisesRegex(ValueError,'budget'):
            self.runtime().infer(np.zeros((8192,3)),np.zeros((1024,3)),candidates=1)


if __name__=='__main__':unittest.main()
