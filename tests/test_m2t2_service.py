"""Regression checks for the PyTorch 2.x M2T2 pose compatibility shim."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.grasp.backends.m2t2_decoder import build_6d_grasp
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for decoder regression")
class M2T2ServiceTests(unittest.TestCase):
    def test_three_predictions_cross_xyz_independently(self):
        x = torch.tensor([[1.,0,0], [0,1.,0], [0,0,1.]])
        z = torch.tensor([[0,0,1.], [1.,0,0], [0,1.,0]])
        contacts = torch.tensor([[.1,.2,.3], [.2,.4,.5], [.3,.4,.6]])
        widths = torch.tensor([.03,.04,.05])
        poses = build_6d_grasp(contacts, x, z, widths).numpy()
        for i, pose in enumerate(poses):
            np.testing.assert_allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3))
            self.assertAlmostEqual(np.linalg.det(pose[:3,:3]), 1.)
            np.testing.assert_allclose(pose[:3,0], x[i])
            np.testing.assert_allclose(pose[:3,2], z[i])
            np.testing.assert_allclose(pose[:3,3], contacts[i] + x[i]*widths[i]/2 - z[i]*.1034)

    def test_invalid_model_axes_are_not_repaired(self):
        x = torch.tensor([[2.,0,0]])
        pose = build_6d_grasp(torch.zeros((1,3)),x,x,torch.tensor([.03])).numpy()[0]
        from mr_liu.grasp.transforms import assert_transform
        with self.assertRaises(ValueError):
            assert_transform(pose)


if __name__ == "__main__":
    unittest.main()
