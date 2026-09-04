"""RGB-D geometry and local wrist segmentation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation, TargetSpec  # noqa: E402
from mr_liu.grasp.geometry import deproject_depth, estimate_object_frame  # noqa: E402
from mr_liu.grasp.segmentation import SeededDepthSegmenter  # noqa: E402
from mr_liu.grasp.transforms import make_transform  # noqa: E402


class GraspGeometryTests(unittest.TestCase):
    def test_deprojection_uses_metric_depth_and_intrinsics(self) -> None:
        intrinsics = CameraIntrinsics(width=3, height=3, fx=100, fy=100, cx=1, cy=1)
        depth = np.full((3, 3), 0.2, dtype=np.float32)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        points = deproject_depth(depth, intrinsics, mask)
        np.testing.assert_allclose(points, [[0.0, 0.0, 0.2]], atol=1e-7)

    def test_object_frame_is_right_handed(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.uniform([-0.04, -0.01, 0.19], [0.04, 0.01, 0.21], size=(500, 3))
        T, extents = estimate_object_frame(points)
        self.assertAlmostEqual(float(np.linalg.det(T[:3, :3])), 1.0, places=6)
        self.assertGreater(extents.max(), extents.min() * 2)

    def test_seeded_segmenter_reprojects_base_hint_through_wrist_pose(self) -> None:
        intrinsics = CameraIntrinsics(width=40, height=30, fx=100, fy=100, cx=20, cy=15)
        depth = np.full((30, 40), 0.5, dtype=np.float32)
        depth[10:21, 15:26] = 0.2
        observation = RGBDObservation(
            sequence=1,
            timestamp_s=0.0,
            rgb=np.zeros((30, 40, 3), dtype=np.uint8),
            depth_m=depth,
            intrinsics=intrinsics,
            T_base_camera=make_transform(translation=np.asarray([0.1, 0.0, 1.0])),
        )
        target = TargetSpec(object_id="new-part", coarse_position_base_m=(0.1, 0.0, 1.2))
        mask = SeededDepthSegmenter(max_radius_px=20).segment(observation, target)
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask.sum()), 121)


if __name__ == "__main__":
    unittest.main()
