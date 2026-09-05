"""RGB-D geometry and local wrist segmentation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation, TargetSpec  # noqa: E402
from mr_liu.grasp.geometry import (  # noqa: E402
    ObjectGeometry,
    deproject_depth,
    estimate_object_frame,
    recover_small_depth_holes,
    support_completed_center,
)
from mr_liu.grasp.segmentation import (  # noqa: E402
    AppearanceDepthTrackerSegmenter,
    SeededDepthSegmenter,
)
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

    def test_support_completion_recovers_hidden_object_center(self) -> None:
        points = np.asarray(
            [[0.2, -0.1, z] for z in np.linspace(1.074, 1.086, 100)], dtype=np.float64
        )
        geometry = ObjectGeometry(
            mask=np.ones((10, 10), dtype=bool),
            points_camera=points.copy(),
            points_base=points,
            T_camera_object=make_transform(translation=np.asarray([0.2, -0.1, 1.080])),
            T_base_object=make_transform(translation=np.asarray([0.2, -0.1, 1.080])),
            valid_depth_ratio=1.0,
            extents_m=np.asarray([0.03, 0.03, 0.012]),
        )
        center = support_completed_center(geometry, table_height_m=1.05)
        self.assertAlmostEqual(center[2], (1.05 + np.percentile(points[:, 2], 95)) * 0.5)
        np.testing.assert_allclose(center[:2], [0.2, -0.1])

    def test_reflective_depth_recovery_fills_only_small_coherent_holes(self) -> None:
        depth = np.full((30, 30), 0.12, dtype=np.float32)
        mask = np.ones((30, 30), dtype=bool)
        depth[10:14, 11:15] = np.nan
        depth[18:29, 1:20] = np.nan
        recovered, count = recover_small_depth_holes(
            depth, mask, max_hole_area_px=50
        )
        self.assertEqual(count, 16)
        np.testing.assert_allclose(recovered[10:14, 11:15], 0.12)
        self.assertTrue(np.isnan(recovered[18:29, 1:20]).all())

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

    def test_appearance_tracker_rejects_background_area_takeover(self) -> None:
        intrinsics = CameraIntrinsics(width=40, height=30, fx=100, fy=100, cx=20, cy=15)
        rgb = np.full((30, 40, 3), [80, 80, 80], dtype=np.uint8)
        rgb[10:21, 15:26] = [220, 20, 20]
        depth = np.full((30, 40), 0.5, dtype=np.float32)
        depth[10:21, 15:26] = 0.2
        first = RGBDObservation(
            sequence=1,
            timestamp_s=0.0,
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            T_base_camera=np.eye(4),
        )
        target = TargetSpec("part", coarse_position_base_m=(0.0, 0.0, 0.2))
        tracker = AppearanceDepthTrackerSegmenter(
            SeededDepthSegmenter(depth_tolerance_m=0.01, max_radius_px=20),
            morphology_kernel_px=1,
        )
        initial = tracker.segment(first, target)
        self.assertIsNotNone(initial)
        assert initial is not None
        self.assertEqual(int(initial.sum()), 121)

        takeover = RGBDObservation(
            sequence=2,
            timestamp_s=0.1,
            rgb=np.full_like(rgb, [220, 20, 20]),
            depth_m=np.full_like(depth, 0.2),
            intrinsics=intrinsics,
            T_base_camera=np.eye(4),
        )
        self.assertIsNone(tracker.segment(takeover, target))


if __name__ == "__main__":
    unittest.main()
