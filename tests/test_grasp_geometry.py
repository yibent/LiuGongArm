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
    register_object_rigid,
    register_object_translation,
    recover_small_depth_holes,
    support_completed_center,
)
from mr_liu.grasp.segmentation import (  # noqa: E402
    AppearanceDepthTrackerSegmenter,
    SeededDepthSegmenter,
)
from mr_liu.grasp.transforms import make_transform, rpy_to_matrix, transform_points  # noqa: E402


class GraspGeometryTests(unittest.TestCase):
    def test_rigid_registration_recovers_reference_to_current_transform(self) -> None:
        rng = np.random.default_rng(23)
        center = np.asarray([0.54, -0.21, 0.78])
        reference = center + rng.uniform(
            [-0.030, -0.018, -0.012], [0.027, 0.021, 0.034], size=(3000, 3)
        )
        rotation = rpy_to_matrix(np.deg2rad([3.0, -2.0, 5.0]))
        center_shift = np.asarray([0.004, -0.003, 0.002])
        current = center + center_shift + (reference - center) @ rotation.T
        current += rng.normal(0.0, 8e-5, size=current.shape)

        result = register_object_rigid(reference, current, max_iterations=20)

        self.assertTrue(result.reliable)
        self.assertAlmostEqual(result.translation_m, np.linalg.norm(center_shift), delta=2e-4)
        self.assertAlmostEqual(
            result.rotation_rad,
            np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)),
            delta=np.deg2rad(0.15),
        )
        # T_reference_current maps reference-frame samples into the current
        # observation.  Keeping this direction explicit prevents applying an
        # inverse correction to the predicted object pose.
        predicted = transform_points(result.T_reference_current, reference)
        np.testing.assert_allclose(predicted, current, atol=5e-4)

    def test_rigid_registration_rejects_insufficient_and_line_like_clouds(self) -> None:
        insufficient = np.zeros((20, 3), dtype=np.float64)
        result = register_object_rigid(insufficient, insufficient, min_inliers=48)
        self.assertFalse(result.reliable)
        self.assertTrue(np.isinf(result.residual_m))

        x = np.linspace(-0.04, 0.04, 300)
        reference = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
        current = reference + np.asarray([0.003, -0.002, 0.001])
        result = register_object_rigid(reference, current)
        self.assertFalse(result.reliable)
        self.assertTrue(np.isinf(result.residual_m))

    def test_rigid_registration_rejects_corrections_outside_local_bounds(self) -> None:
        rng = np.random.default_rng(29)
        reference = rng.uniform(
            [-0.025, -0.018, -0.012], [0.030, 0.021, 0.028], size=(2500, 3)
        )
        translated = reference + np.asarray([0.016, 0.0, 0.0])
        translation_result = register_object_rigid(
            reference,
            translated,
            max_correspondence_m=0.030,
            max_iterations=20,
        )
        self.assertFalse(translation_result.reliable)
        self.assertGreater(translation_result.translation_m, 0.012)

        rotation = rpy_to_matrix(np.deg2rad([0.0, 0.0, 15.0]))
        rotated = reference @ rotation.T
        rotation_result = register_object_rigid(
            reference,
            rotated,
            max_correspondence_m=0.030,
            max_iterations=25,
        )
        self.assertFalse(rotation_result.reliable)
        self.assertGreater(rotation_result.rotation_rad, np.deg2rad(12.0))

    def test_rigid_registration_rejects_mirrored_asymmetric_geometry(self) -> None:
        rng = np.random.default_rng(31)
        anchors = np.asarray(
            [
                [-0.030, -0.018, -0.012],
                [0.028, -0.011, -0.004],
                [-0.013, 0.024, 0.006],
                [0.019, 0.015, 0.034],
            ]
        )
        reference = np.concatenate(
            [anchor + rng.normal(0.0, 7e-4, size=(90, 3)) for anchor in anchors]
        )
        mirrored = reference.copy()
        mirrored[:, 0] *= -1.0

        result = register_object_rigid(
            reference,
            mirrored,
            max_correspondence_m=0.030,
            max_iterations=25,
        )

        self.assertFalse(result.reliable)

    def test_translation_registration_ignores_partial_view_centroid_shift(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.uniform(
            [-0.03, -0.02, 0.0], [0.03, 0.02, 0.05], size=(4000, 3)
        )
        moved = np.asarray([0.004, -0.003, 0.002])
        # The next wrist view retains an asymmetric subset, so its raw centroid
        # moves much farther than the object's true physical translation.
        partial = reference[reference[:, 0] > -0.005][::2] + moved
        # The servo supplies the preceding frame's estimate as the ICP seed.
        result = register_object_translation(
            reference, partial, initial_translation_m=np.asarray([0.003, -0.002, 0.001])
        )
        self.assertTrue(result.reliable)
        np.testing.assert_allclose(result.translation_base_m, moved, atol=8e-4)
        self.assertLess(result.residual_m, 8e-4)

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

    def test_seeded_segmenter_finds_foreground_when_coarse_point_hits_a_hole(self) -> None:
        intrinsics = CameraIntrinsics(width=80, height=60, fx=100, fy=100, cx=40, cy=30)
        depth = np.full((60, 80), 0.30, dtype=np.float32)
        # The projected coarse point (40, 30) is background. A disconnected
        # wrench-jaw-like foreground lies nearby and is 20 mm closer.
        depth[22:39, 50:67] = 0.288
        rgb = np.full((60, 80, 3), [120, 120, 120], dtype=np.uint8)
        observation = RGBDObservation(
            1, 0.0, rgb, depth, intrinsics, np.eye(4)
        )
        target = TargetSpec("concave", coarse_position_base_m=(0.0, 0.0, 0.29))

        mask = SeededDepthSegmenter(
            depth_tolerance_m=0.010,
            max_radius_px=35,
            coarse_search_radius_px=30,
        ).segment(observation, target)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask.sum()), 17 * 17)
        self.assertFalse(mask[30, 40])

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

    def test_appearance_tracker_prefers_fresh_depth_seed_over_colour_merge(self) -> None:
        intrinsics = CameraIntrinsics(width=40, height=30, fx=100, fy=100, cx=20, cy=15)
        depth = np.full((30, 40), 0.5, dtype=np.float32)
        depth[10:21, 15:26] = 0.2
        first_rgb = np.full((30, 40, 3), [80, 80, 80], dtype=np.uint8)
        first_rgb[10:21, 15:26] = [160, 120, 100]
        target = TargetSpec("pastel", coarse_position_base_m=(0.0, 0.0, 0.2))
        tracker = AppearanceDepthTrackerSegmenter(
            SeededDepthSegmenter(depth_tolerance_m=0.01, max_radius_px=20),
            morphology_kernel_px=1,
        )
        first = RGBDObservation(1, 0.0, first_rgb, depth, intrinsics, np.eye(4))
        self.assertEqual(int(tracker.segment(first, target).sum()), 121)

        # All pixels now have the target's chromaticity, so appearance alone is
        # one giant invalid component. Current-frame depth still isolates it.
        merged_rgb = np.full_like(first_rgb, [160, 120, 100])
        second = RGBDObservation(2, 0.1, merged_rgb, depth, intrinsics, np.eye(4))
        recovered = tracker.segment(second, target)
        self.assertIsNotNone(recovered)
        self.assertEqual(int(recovered.sum()), 121)

    def test_appearance_tracker_compensates_large_measured_camera_approach(self) -> None:
        k = CameraIntrinsics(width=120, height=90, fx=100, fy=100, cx=60, cy=45)
        tracker = AppearanceDepthTrackerSegmenter(
            SeededDepthSegmenter(depth_tolerance_m=.01, max_radius_px=60),
            morphology_kernel_px=1,
        )
        target = TargetSpec("part", coarse_position_base_m=(0., 0., .45))
        for sequence, distance, radius in ((1, .45, 5), (2, .15, 16)):
            rgb = np.full((90, 120, 3), 80, np.uint8)
            depth = np.full((90, 120), distance + .1, np.float32)
            rgb[45-radius:46+radius, 60-radius:61+radius] = [220, 20, 20]
            depth[45-radius:46+radius, 60-radius:61+radius] = distance
            pose = np.eye(4)
            pose[2, 3] = .45 - distance
            obs = RGBDObservation(sequence, sequence * .1, rgb, depth, k, pose)
            mask = tracker.segment(obs, target)
            self.assertIsNotNone(mask)
            self.assertEqual(int(mask.sum()), (2 * radius + 1) ** 2)


if __name__ == "__main__":
    unittest.main()
