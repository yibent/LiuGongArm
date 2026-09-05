from __future__ import annotations
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation
from mr_liu.grasp.geometry import extract_object_geometry
from mr_liu.grasp.fusion import FusionConfig, LocalSurfaceFusion, depth_evidence
from mr_liu.grasp.transforms import make_transform
from mr_liu.grasp.contact import FingerGeometry


def frame(sequence, camera_x=0.0, surface_z=0.20):
    k = CameraIntrinsics(80, 60, 150, 150, 40, 30)
    pose = make_transform(translation=np.asarray([camera_x, 0, 0]))
    depth = np.full((60, 80), surface_z, dtype=np.float32)
    return RGBDObservation(sequence, float(sequence), np.zeros((60, 80, 3), np.uint8),
                           depth, k, pose, pose, np.eye(4), target_mask=np.ones_like(depth, bool))


def geom(o):
    return extract_object_geometry(o, o.target_mask, min_depth_m=0.01, max_depth_m=1,
                                   mad_scale=4, self_mask_bottom_fraction=0)


class FusionTests(unittest.TestCase):
    def test_eye_in_hand_fuses_in_base_and_returns_latest_camera_frame(self):
        fusion = LocalSurfaceFusion(FusionConfig())
        a, b = frame(1), frame(2, 0.025)
        fusion.integrate(a, geom(a), "target")
        result = fusion.integrate(b, geom(b), "target")
        self.assertEqual(result["fusion_views"], 2)
        fused = fusion.geometry(b, geom(b))
        np.testing.assert_allclose(fused.points_camera[:, 0] + 0.025, fused.points_base[:, 0])
        self.assertLess(np.ptp(fused.points_base[:, 2]), 1e-6)

    def test_missing_depth_is_unknown_not_free(self):
        o = frame(1)
        o.depth_m[:] = np.nan
        free, surface, valid = depth_evidence(np.asarray([[0, 0, 0.1]]), o, 0.002)
        self.assertFalse(free.any() or surface.any() or valid.any())

    def test_moving_surface_discards_old_map(self):
        fusion = LocalSurfaceFusion(FusionConfig())
        a, b = frame(1), frame(2, 0.025, 0.24)
        fusion.integrate(a, geom(a), "target")
        result = fusion.integrate(b, geom(b), "target")
        self.assertEqual(result["fusion_reset"], "motion_or_identity_inconsistent")
        self.assertEqual(result["fusion_views"], 1)
        self.assertGreater(fusion.points_base[:, 2].min(), 0.23)

    def test_same_view_and_stale_frame_do_not_inflate_coverage(self):
        fusion = LocalSurfaceFusion(FusionConfig())
        for i in [1, 2]:
            o = frame(i)
            fusion.integrate(o, geom(o), "target")
        self.assertEqual(len(fusion.views), 1)
        with self.assertRaises(ValueError):
            fusion.integrate(o, geom(o), "target")

    def test_finger_sweep_catches_collision_between_endpoints(self):
        fingers = FingerGeometry()
        start, end = np.eye(4), np.eye(4)
        end[0, 3] = 0.04
        obstacle = np.asarray([[0.02, 0, -0.02]])
        self.assertEqual(fingers.collision_count(obstacle, start), 0)
        self.assertEqual(fingers.collision_count(obstacle, end), 0)
        self.assertGreater(fingers.path_collision_count(obstacle, start, end), 0)


if __name__ == "__main__":
    unittest.main()
