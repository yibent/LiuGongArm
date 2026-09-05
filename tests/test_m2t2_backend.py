from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.backends.factory import create_grasp_backend
from mr_liu.grasp.backends.m2t2 import M2T2Backend, M2T2Config, M2T2PlacementPlanner
from scripts.serve_m2t2 import _bottom_center
from mr_liu.grasp.contracts import CameraIntrinsics, GraspBackendError, PlaceRequest, RGBDObservation, TargetSpec
from mr_liu.grasp.transforms import make_transform


class FakeTransport:
    def infer_pick(self, scene_points, object_points, **kwargs):
        del kwargs
        self.scene_points, self.object_points = scene_points, object_points
        return {
            "grasps": [make_transform(translation=np.array([0.0, 0.0, 0.2]))],
            "confidences": [0.91],
            "contacts": [object_points.mean(axis=0)],
        }

    def infer_place(self, scene_points, object_points, ee_pose, place_request, **kwargs):
        del scene_points, object_points, ee_pose, place_request, kwargs
        return {"placements": [], "confidences": []}


class FailingTransport:
    def infer_pick(self, scene_points, object_points, **kwargs):
        del scene_points, object_points, kwargs
        raise TimeoutError("test timeout")


class PlaceTransport:
    def infer_place(self, scene_points, object_points, ee_pose, place_request, **kwargs):
        del scene_points, object_points, ee_pose, place_request, kwargs
        return {
            "placements": [
                make_transform(translation=np.array([0.5, 0.0, 1.00])),
                make_transform(translation=np.array([0.5, 0.0, 1.06])),
            ],
            "confidences": [0.95, 0.80],
        }


def observation():
    depth = np.full((20, 24), np.nan, dtype=np.float32)
    depth[5:12, 8:16] = 0.2
    return RGBDObservation(
        sequence=3,
        timestamp_s=1.0,
        rgb=np.zeros((20, 24, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=CameraIntrinsics(24, 20, 30, 30, 12, 10),
        T_base_camera=make_transform(translation=np.array([0.1, 0.0, 1.0])),
    )


class M2T2Tests(unittest.TestCase):
    def test_bottom_center_is_estimated_in_scene_frame(self):
        points = np.array([
            [0.42, -0.03, 1.00], [0.44, -0.03, 1.00],
            [0.42, 0.01, 1.01], [0.44, 0.01, 1.02],
        ])
        center = _bottom_center(points, grid_resolution=0.01)
        np.testing.assert_allclose(center, [0.43, -0.01, 1.00], atol=0.011)

    def test_target_contact_is_converted_to_camera_candidate(self):
        obs = observation()
        target_points = np.array([[0.0, 0.0, 0.2], [0.01, 0.0, 0.2], [0.0, 0.01, 0.2]] * 30)
        transport = FakeTransport()
        backend = M2T2Backend(M2T2Config(model_frame="camera", max_scene_points=64), transport=transport)
        candidates = backend.generate(obs, target_points, TargetSpec("obj"))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].backend, "m2t2")
        self.assertEqual(candidates[0].observation_sequence, 3)
        self.assertGreater(candidates[0].width_m, 0.0)
        self.assertEqual(transport.scene_points.shape[1], 3)

    def test_timeout_is_classified(self):
        backend = M2T2Backend(M2T2Config(model_frame="camera"), transport=FailingTransport())
        with self.assertRaises(GraspBackendError) as raised:
            backend.generate(observation(), np.ones((20, 3)), TargetSpec("obj"))
        self.assertEqual(raised.exception.failure.value, "model_timeout")

    def test_factory_supports_m2t2_and_graspgenx_fallback(self):
        config = {
            "backend": "m2t2",
            "backends": {
                "m2t2": {"model_frame": "camera", "fallback_backend": "graspgenx"},
                "graspgenx": {
                    "sweep_volume": {
                        "extents_open": [0.075, 0.018, 0.045],
                        "offset_open": [0, 0, 0.0225],
                        "extents_mid": [0.0375, 0.018, 0.045],
                        "offset_mid": [0, 0, 0.0225],
                        "fingertip_depth_m": 0.045,
                    }
                },
            },
        }
        backend = create_grasp_backend(config, m2t2_transport=FakeTransport())
        self.assertTrue(backend.name.startswith("m2t2+"))

    def test_placement_surface_height_is_a_safety_filter(self):
        planner = M2T2PlacementPlanner(
            M2T2Config(model_frame="base"), transport=PlaceTransport()
        )
        request = PlaceRequest((0.5, 0.0, 1.05), (0.2, 0.2), surface_z_m=1.05)
        placements = planner.generate(
            observation(),
            np.tile(np.array([[0.4, 0.0, 1.1]]), (20, 1)),
            np.eye(4),
            request,
        )
        self.assertEqual(len(placements), 1)
        self.assertAlmostEqual(placements[0][0][2, 3], 1.06)


if __name__ == "__main__":
    unittest.main()
