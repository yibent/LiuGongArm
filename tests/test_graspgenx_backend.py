"""Pure tests for the isolated GraspGenX client backend."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import fine_grasp_config  # noqa: E402
from mr_liu.grasp.backends.factory import create_grasp_backend  # noqa: E402
from mr_liu.grasp.backends.graspgenx import (  # noqa: E402
    FallbackGraspBackend,
    GraspGenXBackend,
    GraspGenXConfig,
)
from mr_liu.grasp.contracts import (  # noqa: E402
    CameraIntrinsics,
    FailureCode,
    GraspBackendError,
    ObjectProperties,
    RGBDObservation,
    TargetSpec,
)
from mr_liu.grasp.transforms import make_transform  # noqa: E402


def object_points() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform([-0.010, -0.015, 0.18], [0.010, 0.015, 0.22], size=(256, 3))


def observation(sequence: int = 1, camera_x: float = 0.0) -> RGBDObservation:
    depth = np.full((12, 16), 0.2, dtype=np.float32)
    return RGBDObservation(
        sequence=sequence,
        timestamp_s=1.0,
        rgb=np.zeros((12, 16, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=CameraIntrinsics(16, 12, 100.0, 100.0, 8.0, 6.0),
        T_base_camera=make_transform(translation=np.asarray([camera_x, 0.0, 0.0])),
        target_mask=np.ones_like(depth, dtype=bool),
    )


class QueueTransport:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def _next(self):
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def infer_object(self, point_cloud, sweep_volume_params, **kwargs):
        self.calls.append(("object_points", point_cloud, sweep_volume_params, kwargs))
        return self._next()

    def infer_scene_depth(self, depth, intrinsics, instance_mask, sweep_volume_params, **kwargs):
        self.calls.append(("scene_depth", depth, intrinsics, instance_mask, kwargs))
        return self._next()


class GraspGenXBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        mapping = fine_grasp_config()["backends"]["graspgenx"]
        self.config = GraspGenXConfig.from_mapping(mapping)
        self.target = TargetSpec("unknown-1", "unseen part")

    @staticmethod
    def model_pose(x: float = 0.0, score: float = 0.8):
        pose = make_transform(translation=np.asarray([x, 0.0, 0.155]))
        return pose, score

    def test_model_base_pose_becomes_camera_frame_pinch_center_in_metres(self) -> None:
        pose, score = self.model_pose()
        transport = QueueTransport([(np.asarray([pose]), np.asarray([score]), ["diff"])])
        backend = GraspGenXBackend(self.config, transport=transport)
        candidates = backend.generate(observation(), object_points(), self.target)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].T_camera_grasp[2, 3], 0.200, places=6)
        self.assertGreater(candidates[0].width_m, 0.02)
        self.assertLess(candidates[0].width_m, self.config.sweep_volume.max_width_m)
        self.assertEqual(candidates[0].observation_sequence, 1)
        self.assertEqual(candidates[0].metadata["units"], "m")
        self.assertIn("T_camera_pinch_center", candidates[0].metadata["output_pose_convention"])
        self.assertEqual(transport.calls[0][0], "object_points")

    def test_top_down_width_covers_larger_one_sided_local_extent(self) -> None:
        pose = make_transform(
            np.diag([1.0, -1.0, -1.0]), np.asarray([0.0, 0.0, 0.245])
        )
        score = 0.8
        transport = QueueTransport([(np.asarray([pose]), np.asarray([score]), ["obb"])])
        config = replace(self.config, width_margin_m=0.006)
        backend = GraspGenXBackend(config, transport=transport)
        # The pinch centre is 10 mm off-centre in X. A symmetric span would
        # request only 56 mm including margin, while the fixed finger needs
        # about 64 mm to clear the robust +29 mm side with 3 mm clearance.
        xs = np.linspace(-0.020, 0.030, 64)
        ys = np.linspace(-0.004, 0.004, 8)
        points = np.asarray(
            [[x, y, 0.200] for x in xs for y in ys], dtype=np.float64
        )

        candidate = backend.generate(observation(), points, self.target)[0]

        self.assertGreater(candidate.width_m, 0.063)
        self.assertEqual(
            candidate.metadata["width_strategy"], "fixed_finger_local_section"
        )
        self.assertGreater(candidate.metadata["width_section_points"], 24)

    def test_handle_hint_cannot_move_pose_without_rescoring(self) -> None:
        pose = make_transform(
            np.diag([1.0, -1.0, -1.0]), np.asarray([0.040, 0.0, 0.245])
        )
        handle = np.asarray(
            [
                [x, y, 0.200]
                for x in np.linspace(-0.065, 0.015, 90)
                for y in np.linspace(-0.006, 0.006, 14)
            ]
        )
        head = np.asarray(
            [
                [x, y, 0.200]
                for x in np.linspace(0.020, 0.055, 36)
                for y in np.linspace(-0.025, 0.025, 28)
            ]
        )
        points = np.concatenate((handle, head), axis=0)
        target = TargetSpec(
            "tool",
            "unseen industrial tool",
            properties=ObjectProperties(metadata={"preferred_region": "handle"}),
        )
        backend = GraspGenXBackend(
            self.config,
            transport=QueueTransport(
                [(np.asarray([pose]), np.asarray([0.8]), ["obb"])]
            ),
        )

        candidate = backend.generate(observation(), points, target)[0]

        self.assertAlmostEqual(candidate.T_camera_grasp[0, 3], 0.040)
        self.assertEqual(
            candidate.metadata["region_strategy"], "model_pose"
        )
        self.assertEqual(candidate.metadata["region_shift_m"], 0.0)

    def test_large_close_range_cloud_is_deterministically_capped(self) -> None:
        pose, score = self.model_pose()
        transport = QueueTransport([(np.asarray([pose]), np.asarray([score]), ["diff"])])
        config = replace(self.config, max_object_points=128)
        backend = GraspGenXBackend(config, transport=transport)
        dense = np.repeat(object_points(), 32, axis=0)

        candidates = backend.generate(observation(), dense, self.target)

        sent = transport.calls[0][1]
        self.assertEqual(sent.shape, (128, 3))
        self.assertEqual(candidates[0].metadata["source_object_points"], len(dense))
        self.assertEqual(candidates[0].metadata["model_input_points"], 128)

    def test_scene_depth_mode_uses_integer_target_instance_in_camera_frame(self) -> None:
        pose, score = self.model_pose()
        config = replace(self.config, request_mode="scene_depth", inference_frame="camera")
        transport = QueueTransport([{1: (np.asarray([pose]), np.asarray([score]), ["obb"])}])
        backend = GraspGenXBackend(config, transport=transport)
        candidates = backend.generate(observation(), object_points(), self.target)

        self.assertEqual(len(candidates), 1)
        call = transport.calls[0]
        self.assertEqual(call[0], "scene_depth")
        self.assertTrue(np.issubdtype(call[3].dtype, np.integer))
        self.assertEqual(set(np.unique(call[3])), {1})
        self.assertEqual(candidates[0].metadata["model_frame"], "wrist_camera")

    def test_base_frame_inference_uses_current_hand_eye_transform(self) -> None:
        pose, score = self.model_pose()
        transport = QueueTransport([(np.asarray([pose]), np.asarray([score]), ["obb"])])
        backend = GraspGenXBackend(replace(self.config, inference_frame="base"), transport=transport)
        obs = observation(camera_x=0.1)

        candidates = backend.generate(obs, object_points(), self.target)

        sent = transport.calls[0][1]
        np.testing.assert_allclose(sent[:, 0], object_points()[:, 0] + 0.1, atol=1e-6)
        np.testing.assert_allclose(
            obs.T_base_camera @ candidates[0].T_camera_grasp,
            make_transform(translation=np.asarray([0.0, 0.0, 0.200])),
            atol=1e-6,
        )
        self.assertEqual(candidates[0].metadata["model_frame"], "base")

    def test_consensus_cluster_outranks_a_slightly_higher_singleton(self) -> None:
        values = [
            self.model_pose(0.000, 0.70),
            self.model_pose(0.002, 0.69),
            self.model_pose(-0.002, 0.68),
            self.model_pose(0.100, 0.71),
        ]
        transport = QueueTransport(
            [(np.asarray([item[0] for item in values]), np.asarray([item[1] for item in values]), ["diff"] * 4)]
        )
        candidate = GraspGenXBackend(self.config, transport=transport).generate(
            observation(), object_points(), self.target
        )[0]
        self.assertLess(abs(candidate.T_camera_grasp[0, 3]), 0.01)
        self.assertEqual(candidate.metadata["cluster_size"], 3)
        self.assertGreater(candidate.metadata["consensus_bonus"], 0.0)

    def test_continuity_is_compared_in_base_not_moving_camera_frame(self) -> None:
        first, _ = self.model_pose(0.0, 0.85)
        # The model now receives gravity-aligned base-frame points.  Its stable
        # output remains at base X=0 while the camera-frame representation moves.
        stable, _ = self.model_pose(0.0, 0.86)
        alternate, _ = self.model_pose(0.100, 0.91)
        transport = QueueTransport(
            [
                (np.asarray([first]), np.asarray([0.85]), ["diff"]),
                (np.asarray([alternate, stable]), np.asarray([0.91, 0.86]), ["diff", "diff"]),
            ]
        )
        backend = GraspGenXBackend(self.config, transport=transport)
        backend.generate(observation(1, 0.0), object_points(), self.target)
        second = backend.generate(observation(2, 0.010), object_points(), self.target)

        self.assertAlmostEqual(second[0].T_camera_grasp[0, 3], -0.010, places=6)
        self.assertGreater(second[0].metadata["continuity_bonus"], 0.0)
        T_base = observation(2, 0.010).T_base_camera @ second[0].T_camera_grasp
        self.assertAlmostEqual(T_base[0, 3], 0.0, places=6)

    def test_timeout_and_malformed_se3_have_distinct_failure_codes(self) -> None:
        timeout = GraspGenXBackend(
            self.config, transport=QueueTransport([TimeoutError("slow")])
        )
        with self.assertRaises(GraspBackendError) as timeout_context:
            timeout.generate(observation(), object_points(), self.target)
        self.assertEqual(timeout_context.exception.failure, FailureCode.MODEL_TIMEOUT)
        self.assertTrue(timeout_context.exception.retryable)

        malformed = np.eye(4)
        malformed[3, 3] = 0.0
        protocol = GraspGenXBackend(
            self.config,
            transport=QueueTransport([(np.asarray([malformed]), np.asarray([0.8]), ["diff"])]),
        )
        with self.assertRaises(GraspBackendError) as protocol_context:
            protocol.generate(observation(), object_points(), self.target)
        self.assertEqual(protocol_context.exception.failure, FailureCode.MODEL_PROTOCOL_ERROR)
        self.assertFalse(protocol_context.exception.retryable)

    def test_factory_can_disable_or_enable_geometric_failover(self) -> None:
        config = fine_grasp_config()
        config["backend"] = "graspgenx"
        config["backends"]["graspgenx"]["fallback_backend"] = "none"
        direct = create_grasp_backend(
            config, graspgenx_transport=QueueTransport([TimeoutError("offline")])
        )
        self.assertIsInstance(direct, GraspGenXBackend)

        config["backends"]["graspgenx"]["fallback_backend"] = "geometric"
        fallback = create_grasp_backend(
            config, graspgenx_transport=QueueTransport([TimeoutError("offline")])
        )
        self.assertIsInstance(fallback, FallbackGraspBackend)
        candidates = fallback.generate(observation(), object_points(), self.target)
        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates[0].metadata["fallback_reason"], "model_timeout")


if __name__ == "__main__":
    unittest.main()
