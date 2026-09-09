"""Isolated GraspGenX client backend for the eye-in-hand slow loop.

The neural environment and its weights live outside the BusAgent process.  A
small ZMQ wire client is kept here so Isaac Sim only needs numpy, msgpack and
pyzmq; importing this module never imports torch or the GraspGenX package.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from mr_liu.grasp.contracts import (
    FailureCode,
    GraspBackendError,
    GraspCandidate,
    RGBDObservation,
    TargetSpec,
)
from mr_liu.grasp.transforms import (
    invert_transform,
    make_transform,
    rotation_angle_rad,
    transform_points,
)


@dataclass(frozen=True)
class SweepVolume:
    """GraspGenX ``sweep_volume_v2`` conditioning in metres."""

    extents_open: tuple[float, float, float]
    offset_open: tuple[float, float, float]
    extents_mid: tuple[float, float, float]
    offset_mid: tuple[float, float, float]
    gripper_type: int = 0
    fingertip_depth_m: float = 0.045

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SweepVolume":
        def vector(name: str) -> tuple[float, float, float]:
            raw = tuple(float(item) for item in value[name])
            if len(raw) != 3 or not np.all(np.isfinite(raw)):
                raise ValueError(f"graspgenx.sweep_volume.{name} must contain three finite values")
            return raw

        result = cls(
            extents_open=vector("extents_open"),
            offset_open=vector("offset_open"),
            extents_mid=vector("extents_mid"),
            offset_mid=vector("offset_mid"),
            gripper_type=int(value.get("gripper_type", 0)),
            fingertip_depth_m=float(value["fingertip_depth_m"]),
        )
        if min(*result.extents_open, *result.extents_mid, result.fingertip_depth_m) <= 0.0:
            raise ValueError("GraspGenX sweep extents and fingertip depth must be positive")
        if result.gripper_type not in (0, 1, 2):
            raise ValueError("GraspGenX gripper_type must be 0, 1 or 2")
        return result

    @property
    def max_width_m(self) -> float:
        return float(self.extents_open[0])

    def to_wire(self) -> dict[str, Any]:
        return {
            "extents_open": np.asarray(self.extents_open, dtype=np.float32),
            "offset_open": np.asarray(self.offset_open, dtype=np.float32),
            "extents_mid": np.asarray(self.extents_mid, dtype=np.float32),
            "offset_mid": np.asarray(self.offset_mid, dtype=np.float32),
            "gripper_type": self.gripper_type,
            "fingertip_depth": self.fingertip_depth_m,
        }


@dataclass(frozen=True)
class GraspGenXConfig:
    host: str
    port: int
    timeout_ms: int
    request_mode: str
    inference_frame: str
    planner: str
    num_grasps: int
    max_candidates: int
    min_object_points: int
    max_object_points: int
    min_confidence: float
    width_margin_m: float
    cluster_translation_m: float
    cluster_rotation_rad: float
    consensus_bonus: float
    continuity_translation_m: float
    continuity_rotation_rad: float
    continuity_bonus: float
    switch_score_margin: float
    sweep_volume: SweepVolume

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GraspGenXConfig":
        cluster = value.get("stability", {})
        sweep = SweepVolume.from_mapping(value["sweep_volume"])
        result = cls(
            host=str(value.get("host", "127.0.0.1")),
            port=int(value.get("port", 5556)),
            timeout_ms=int(value.get("timeout_ms", 3000)),
            request_mode=str(value.get("request_mode", "object_points")),
            inference_frame=str(value.get("inference_frame", "base")),
            planner=str(value.get("planner", "diffusion")),
            num_grasps=int(value.get("num_grasps", 100)),
            max_candidates=int(value.get("max_candidates", 64)),
            min_object_points=int(value.get("min_object_points", 64)),
            max_object_points=int(value.get("max_object_points", 4096)),
            min_confidence=float(value.get("min_confidence", 0.10)),
            width_margin_m=float(value.get("width_margin_m", 0.006)),
            cluster_translation_m=float(cluster.get("cluster_translation_m", 0.025)),
            cluster_rotation_rad=math.radians(float(cluster.get("cluster_rotation_deg", 25.0))),
            consensus_bonus=float(cluster.get("consensus_bonus", 0.04)),
            continuity_translation_m=float(cluster.get("continuity_translation_m", 0.040)),
            continuity_rotation_rad=math.radians(
                float(cluster.get("continuity_rotation_deg", 40.0))
            ),
            continuity_bonus=float(cluster.get("continuity_bonus", 0.10)),
            switch_score_margin=float(cluster.get("switch_score_margin", 0.08)),
            sweep_volume=sweep,
        )
        if not result.host or not 0 < result.port < 65536 or result.timeout_ms <= 0:
            raise ValueError("Invalid GraspGenX server address or timeout")
        if result.request_mode not in {"object_points", "scene_depth"}:
            raise ValueError("GraspGenX request_mode must be object_points or scene_depth")
        if result.inference_frame not in {"base", "camera"}:
            raise ValueError("GraspGenX inference_frame must be base or camera")
        if result.request_mode == "scene_depth" and result.inference_frame != "camera":
            raise ValueError("GraspGenX scene_depth mode requires inference_frame=camera")
        if result.planner not in {"diffusion", "graspmoe"}:
            raise ValueError("GraspGenX planner must be diffusion or graspmoe")
        if (
            result.num_grasps <= 0
            or result.max_candidates <= 0
            or result.min_object_points < 6
            or result.max_object_points < result.min_object_points
        ):
            raise ValueError("Invalid GraspGenX candidate or point counts")
        positive = (
            result.width_margin_m,
            result.cluster_translation_m,
            result.cluster_rotation_rad,
            result.continuity_translation_m,
            result.continuity_rotation_rad,
        )
        if min(positive) <= 0.0:
            raise ValueError("GraspGenX distance and angle thresholds must be positive")
        return result


class GraspGenXTransport(Protocol):
    def infer_object(self, point_cloud: np.ndarray, sweep_volume_params: Mapping[str, Any], **kwargs): ...

    def infer_scene_depth(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        sweep_volume_params: Mapping[str, Any],
        **kwargs,
    ): ...


class ZmqGraspGenXTransport:
    """Torch-free implementation of the official GraspGenX msgpack protocol."""

    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self._socket = None
        self._lock = threading.Lock()
        self._msgpack = None
        self._zmq = None

    def _load_dependencies(self) -> None:
        if self._zmq is not None:
            return
        try:
            import msgpack
            import msgpack_numpy
            import zmq
        except ImportError as exc:
            raise GraspBackendError(
                FailureCode.MODEL_UNAVAILABLE,
                "GraspGenX client dependencies are missing; install requirements-graspgenx-client.txt",
                retryable=False,
            ) from exc
        msgpack_numpy.patch()
        self._msgpack = msgpack
        self._zmq = zmq

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _connect(self) -> None:
        self._load_dependencies()
        if self._socket is not None:
            return
        assert self._zmq is not None
        socket = self._zmq.Context.instance().socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.connect(self.address)
        self._socket = socket

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._connect()
            assert self._msgpack is not None and self._zmq is not None
            try:
                self._socket.send(self._msgpack.packb(payload, use_bin_type=True))
                raw = self._socket.recv()
            except Exception as exc:
                self.close()
                if isinstance(exc, self._zmq.error.Again):
                    raise TimeoutError(
                        f"GraspGenX server {self.address} timed out after {self.timeout_ms} ms"
                    ) from exc
                raise ConnectionError(f"GraspGenX transport failed at {self.address}: {exc}") from exc
        response = self._msgpack.unpackb(raw, raw=False)
        if not isinstance(response, dict):
            raise ValueError(f"GraspGenX response must be a mapping, got {type(response).__name__}")
        if "error" in response:
            raise RuntimeError(f"GraspGenX inference server error: {response['error']}")
        return response

    @staticmethod
    def _filter(
        grasps: Any,
        scores: Any,
        tags: Sequence[str],
        threshold: float,
        topk: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        poses = np.asarray(grasps, dtype=np.float32).reshape(-1, 4, 4)
        confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(poses) != len(confidence):
            raise ValueError("GraspGenX pose/score counts differ")
        branch_tags = list(tags) if tags else ["unknown"] * len(poses)
        if len(branch_tags) != len(poses):
            raise ValueError("GraspGenX pose/branch-tag counts differ")
        keep = np.isfinite(confidence) & (confidence >= threshold)
        poses, confidence = poses[keep], confidence[keep]
        branch_tags = [tag for tag, include in zip(branch_tags, keep) if include]
        order = np.argsort(-confidence, kind="stable")[:topk]
        if "tilt_scored" in branch_tags and topk > 1:
            ranked = np.argsort(-confidence, kind="stable")
            tilted = [i for i in ranked if branch_tags[i] == "tilt_scored"]
            original = [i for i in ranked if branch_tags[i] != "tilt_scored"]
            # Preserve approach diversity before IK; 128 vertical yaws cannot
            # solve a five-axis arm's missing approach direction.
            chosen = tilted[:3 * topk // 4] + original[:topk // 4]
            chosen += [i for i in ranked if i not in chosen][:topk - len(chosen)]
            order = np.asarray(sorted(chosen, key=lambda i: -confidence[i]))
        return poses[order], confidence[order], [branch_tags[index] for index in order]

    def infer_object(self, point_cloud, sweep_volume_params, **kwargs):
        response = self._request(
            {
                "action": "infer_object",
                "point_cloud": np.asarray(point_cloud, dtype=np.float32),
                "sweep_volume_params": dict(sweep_volume_params),
                "planner": str(kwargs.get("planner", "diffusion")),
                "num_grasps": int(kwargs.get("num_grasps", 100)),
                "seed": int(kwargs.get("seed", 0)),
                "approach_context": kwargs.get("approach_context"),
            }
        )
        result = self._filter(
            response["grasps"],
            response["confidences"],
            response.get("branch_tags", []),
            float(kwargs.get("grasp_threshold", -1.0)),
            int(kwargs.get("topk_num_grasps", 100)),
        )
        return result if kwargs.get("return_branch_tags", False) else result[:2]

    def infer_scene_depth(self, depth, intrinsics, instance_mask, sweep_volume_params, **kwargs):
        response = self._request(
            {
                "action": "infer_scene_depth",
                "depth": np.asarray(depth, dtype=np.float32),
                "intrinsics": np.asarray(intrinsics, dtype=np.float64),
                "instance_mask": np.asarray(instance_mask, dtype=np.int32),
                "sweep_volume_params": dict(sweep_volume_params),
                "planner": str(kwargs.get("planner", "diffusion")),
                "min_object_points": int(kwargs.get("min_object_points", 64)),
                "seed": int(kwargs.get("seed", 0)),
                "num_grasps": int(kwargs.get("num_grasps", 100)),
            }
        )
        ids = np.asarray(response["instance_ids"]).reshape(-1)
        poses = response["grasps"]
        scores = response["confidences"]
        tags = response.get("branch_tags", [[] for _ in poses])
        if not (len(ids) == len(poses) == len(scores) == len(tags)):
            raise ValueError("GraspGenX scene response list counts differ")
        result = {}
        for index, instance_id in enumerate(ids):
            item = self._filter(
                poses[index],
                scores[index],
                tags[index],
                float(kwargs.get("grasp_threshold", -1.0)),
                int(kwargs.get("topk_num_grasps", 100)),
            )
            result[int(instance_id)] = item if kwargs.get("return_branch_tags", False) else item[:2]
        return result


@dataclass
class _Proposal:
    T_camera_grasp: np.ndarray
    T_base_grasp: np.ndarray
    raw_score: float
    width_m: float
    branch: str
    source_index: int
    width_strategy: str = "global_percentile"
    width_section_points: int = 0
    region_strategy: str = "model_pose"
    region_shift_m: float = 0.0
    cluster_id: int = -1
    cluster_size: int = 1
    consensus_bonus: float = 0.0
    continuity_bonus: float = 0.0
    score: float = 0.0


def _conservative_top_down_width(
    points: np.ndarray,
    T_model_grasp: np.ndarray,
    *,
    transverse_span_m: float,
    width_margin_m: float,
    fallback_width_m: float,
) -> tuple[float, int]:
    """Estimate the opening needed by an asymmetric fixed-finger gripper.

    GraspGenX exposes a symmetric pinch-centre pose, but SO-101 approaches with
    one finger fixed. A global percentile width silently assumes the proposal
    is exactly centred. That is unsafe for partial wrist views and compound
    objects: a mug handle or tool head can move the proposal a few millimetres,
    putting the fixed finger inside the main body.

    Only points inside the physical finger's transverse slab are relevant to
    insertion. Twice the larger one-sided extent keeps either jaw-swap
    interpretation outside the observed silhouette; the explicit margin
    covers depth quantisation and the unseen back surface.
    """
    local = (np.asarray(points, dtype=np.float64) - T_model_grasp[:3, 3]) @ (
        T_model_grasp[:3, :3]
    )
    half_span = max(float(transverse_span_m) * 0.5, 0.004)
    section = local[np.abs(local[:, 1]) <= half_span]
    minimum = max(24, int(math.ceil(len(local) * 0.01)))
    if len(section) < minimum:
        return float(fallback_width_m), int(len(section))
    low, high = np.percentile(section[:, 0], [3.0, 97.0])
    one_sided_extent = max(abs(float(low)), abs(float(high)))
    required = 2.0 * one_sided_extent + float(width_margin_m)
    return max(float(fallback_width_m), required), int(len(section))


def _stable_elongated_region_center(
    points: np.ndarray,
    *,
    min_width_m: float = 0.008,
    max_width_m: float = 0.065,
) -> tuple[np.ndarray, float, float] | None:
    """Find the longest stable-width region of an elongated point cloud.

    This is a geometry affordance, not an object-category template. It finds a
    long section whose transverse width remains approximately constant, which
    is the property shared by hammer, wrench, screwdriver and similar handles.
    Bulky heads, open jaws and thin shafts form shorter or invalid runs.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if len(cloud) < 128:
        return None
    xy = cloud[:, :2]
    origin = np.median(xy, axis=0)
    covariance = np.cov(xy - origin, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 1e-12:
        return None
    elongation = float(np.sqrt(eigenvalues[1] / eigenvalues[0]))
    if elongation < 2.0:
        return None
    major = eigenvectors[:, 1]
    minor = eigenvectors[:, 0]
    projected_major = (xy - origin) @ major
    projected_minor = (xy - origin) @ minor
    low_t, high_t = np.percentile(projected_major, [3.0, 97.0])
    if high_t - low_t < 0.045:
        return None
    step = 0.004
    half_window = 0.009
    centers = np.arange(low_t + half_window, high_t - half_window + step * 0.5, step)
    minimum_points = max(32, int(math.ceil(len(cloud) * 0.008)))
    samples: list[tuple[float, float, float, int]] = []
    for center_t in centers:
        section_mask = np.abs(projected_major - center_t) <= half_window
        count = int(section_mask.sum())
        if count < minimum_points:
            continue
        section_minor = projected_minor[section_mask]
        low_s, high_s = np.percentile(section_minor, [5.0, 95.0])
        width = float(high_s - low_s)
        if not min_width_m <= width <= max_width_m:
            continue
        samples.append((float(center_t), width, float(np.median(section_minor)), count))
    if not samples:
        return None

    runs: list[list[tuple[float, float, float, int]]] = []
    current: list[tuple[float, float, float, int]] = []
    for sample in samples:
        if current:
            gap = sample[0] - current[-1][0]
            width_ratio = max(sample[1], current[-1][1]) / max(
                min(sample[1], current[-1][1]), 1e-6
            )
            if gap > step * 1.6 or width_ratio > 1.40:
                runs.append(current)
                current = []
        current.append(sample)
    if current:
        runs.append(current)
    viable = [run for run in runs if run[-1][0] - run[0][0] >= 0.028]
    if not viable:
        return None
    # Length is the primary affordance. Integrated point support breaks ties,
    # which favours a substantial driver handle over a similarly long shaft.
    best = max(
        viable,
        key=lambda run: (
            run[-1][0] - run[0][0],
            sum(item[3] for item in run),
            np.median([item[1] for item in run]),
        ),
    )
    center_sample = best[len(best) // 2]
    center_xy = origin + major * center_sample[0] + minor * center_sample[2]
    run_length = float(best[-1][0] - best[0][0] + 2.0 * half_window)
    return center_xy, float(np.median([item[1] for item in best])), run_length


def _symmetric_rotation_distance(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Parallel jaws are unchanged by a 180-degree rotation about approach Z."""

    jaw_swap = np.diag([-1.0, -1.0, 1.0])
    return min(rotation_angle_rad(R_a, R_b), rotation_angle_rad(R_a, R_b @ jaw_swap))


def _pose_distance(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])),
        _symmetric_rotation_distance(T_a[:3, :3], T_b[:3, :3]),
    )


def _sanitize_model_pose(raw: Any) -> np.ndarray | None:
    pose = np.asarray(raw, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-3):
        return None
    U, _, Vt = np.linalg.svd(pose[:3, :3])
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0.0:
        U[:, -1] *= -1.0
        rotation = U @ Vt
    if np.linalg.norm(rotation - pose[:3, :3], ord="fro") > 0.08:
        return None
    return make_transform(rotation, pose[:3, 3])


class GraspGenXBackend:
    """GraspGenX proposal generator with camera-frame and temporal stabilization."""

    name = "graspgenx"

    def __init__(
        self,
        config: GraspGenXConfig,
        *,
        transport: GraspGenXTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or ZmqGraspGenXTransport(
            config.host, config.port, config.timeout_ms
        )
        self._previous_target_id: str | None = None
        self._previous_T_base_grasp: np.ndarray | None = None
        self._lock = threading.Lock()

    def generate(
        self,
        observation: RGBDObservation,
        object_points_camera: np.ndarray,
        target: TargetSpec,
    ) -> Sequence[GraspCandidate]:
        points = np.asarray(object_points_camera, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                f"GraspGenX requires camera-frame Nx3 points, got {points.shape}",
                retryable=False,
            )
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < self.config.min_object_points:
            return []
        source_point_count = len(points)
        if source_point_count > self.config.max_object_points:
            # Segmentation produces points in image raster order.  Evenly
            # spaced sampling is deterministic across retries and preserves
            # coverage without importing a second point-cloud framework into
            # the Isaac process.
            sample_indices = np.linspace(
                0,
                source_point_count - 1,
                self.config.max_object_points,
                dtype=np.int64,
            )
            points = points[sample_indices]
        model_points = points
        if self.config.inference_frame == "base":
            model_points = transform_points(observation.T_base_camera, points).astype(np.float32)
        with self._lock:
            grasps, scores, tags = self._infer(observation, model_points)
            proposals = self._proposals(
                observation, model_points, grasps, scores, tags, target
            )
            candidates = self._stabilize(observation, target, proposals)
            return [
                replace(
                    candidate,
                    metadata={
                        **dict(candidate.metadata),
                        "source_object_points": source_point_count,
                        "model_input_points": len(model_points),
                    },
                )
                for candidate in candidates
            ]

    def _infer(
        self, observation: RGBDObservation, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        common = dict(
            seed=int(observation.metadata.get("model_seed", 0)),
            planner=self.config.planner,
            num_grasps=self.config.num_grasps,
            grasp_threshold=self.config.min_confidence,
            topk_num_grasps=self.config.max_candidates,
            return_branch_tags=True,
            approach_context=(observation.metadata.get("approach_context")
                              if self.config.inference_frame == "base" else None),
        )
        try:
            if self.config.request_mode == "object_points":
                result = self.transport.infer_object(
                    points, self.config.sweep_volume.to_wire(), **common
                )
            else:
                if observation.target_mask is None:
                    raise ValueError("scene_depth mode requires observation.target_mask")
                labels = np.where(observation.target_mask, 1, 0).astype(np.int32)
                outputs = self.transport.infer_scene_depth(
                    np.asarray(observation.depth_m, dtype=np.float32),
                    observation.intrinsics.matrix,
                    labels,
                    self.config.sweep_volume.to_wire(),
                    min_object_points=self.config.min_object_points,
                    **common,
                )
                result = outputs.get(1, (np.empty((0, 4, 4)), np.empty((0,)), []))
        except GraspBackendError:
            raise
        except TimeoutError as exc:
            raise GraspBackendError(
                FailureCode.MODEL_TIMEOUT,
                f"GraspGenX inference timed out: {exc}",
                retryable=True,
            ) from exc
        except (ConnectionError, OSError) as exc:
            raise GraspBackendError(
                FailureCode.MODEL_UNAVAILABLE,
                f"GraspGenX server is unavailable: {exc}",
                retryable=True,
            ) from exc
        except ValueError as exc:
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                f"Invalid GraspGenX request or response: {exc}",
                retryable=False,
            ) from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                f"Malformed GraspGenX response: {type(exc).__name__}: {exc}",
                retryable=False,
            ) from exc
        except RuntimeError as exc:
            raise GraspBackendError(
                FailureCode.MODEL_INFERENCE_FAILED,
                f"GraspGenX inference failed: {exc}",
                retryable=True,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - external client boundary
            raise GraspBackendError(
                FailureCode.MODEL_INFERENCE_FAILED,
                f"Unexpected GraspGenX failure: {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc
        try:
            poses = np.asarray(result[0], dtype=np.float64).reshape(-1, 4, 4)
            confidence = np.asarray(result[1], dtype=np.float64).reshape(-1)
            branches = list(result[2]) if len(result) >= 3 else ["unknown"] * len(poses)
        except (ValueError, TypeError, IndexError) as exc:
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                f"Malformed GraspGenX candidate arrays: {exc}",
                retryable=False,
            ) from exc
        if len(poses) != len(confidence) or len(branches) != len(poses):
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                "GraspGenX returned different pose, score and branch counts",
                retryable=False,
            )
        return poses, confidence, [str(branch) for branch in branches]

    def _proposals(
        self,
        observation: RGBDObservation,
        points: np.ndarray,
        grasps: np.ndarray,
        scores: np.ndarray,
        tags: Sequence[str],
        target: TargetSpec,
    ) -> list[_Proposal]:
        proposals = []
        # GraspGenX poses are gripper-base poses.  FineGrasp candidates use a
        # symmetric fingertip/pinch centre so robot-specific EE offsets remain
        # the motion adapter's responsibility.
        T_model_base_pinch = make_transform(
            translation=np.asarray([0.0, 0.0, self.config.sweep_volume.fingertip_depth_m])
        )
        invalid_poses = 0
        preferred_region = str(target.properties.metadata.get("preferred_region", ""))
        # A semantic hint may filter a scored pose, but must not move it while
        # retaining the old discriminator score. Region proposal/rescoring is
        # a separate operation, not a post-hoc pose edit.
        stable_region = None
        for index, (raw_pose, raw_score, branch) in enumerate(zip(grasps, scores, tags)):
            T_model_model_base = _sanitize_model_pose(raw_pose)
            if T_model_model_base is None or not np.isfinite(raw_score):
                invalid_poses += 1
                continue
            T_model_grasp = T_model_model_base @ T_model_base_pinch
            region_strategy = "model_pose"
            region_shift_m = 0.0
            T_base_for_direction = (
                T_model_grasp
                if self.config.inference_frame == "base"
                else observation.T_base_camera @ T_model_grasp
            )
            if (
                stable_region is not None
                and str(branch) == "obb"
                and float(T_base_for_direction[2, 2]) < -0.75
            ):
                original_xy = T_model_grasp[:2, 3].copy()
                T_model_grasp = T_model_grasp.copy()
                T_model_grasp[:2, 3] = stable_region[0]
                region_shift_m = float(np.linalg.norm(T_model_grasp[:2, 3] - original_xy))
                region_strategy = "stable_elongated_handle"
            local_x = (points - T_model_grasp[:3, 3]) @ T_model_grasp[:3, 0]
            low, high = np.percentile(local_x, [5.0, 95.0])
            width = max(float(high - low) + self.config.width_margin_m, 0.0)
            if self.config.inference_frame == "base":
                T_base_grasp = T_model_grasp
                T_camera_grasp = invert_transform(observation.T_base_camera) @ T_base_grasp
            else:
                T_camera_grasp = T_model_grasp
                T_base_grasp = observation.T_base_camera @ T_camera_grasp
            width_strategy = "global_percentile"
            width_section_points = 0
            if float(T_base_grasp[2, 2]) < -0.75:
                width, width_section_points = _conservative_top_down_width(
                    points,
                    T_model_grasp,
                    transverse_span_m=self.config.sweep_volume.extents_mid[1],
                    width_margin_m=self.config.width_margin_m,
                    fallback_width_m=width,
                )
                width_strategy = "fixed_finger_local_section"
            proposals.append(
                _Proposal(
                    T_camera_grasp=T_camera_grasp,
                    T_base_grasp=T_base_grasp,
                    raw_score=float(np.clip(raw_score, 0.0, 1.0)),
                    width_m=width,
                    branch=branch,
                    source_index=index,
                    width_strategy=width_strategy,
                    width_section_points=width_section_points,
                    region_strategy=region_strategy,
                    region_shift_m=region_shift_m,
                )
            )
        if len(grasps) and not proposals:
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                f"All {invalid_poses} GraspGenX transforms were invalid SE(3) poses",
                retryable=False,
            )
        return proposals

    def _stabilize(
        self,
        observation: RGBDObservation,
        target: TargetSpec,
        proposals: list[_Proposal],
    ) -> list[GraspCandidate]:
        if not proposals:
            return []
        parent = list(range(len(proposals)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            a, b = root(a), root(b)
            if a != b:
                parent[max(a, b)] = min(a, b)

        for first in range(len(proposals)):
            for second in range(first + 1, len(proposals)):
                translation, rotation = _pose_distance(
                    proposals[first].T_base_grasp, proposals[second].T_base_grasp
                )
                if (
                    translation <= self.config.cluster_translation_m
                    and rotation <= self.config.cluster_rotation_rad
                ):
                    union(first, second)
        components: dict[int, list[int]] = {}
        for index in range(len(proposals)):
            components.setdefault(root(index), []).append(index)

        def canonical_key(index: int) -> tuple[float, ...]:
            pose = proposals[index].T_base_grasp
            return tuple(np.round(np.concatenate((pose[:3, 3], pose[:3, :3].reshape(-1))), 8))

        ordered_components = sorted(components.values(), key=lambda group: min(canonical_key(i) for i in group))
        for cluster_id, members in enumerate(ordered_components):
            bonus = self.config.consensus_bonus * min((len(members) - 1) / 3.0, 1.0)
            for index in members:
                proposals[index].cluster_id = cluster_id
                proposals[index].cluster_size = len(members)
                proposals[index].consensus_bonus = bonus

        previous = self._previous_T_base_grasp if self._previous_target_id == target.object_id else None
        continuity_indices: list[int] = []
        if previous is not None:
            for index, proposal in enumerate(proposals):
                translation, rotation = _pose_distance(previous, proposal.T_base_grasp)
                if (
                    translation <= self.config.continuity_translation_m
                    and rotation <= self.config.continuity_rotation_rad
                ):
                    closeness = 1.0 - 0.5 * (
                        translation / self.config.continuity_translation_m
                        + rotation / self.config.continuity_rotation_rad
                    )
                    proposal.continuity_bonus = self.config.continuity_bonus * max(closeness, 0.0)
                    continuity_indices.append(index)

        for proposal in proposals:
            proposal.score = float(
                np.clip(
                    proposal.raw_score + proposal.consensus_bonus + proposal.continuity_bonus,
                    0.0,
                    1.0,
                )
            )

        preferred: int | None = None
        if continuity_indices:
            preferred = max(
                continuity_indices,
                key=lambda i: (proposals[i].score, proposals[i].raw_score, tuple(-v for v in canonical_key(i))),
            )
            best = max(range(len(proposals)), key=lambda i: proposals[i].score)
            if (
                best != preferred
                and proposals[best].score < proposals[preferred].score + self.config.switch_score_margin
            ):
                proposals[preferred].score = min(1.0, proposals[best].score + 1e-6)

        order = sorted(
            range(len(proposals)),
            key=lambda index: (
                -proposals[index].score,
                0 if index == preferred else 1,
                -proposals[index].raw_score,
                canonical_key(index),
            ),
        )[: self.config.max_candidates]
        self._previous_target_id = target.object_id
        self._previous_T_base_grasp = proposals[order[0]].T_base_grasp.copy()

        return [
            GraspCandidate(
                T_camera_grasp=proposals[index].T_camera_grasp,
                width_m=proposals[index].width_m,
                score=proposals[index].score,
                observation_sequence=observation.sequence,
                backend=self.name,
                metadata={
                    "raw_model_score": proposals[index].raw_score,
                    "branch": proposals[index].branch,
                    "contact_height_scored": proposals[index].branch == "tilt_scored",
                    "contact_width_m": max(0., proposals[index].width_m - self.config.width_margin_m),
                    "cluster_id": proposals[index].cluster_id,
                    "cluster_size": proposals[index].cluster_size,
                    "consensus_bonus": proposals[index].consensus_bonus,
                    "continuity_bonus": proposals[index].continuity_bonus,
                    "model_source_index": proposals[index].source_index,
                    "model_frame": (
                        observation.base_frame
                        if self.config.inference_frame == "base"
                        else observation.camera_frame
                    ),
                    "model_pose_convention": "T_model_gripper_base,+X_close,+Z_approach",
                    "output_pose_convention": "T_camera_pinch_center,+X_close,+Z_approach",
                    "model_base_to_pinch_center_m": self.config.sweep_volume.fingertip_depth_m,
                    "width_strategy": proposals[index].width_strategy,
                    "width_section_points": proposals[index].width_section_points,
                    "region_strategy": proposals[index].region_strategy,
                    "region_shift_m": proposals[index].region_shift_m,
                    "units": "m",
                },
            )
            for index in order
        ]


class FallbackGraspBackend:
    """Use a local backend when the neural service fails or returns no poses."""

    def __init__(self, primary, fallback, *, fallback_on_empty: bool = True) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_on_empty = bool(fallback_on_empty)
        self.name = f"{primary.name}+{fallback.name}_fallback"
        self.last_failure: GraspBackendError | None = None

    def generate(self, observation, object_points_camera, target):
        self.last_failure = None
        try:
            candidates = list(self.primary.generate(observation, object_points_camera, target))
            if candidates or not self.fallback_on_empty:
                return candidates
            reason = "empty_model_output"
        except GraspBackendError as exc:
            self.last_failure = exc
            reason = exc.failure.value
        fallback = self.fallback.generate(observation, object_points_camera, target)
        return [
            replace(
                candidate,
                backend=self.name,
                metadata={**dict(candidate.metadata), "fallback_reason": reason},
            )
            for candidate in fallback
        ]
