"""M2T2 pick/place adapter.

The Isaac/BusAgent process deliberately stays torch-free.  ``M2T2Backend``
uses the same small request/response boundary as GraspGenX and talks to the
official M2T2 implementation in :mod:`scripts.serve_m2t2`.  This keeps the
existing FineGrasp segmentation, IK, collision and verification gates in the
control path while allowing M2T2 to provide both grasp and placement poses.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from mr_liu.grasp.contracts import (
    FailureCode,
    GraspBackendError,
    GraspCandidate,
    PlaceRequest,
    RGBDObservation,
    TargetSpec,
)
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import invert_transform, make_transform, transform_points


@dataclass(frozen=True)
class M2T2Config:
    host: str = "127.0.0.1"
    port: int = 5562
    timeout_ms: int = 5000
    model_frame: str = "base"
    checkpoint: str = "_models/m2t2/m2t2.pth"
    max_scene_points: int = 16384
    max_object_points: int = 1024
    max_candidates: int = 64
    min_confidence: float = 0.0
    contact_radius_m: float = 0.030
    width_margin_m: float = 0.010
    gripper_depth_m: float = 0.1034

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "M2T2Config":
        result = cls(
            host=str(value.get("host", cls.host)),
            port=int(value.get("port", cls.port)),
            timeout_ms=int(value.get("timeout_ms", cls.timeout_ms)),
            model_frame=str(value.get("model_frame", cls.model_frame)),
            checkpoint=str(value.get("checkpoint", cls.checkpoint)),
            max_scene_points=int(value.get("max_scene_points", cls.max_scene_points)),
            max_object_points=int(value.get("max_object_points", cls.max_object_points)),
            max_candidates=int(value.get("max_candidates", cls.max_candidates)),
            min_confidence=float(value.get("min_confidence", cls.min_confidence)),
            contact_radius_m=float(value.get("contact_radius_m", cls.contact_radius_m)),
            width_margin_m=float(value.get("width_margin_m", cls.width_margin_m)),
            gripper_depth_m=float(value.get("gripper_depth_m", cls.gripper_depth_m)),
        )
        if not result.host or not 0 < result.port < 65536 or result.timeout_ms <= 0:
            raise ValueError("Invalid M2T2 server address or timeout")
        if result.model_frame not in {"base", "camera"}:
            raise ValueError("M2T2 model_frame must be base or camera")
        if min(result.max_scene_points, result.max_object_points, result.max_candidates) <= 0:
            raise ValueError("M2T2 point and candidate limits must be positive")
        if result.min_confidence < 0.0 or result.contact_radius_m <= 0.0:
            raise ValueError("M2T2 confidence and contact radius are invalid")
        return result


class M2T2Transport(Protocol):
    def infer_pick(
        self,
        scene_points: np.ndarray,
        object_points: np.ndarray,
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...

    def infer_place(
        self,
        scene_points: np.ndarray,
        object_points: np.ndarray,
        ee_pose: np.ndarray,
        place_request: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...


class ZmqM2T2Transport:
    """Torch-free msgpack transport for the isolated M2T2 server."""

    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.host, self.port, self.timeout_ms = str(host), int(port), int(timeout_ms)
        self._socket = None
        self._lock = threading.Lock()
        self._msgpack = None
        self._zmq = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

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
                "M2T2 client dependencies are missing; install requirements-m2t2-client.txt",
                retryable=False,
            ) from exc
        msgpack_numpy.patch()
        self._msgpack, self._zmq = msgpack, zmq

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

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            self._connect()
            assert self._msgpack is not None and self._zmq is not None
            try:
                self._socket.send(self._msgpack.packb(dict(payload), use_bin_type=True))
                raw = self._socket.recv()
            except Exception as exc:
                again = isinstance(exc, self._zmq.error.Again)
                self.close()
                if again:
                    raise TimeoutError(f"M2T2 server {self.address} timed out") from exc
                raise ConnectionError(f"M2T2 transport failed at {self.address}: {exc}") from exc
        response = self._msgpack.unpackb(raw, raw=False)
        if not isinstance(response, Mapping):
            raise ValueError("M2T2 response must be a mapping")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return response

    def infer_pick(self, scene_points, object_points, **kwargs):
        return self._request({
            "action": "infer_pick",
            "scene_points": np.asarray(scene_points, dtype=np.float32),
            "object_points": np.asarray(object_points, dtype=np.float32),
            "model_frame": str(kwargs.get("model_frame", "base")),
            "seed": int(kwargs.get("seed", 0)),
            "max_candidates": int(kwargs.get("max_candidates", 64)),
            "min_confidence": float(kwargs.get("min_confidence", 0.0)),
        })

    def infer_place(self, scene_points, object_points, ee_pose, place_request, **kwargs):
        return self._request({
            "action": "infer_place",
            "scene_points": np.asarray(scene_points, dtype=np.float32),
            "object_points": np.asarray(object_points, dtype=np.float32),
            "ee_pose": np.asarray(ee_pose, dtype=np.float32),
            "place_request": dict(place_request),
            "model_frame": str(kwargs.get("model_frame", "base")),
            "seed": int(kwargs.get("seed", 0)),
            "max_candidates": int(kwargs.get("max_candidates", 64)),
            "min_confidence": float(kwargs.get("min_confidence", 0.0)),
        })


def _sample_points(points: np.ndarray, limit: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def _scene_points(observation: RGBDObservation, limit: int, model_frame: str) -> np.ndarray:
    valid = np.isfinite(observation.depth_m) & (observation.depth_m > 0)
    points = deproject_depth(observation.depth_m, observation.intrinsics, valid)
    if model_frame == "base":
        points = transform_points(observation.T_base_camera, points)
    return _sample_points(points, limit)


def _sanitize_pose(raw: Any) -> np.ndarray | None:
    pose = np.asarray(raw, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    try:
        return make_transform(pose[:3, :3], pose[:3, 3])
    except ValueError:
        return None


def _width_for_grasp(points: np.ndarray, pose: np.ndarray, margin: float) -> float:
    local = (np.asarray(points, dtype=np.float64) - pose[:3, 3]) @ pose[:3, :3]
    if len(local) < 8:
        return 0.020
    low, high = np.percentile(local[:, 0], [3.0, 97.0])
    return max(0.004, float(high - low) + float(margin))


class M2T2Backend:
    """Generate target-filtered camera-frame candidates from M2T2."""

    name = "m2t2"

    def __init__(self, config: M2T2Config, *, transport: M2T2Transport | None = None) -> None:
        self.config = config
        self.transport = transport or ZmqM2T2Transport(config.host, config.port, config.timeout_ms)
        self._lock = threading.Lock()

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def generate(self, observation, object_points_camera, target) -> Sequence[GraspCandidate]:
        target_points = _sample_points(object_points_camera, self.config.max_object_points)
        if len(target_points) < 6:
            return []
        scene = _scene_points(observation, self.config.max_scene_points, self.config.model_frame)
        if self.config.model_frame == "base":
            object_for_model = transform_points(observation.T_base_camera, target_points)
        else:
            object_for_model = target_points
        try:
            with self._lock:
                response = self.transport.infer_pick(
                    scene,
                    object_for_model,
                    seed=int(observation.metadata.get("model_seed", 0)),
                    model_frame=self.config.model_frame,
                    max_candidates=self.config.max_candidates,
                    min_confidence=self.config.min_confidence,
                )
        except GraspBackendError:
            raise
        except TimeoutError as exc:
            raise GraspBackendError(FailureCode.MODEL_TIMEOUT, str(exc), retryable=True) from exc
        except (ConnectionError, ImportError) as exc:
            raise GraspBackendError(FailureCode.MODEL_UNAVAILABLE, str(exc), retryable=True) from exc
        except Exception as exc:
            raise GraspBackendError(FailureCode.MODEL_INFERENCE_FAILED, str(exc), retryable=True) from exc

        response_frame = str(response.get("model_frame", self.config.model_frame))
        if response_frame != self.config.model_frame:
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                f"M2T2 response frame {response_frame!r} does not match configured {self.config.model_frame!r}",
                retryable=False,
            )
        raw_poses = response.get("grasps", [])
        scores = np.asarray(response.get("confidences", []), dtype=np.float64).reshape(-1)
        contacts = np.asarray(response.get("contacts", []), dtype=np.float64).reshape(-1, 3) if response.get("contacts") is not None else np.empty((0, 3))
        if len(raw_poses) != len(scores):
            raise GraspBackendError(FailureCode.MODEL_PROTOCOL_ERROR, "M2T2 pose/score counts differ", retryable=False)
        candidates: list[GraspCandidate] = []
        for index, (raw, score) in enumerate(zip(raw_poses, scores)):
            pose_model = _sanitize_pose(raw)
            if pose_model is None or not np.isfinite(score) or score < self.config.min_confidence:
                continue
            if len(contacts) != len(raw_poses):
                # M2T2's generic model predicts scene-wide queries. Without a
                # contact point for each pose there is no safe way to associate
                # a query with the user-selected object.
                continue
            contact = contacts[index]
            contact_target = object_for_model[np.linalg.norm(object_for_model - contact, axis=1).argmin()]
            if float(np.linalg.norm(contact - contact_target)) > self.config.contact_radius_m:
                continue
            if self.config.model_frame == "base":
                pose_base = pose_model
                pose_camera = invert_transform(observation.T_base_camera) @ pose_base
                width = _width_for_grasp(object_for_model, pose_base, self.config.width_margin_m)
                contact_camera = transform_points(
                    invert_transform(observation.T_base_camera), contact.reshape(1, 3)
                )
            else:
                pose_camera = pose_model
                width = _width_for_grasp(target_points, pose_camera, self.config.width_margin_m)
                pose_base = observation.T_base_camera @ pose_camera
                contact_camera = contact.reshape(1, 3)
            candidates.append(GraspCandidate(
                T_camera_grasp=pose_camera,
                width_m=float(width),
                score=float(np.clip(score, 0.0, 1.0)),
                observation_sequence=observation.sequence,
                backend=self.name,
                contacts_camera=contact_camera,
                metadata={
                    "model": "M2T2",
                    "model_frame": self.config.model_frame,
                    "model_pose_convention": "T_model_gripper,+X_close,+Z_approach",
                    "output_pose_convention": "T_camera_pinch_center,+X_close,+Z_approach",
                    "source_index": index,
                    "scene_points": len(scene),
                    "target_points": len(target_points),
                    "gripper_depth_m": self.config.gripper_depth_m,
                    "target_id": target.object_id,
                },
            ))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[: self.config.max_candidates]


class M2T2PlacementPlanner:
    """Placement candidate adapter used after a verified M2T2 grasp."""

    def __init__(self, config: M2T2Config, *, transport: M2T2Transport | None = None) -> None:
        self.config = config
        self.transport = transport or ZmqM2T2Transport(config.host, config.port, config.timeout_ms)

    def generate(
        self,
        observation: RGBDObservation,
        held_object_points_camera: np.ndarray,
        current_ee_pose_base: np.ndarray,
        request: PlaceRequest,
    ) -> list[tuple[np.ndarray, float]]:
        scene = _scene_points(observation, self.config.max_scene_points, self.config.model_frame)
        object_points = _sample_points(held_object_points_camera, self.config.max_object_points)
        if self.config.model_frame == "base":
            object_points = transform_points(observation.T_base_camera, object_points)
        else:
            current_ee_pose_base = invert_transform(observation.T_base_camera) @ current_ee_pose_base
        payload = {
            "center_base_m": tuple(float(v) for v in request.center_base_m),
            "size_xy_m": tuple(float(v) for v in request.size_xy_m),
            "surface_z_m": request.surface_z_m,
        }
        try:
            response = self.transport.infer_place(
                scene, object_points, current_ee_pose_base, payload,
                seed=int(observation.metadata.get("model_seed", 0)),
                model_frame=self.config.model_frame,
                max_candidates=self.config.max_candidates,
                min_confidence=self.config.min_confidence,
            )
        except GraspBackendError:
            raise
        except TimeoutError as exc:
            raise GraspBackendError(FailureCode.MODEL_TIMEOUT, str(exc), retryable=True) from exc
        except (ConnectionError, ImportError) as exc:
            raise GraspBackendError(FailureCode.MODEL_UNAVAILABLE, str(exc), retryable=True) from exc
        except Exception as exc:
            raise GraspBackendError(FailureCode.MODEL_INFERENCE_FAILED, str(exc), retryable=True) from exc
        response_frame = str(response.get("model_frame", self.config.model_frame))
        if response_frame != self.config.model_frame:
            raise GraspBackendError(
                FailureCode.MODEL_PROTOCOL_ERROR,
                f"M2T2 placement response frame {response_frame!r} does not match configured {self.config.model_frame!r}",
                retryable=False,
            )
        poses = response.get("placements", [])
        scores = np.asarray(response.get("confidences", []), dtype=np.float64).reshape(-1)
        if len(poses) != len(scores):
            raise GraspBackendError(FailureCode.MODEL_PROTOCOL_ERROR, "M2T2 placement pose/score counts differ", retryable=False)
        result: list[tuple[np.ndarray, float]] = []
        center = np.asarray(request.center_base_m, dtype=float)
        half = np.asarray(request.size_xy_m, dtype=float) * 0.5
        for raw, score in zip(poses, scores):
            pose = _sanitize_pose(raw)
            if pose is None or not np.isfinite(score) or score < self.config.min_confidence:
                continue
            if self.config.model_frame == "camera":
                pose = observation.T_base_camera @ pose
            xy = pose[:2, 3]
            if np.any(np.abs(xy - center[:2]) > half + 0.010):
                continue
            if request.surface_z_m is not None and pose[2, 3] < float(request.surface_z_m) - 0.005:
                continue
            result.append((pose, float(np.clip(score, 0.0, 1.0))))
        return sorted(result, key=lambda item: item[1], reverse=True)[: self.config.max_candidates]
