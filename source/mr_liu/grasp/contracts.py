"""Stable BusAgent-facing contracts for the FineGrasp plugin.

The contracts deliberately carry frame names and observation sequence numbers.
This prevents an eye-in-hand estimate from silently being reused after the arm
and camera have moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


Matrix4 = np.ndarray


class FailureCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    NO_OBSERVATION = "no_observation"
    STALE_OBSERVATION = "stale_observation"
    TARGET_NOT_VISIBLE = "target_not_visible"
    INCONSISTENT_OBSERVATION = "inconsistent_observation"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    CALIBRATION_INVALID = "calibration_invalid"
    NO_GRASP_CANDIDATES = "no_grasp_candidates"
    GRIPPER_WIDTH_INFEASIBLE = "gripper_width_infeasible"
    TABLE_CLEARANCE = "table_clearance"
    COLLISION = "collision"
    IK_UNREACHABLE = "ik_unreachable"
    SERVO_TIMEOUT = "servo_timeout"
    TARGET_MOVED = "target_moved"
    GRIPPER_FAILURE = "gripper_failure"
    GRASP_NOT_DETECTED = "grasp_not_detected"
    LIFT_VERIFICATION_FAILED = "lift_verification_failed"
    ABORTED = "aborted"
    INTERNAL_ERROR = "internal_error"


class FineGraspPhase(str, Enum):
    OBSERVE = "observe"
    GENERATE = "generate"
    PREGRASP = "pregrasp"
    SERVO = "servo"
    CLOSE = "close"
    VERIFY_CLOSE = "verify_close"
    LIFT = "lift"
    VERIFY_LIFT = "verify_lift"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera dimensions and focal lengths must be positive")

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ObjectProperties:
    fragile: bool = False
    thin: bool = False
    reflective: bool = False
    deformable: bool = False
    max_grip_force_n: float | None = None
    preferred_grasp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    object_id: str
    semantic_label: str | None = None
    coarse_position_base_m: tuple[float, float, float] | None = None
    properties: ObjectProperties = field(default_factory=ObjectProperties)

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id is required")


@dataclass(frozen=True)
class RGBDObservation:
    sequence: int
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    T_base_camera: Matrix4
    T_base_ee: Matrix4 | None = None
    T_ee_camera: Matrix4 | None = None
    camera_frame: str = "wrist_camera"
    base_frame: str = "base"
    target_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("Observation sequence must be non-negative")
        if self.depth_m.shape != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError("Depth shape does not match camera intrinsics")
        if self.rgb.shape[:2] != self.depth_m.shape:
            raise ValueError("RGB and depth resolutions differ")
        if self.target_mask is not None and self.target_mask.shape != self.depth_m.shape:
            raise ValueError("Target mask shape does not match depth")
        if np.asarray(self.T_base_camera).shape != (4, 4):
            raise ValueError("T_base_camera must be 4x4")
        if (self.T_base_ee is None) != (self.T_ee_camera is None):
            raise ValueError("T_base_ee and T_ee_camera must be supplied together")
        if self.T_base_ee is not None and self.T_ee_camera is not None:
            if np.asarray(self.T_base_ee).shape != (4, 4) or np.asarray(self.T_ee_camera).shape != (4, 4):
                raise ValueError("T_base_ee and T_ee_camera must be 4x4")
            chained = np.asarray(self.T_base_ee) @ np.asarray(self.T_ee_camera)
            if not np.allclose(chained, self.T_base_camera, atol=2e-4):
                raise ValueError("T_base_camera != T_base_ee @ T_ee_camera")


@dataclass(frozen=True)
class GraspCandidate:
    """A candidate expressed in the camera frame of ``observation_sequence``.

    Grasp convention: +X is the finger closing axis, +Z points from the
    gripper toward the object (approach direction), and +Y completes a
    right-handed frame.
    """

    T_camera_grasp: Matrix4
    width_m: float
    score: float
    observation_sequence: int
    backend: str
    contacts_camera: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if np.asarray(self.T_camera_grasp).shape != (4, 4):
            raise ValueError("T_camera_grasp must be 4x4")
        if not np.isfinite(self.width_m) or self.width_m < 0:
            raise ValueError("width_m must be finite and non-negative")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class SelectedGrasp:
    T_base_grasp: Matrix4
    T_base_pregrasp: Matrix4
    T_object_grasp: Matrix4
    width_m: float
    score: float
    observation_sequence: int
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotState:
    timestamp_s: float
    T_base_ee: Matrix4
    joint_positions: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GripperState:
    timestamp_s: float
    width_m: float | None = None
    effort_n: float | None = None
    contact: bool | None = None
    stalled: bool | None = None


@dataclass(frozen=True)
class FineGraspRequest:
    target: TargetSpec
    request_id: str | None = None
    timeout_s: float | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class FineGraspResult:
    success: bool
    phase: FineGraspPhase
    request_id: str | None
    failure: FailureCode | None = None
    message: str = ""
    selected_grasp: SelectedGrasp | None = None
    metrics: Mapping[str, float | int | str] = field(default_factory=dict)
    debug_artifacts: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    confidence: float
    reason: str = ""
    metrics: Mapping[str, float | int | str] = field(default_factory=dict)


class WristCamera(Protocol):
    def capture(self, target: TargetSpec) -> RGBDObservation | None: ...


class TargetSegmenter(Protocol):
    def segment(self, observation: RGBDObservation, target: TargetSpec) -> np.ndarray | None: ...


class GraspBackend(Protocol):
    name: str

    def generate(
        self,
        observation: RGBDObservation,
        object_points_camera: np.ndarray,
        target: TargetSpec,
    ) -> Sequence[GraspCandidate]: ...


class MotionExecutor(Protocol):
    def robot_state(self) -> RobotState: ...

    def is_reachable(self, T_base_ee: Matrix4) -> bool: ...

    def is_collision_free(
        self, T_base_ee: Matrix4, *, margin_m: float, allow_target_contact: bool = False
    ) -> bool: ...

    def move_to(self, T_base_ee: Matrix4, *, speed_scale: float) -> bool: ...

    def servo_to(self, T_base_ee: Matrix4, *, speed_scale: float) -> bool: ...

    def stop(self) -> None: ...


class GripperController(Protocol):
    def open(self, width_m: float, *, speed_mps: float) -> bool: ...

    def close(self, width_m: float, *, force_n: float, speed_mps: float) -> bool: ...

    def state(self) -> GripperState: ...


class GraspVerifier(Protocol):
    def verify_closed(
        self, before: RGBDObservation, after: RGBDObservation, gripper: GripperState
    ) -> VerificationResult: ...

    def verify_lifted(
        self, before: RGBDObservation, after: RGBDObservation, gripper: GripperState
    ) -> VerificationResult: ...
