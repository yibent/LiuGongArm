"""Placement inputs use metric base coordinates and explicit held-object state."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from mr_liu.grasp.contracts import RGBDObservation
from mr_liu.grasp.transforms import assert_transform


class PlaceError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class PlaceRequest:
    destination_label: str
    relation: str = "on"                 # on a visible region, inside, or relative
    offset_base_m: tuple[float, float] = (0., 0.)
    request_id: str = "place"
    dry_run: bool = False

    def __post_init__(self):
        if not self.destination_label.strip() or len(self.destination_label) > 256:
            raise ValueError("A destination label is required")
        if self.relation not in {"on", "inside", "relative"}:
            raise ValueError("Unsupported relation")
        offset = np.asarray(self.offset_base_m, float)
        if offset.shape != (2,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .4:
            raise ValueError("offset_base_m is a bounded XY offset in the calibrated base frame")
        if self.relation != "relative" and np.any(offset):
            raise ValueError("Only relative placement takes an offset")


@dataclass(frozen=True)
class HeldObject:
    object_id: str
    T_ee_object: np.ndarray
    half_extents_m: np.ndarray
    reference_points_object: np.ndarray
    chromaticity: np.ndarray
    verified_at_s: float
    # A stable base is an explicit upstream assertion, not inferred from a caption.
    stable_base_verified: bool = False
    uncertainty_m: float = .003

    def __post_init__(self):
        assert_transform(self.T_ee_object, name="T_ee_object")
        half = np.asarray(self.half_extents_m)
        points = np.asarray(self.reference_points_object)
        if not self.object_id or half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError("Invalid held geometry")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32 or not np.isfinite(points).all():
            raise ValueError("Measured held-object reference cloud required")
        if np.asarray(self.chromaticity).shape != (3,) or not np.isfinite(self.chromaticity).all():
            raise ValueError("Invalid identity anchor")
        if not np.isfinite(self.verified_at_s) or not 0 < self.uncertainty_m <= .005:
            raise ValueError("Invalid held-object uncertainty/time")


@dataclass(frozen=True)
class PlaceSettings:
    max_age_s: float = .35
    stationary_release_max_age_s: float = .35
    timeout_s: float = 120.
    max_step_m: float = .006
    max_fine_travel_m: float = .12
    preplace_height_m: float = .045
    release_gap_m: float = .003
    release_gap_tolerance_m: float = .002
    xy_tolerance_m: float = .003
    max_object_tilt_deg: float = 5.
    boundary_margin_m: float = .010
    max_destination_shift_m: float = .012
    max_slip_m: float = .008
    open_width_m: float = .075
    open_speed_mps: float = .015
    speed_scale: float = .20
    max_iterations: int = 60
    settle_frames: int = 5
    settle_min_s: float = .35

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid positive setting: {name}")
        if self.max_step_m > .01 or self.release_gap_m + self.release_gap_tolerance_m > .008:
            raise ValueError("Fine-place steps/release gap exceed bounded V1 envelope")
        if self.settle_frames < 3 or self.max_iterations < 1:
            raise ValueError("Insufficient verification/iteration budget")
        if not self.max_age_s <= self.stationary_release_max_age_s <= 1.:
            raise ValueError("Stationary release age must be between max_age_s and 1 second")


@dataclass
class DestinationEvidence:
    observation: RGBDObservation
    surface_points: np.ndarray
    scene_points: np.ndarray
    center_base_m: np.ndarray
    support_z_m: float
    mask: np.ndarray
    description: dict | None = None


@dataclass
class PayloadEvidence:
    observation: RGBDObservation
    T_base_object: np.ndarray
    points_base: np.ndarray
    residual_m: float


@dataclass
class PlaceResult:
    success: bool
    phase: str
    request_id: str
    failure: str | None = None
    message: str = ""
    released: bool = False
    verified: bool = False
    metrics: dict = field(default_factory=dict)


class PlacePerception(Protocol):
    def destination(self, request: PlaceRequest) -> DestinationEvidence: ...
    def payload(self, held: HeldObject, expected: np.ndarray, *, released=False) -> PayloadEvidence: ...


class PlaceMotion(Protocol):
    def robot_state(self): ...
    def move_checked(self, goal: np.ndarray, held: HeldObject | None,
                     evidence: DestinationEvidence, *, opening: bool = False) -> bool: ...
    def opening_safe(self, held: HeldObject, evidence: DestinationEvidence) -> bool: ...
    def retreat_target(self) -> np.ndarray: ...
    def stop(self): ...
