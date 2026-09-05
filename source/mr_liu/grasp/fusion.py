"""Bounded eye-in-hand surface fusion with explicit free/unknown space.

This is a voxel-downsampled surface map, not a watertight completion or TSDF.
Depth rays are retained so missing pixels and occluded space remain unknown.
No simulator object geometry or ground-truth object poses enter this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial import cKDTree

from mr_liu.grasp.contracts import RGBDObservation
from mr_liu.grasp.geometry import ObjectGeometry, estimate_object_frame
from mr_liu.grasp.transforms import invert_transform, transform_points


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = False
    voxel_m: float = 0.002
    max_views: int = 6
    max_age_s: float = 20.0
    min_baseline_m: float = 0.012
    surface_tolerance_m: float = 0.004
    motion_tolerance_m: float = 0.008
    max_free_conflict_ratio: float = 0.30

    def __post_init__(self):
        if min(self.voxel_m, self.max_age_s, self.min_baseline_m,
               self.surface_tolerance_m, self.motion_tolerance_m) <= 0 or self.max_views < 1:
            raise ValueError("Invalid fusion limits")


def voxel_centers(points: np.ndarray, size_m: float) -> np.ndarray:
    keys = np.floor(points / size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    total = np.zeros((len(counts), 3))
    np.add.at(total, inverse, points)
    return total / counts[:, None]


def depth_evidence(points_base: np.ndarray, observation: RGBDObservation,
                   tolerance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed-free, observed-surface, valid-ray flags per query point.

    Points behind a measured surface are occluded/unknown, not occupied/free.
    A missing depth pixel provides no evidence in either direction.
    """
    pc = transform_points(invert_transform(observation.T_base_camera), points_base)
    k = observation.intrinsics
    z = pc[:, 2]
    positive = np.isfinite(pc).all(axis=1) & (z > 1e-6)
    safe_z = np.where(positive, z, 1.0)
    u = np.rint(k.fx * pc[:, 0] / safe_z + k.cx).astype(int)
    v = np.rint(k.fy * pc[:, 1] / safe_z + k.cy).astype(int)
    inside = positive & (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)
    measured = np.full(len(pc), np.nan)
    measured[inside] = observation.depth_m[v[inside], u[inside]]
    valid = inside & np.isfinite(measured) & (measured > 0)
    return (valid & (z < measured - tolerance_m),
            valid & (np.abs(z - measured) <= tolerance_m), valid)


class LocalSurfaceFusion:
    def __init__(self, config: FusionConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.views: list[tuple[RGBDObservation, np.ndarray]] = []
        self.object_id: str | None = None
        self.last_sequence = -1
        self.last_timestamp = -float("inf")
        self.last_reset_reason = "initial"

    @property
    def points_base(self) -> np.ndarray:
        if not self.views:
            return np.empty((0, 3))
        return voxel_centers(np.concatenate([p for _, p in self.views]), self.config.voxel_m)

    def integrate(self, observation: RGBDObservation, geometry: ObjectGeometry,
                  object_id: str) -> dict[str, float | int | str]:
        if observation.T_base_ee is None or observation.T_ee_camera is None:
            raise ValueError("Fusion requires synchronized wrist hand-eye transforms")
        if observation.sequence <= self.last_sequence or observation.timestamp_s <= self.last_timestamp:
            raise ValueError("Fusion rejected non-increasing observation time/sequence")
        if self.object_id is not None and self.object_id != object_id:
            self.reset()
            self.last_reset_reason = "identity_changed"
        self.object_id = object_id
        self.views = [(o, p) for o, p in self.views
                      if observation.timestamp_s - o.timestamp_s <= self.config.max_age_s]
        previous = self.points_base
        current = voxel_centers(geometry.points_base, self.config.voxel_m)
        reset_reason = "none"
        conflict = 0.0
        if len(previous) >= 16 and len(current) >= 16:
            free, surface, _ = depth_evidence(previous, observation, self.config.motion_tolerance_m)
            observable = free | surface
            conflict = float(free.sum() / max(observable.sum(), 1))
            distances, _ = cKDTree(previous).query(current)
            overlap = float(np.mean(distances < self.config.motion_tolerance_m))
            if conflict > self.config.max_free_conflict_ratio or overlap < 0.15:
                self.views.clear()
                reset_reason = "motion_or_identity_inconsistent"
                self.last_reset_reason = reset_reason
        baseline = min((np.linalg.norm(observation.T_base_camera[:3, 3] - o.T_base_camera[:3, 3])
                        for o, _ in self.views), default=float("inf"))
        # Replace a redundant view so the most recent image is always retained,
        # without claiming repeated stationary captures as independent views.
        if self.views and baseline < self.config.min_baseline_m:
            nearest = int(np.argmin([np.linalg.norm(observation.T_base_camera[:3, 3] - o.T_base_camera[:3, 3])
                                    for o, _ in self.views]))
            self.views[nearest] = (observation, current)
        else:
            self.views.append((observation, current))
        self.views = self.views[-self.config.max_views:]
        self.last_sequence = observation.sequence
        self.last_timestamp = observation.timestamp_s
        return {"fusion_views": len(self.views), "fusion_points": len(self.points_base),
                "fusion_free_conflict_ratio": conflict, "fusion_reset": reset_reason}

    def geometry(self, latest: RGBDObservation, current: ObjectGeometry) -> ObjectGeometry:
        cloud = self.points_base
        if len(cloud) < 6:
            return current
        pose, extents = estimate_object_frame(cloud)
        lower, upper = np.percentile(cloud, [3, 97], axis=0)
        pose[:3, 3] = 0.5 * (lower + upper)
        camera_pose = invert_transform(latest.T_base_camera) @ pose
        return replace(current, points_base=cloud,
                       points_camera=transform_points(invert_transform(latest.T_base_camera), cloud),
                       T_base_object=pose, T_camera_object=camera_pose, extents_m=extents)

    def observed_free(self, points_base: np.ndarray) -> np.ndarray:
        free = np.zeros(len(points_base), dtype=bool)
        occupied = np.zeros(len(points_base), dtype=bool)
        for observation, _ in self.views:
            view_free, surface, _ = depth_evidence(points_base, observation, self.config.surface_tolerance_m)
            free |= view_free
            occupied |= surface
        return free & ~occupied


@dataclass(frozen=True)
class ActiveViewConfig:
    enabled: bool = False
    max_views: int = 3
    offset_m: float = 0.025
    speed_scale: float = 0.25
    min_views: int = 2

    def __post_init__(self):
        if not 1 <= self.min_views <= self.max_views <= 6 or not 0 < self.offset_m <= 0.05:
            raise ValueError("Invalid active-view bounds")


def propose_wrist_views(observation: RGBDObservation, target_center: np.ndarray,
                        config: ActiveViewConfig) -> list[np.ndarray]:
    if observation.T_base_ee is None or observation.T_ee_camera is None:
        return []
    proposals = []
    # Small base-frame lateral moves preserve the current approach orientation.
    # They are only proposals: IK + world + local swept geometry must accept.
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        pose = observation.T_base_ee.copy()
        pose[:3, 3] += np.asarray([dx * config.offset_m, dy * config.offset_m, 0.005])
        camera_pose = pose @ observation.T_ee_camera
        pc = transform_points(invert_transform(camera_pose), target_center[None])[0]
        if pc[2] <= 0.03:
            continue
        k = observation.intrinsics
        u, v = k.fx * pc[0] / pc[2] + k.cx, k.fy * pc[1] / pc[2] + k.cy
        if 0.15 * k.width < u < 0.85 * k.width and 0.15 * k.height < v < 0.85 * k.height:
            proposals.append(pose)
    return proposals
