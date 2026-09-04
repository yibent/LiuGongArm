"""RGB-D cleanup, deprojection and local object-frame estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation
from mr_liu.grasp.transforms import make_transform, transform_points


@dataclass(frozen=True)
class ObjectGeometry:
    mask: np.ndarray
    points_camera: np.ndarray
    points_base: np.ndarray
    T_camera_object: np.ndarray
    T_base_object: np.ndarray
    valid_depth_ratio: float
    extents_m: np.ndarray


def self_occlusion_mask(shape: tuple[int, int], bottom_fraction: float) -> np.ndarray:
    height, width = shape
    fraction = float(np.clip(bottom_fraction, 0.0, 0.95))
    mask = np.ones((height, width), dtype=bool)
    if fraction > 0.0:
        mask[int(round(height * (1.0 - fraction))) :, :] = False
    return mask


def robust_depth_mask(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    mad_scale: float,
) -> tuple[np.ndarray, float]:
    depth = np.asarray(depth_m, dtype=np.float64)
    requested = np.asarray(target_mask, dtype=bool)
    valid = requested & np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    ratio = float(valid.sum() / max(int(requested.sum()), 1))
    values = depth[valid]
    if values.size >= 8:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 1e-6:
            valid &= np.abs(depth - median) <= mad_scale * 1.4826 * mad
    return valid, ratio


def deproject_depth(depth_m: np.ndarray, intrinsics: CameraIntrinsics, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = np.asarray(depth_m, dtype=np.float64)[rows, cols]
    x = (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z))


def estimate_object_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a right-handed PCA frame and axis-aligned extents.

    Eigenvector signs are made deterministic relative to camera axes to reduce
    frame flips. Temporal sign continuity can be added by a tracker adapter.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or cloud.shape[0] < 6:
        raise ValueError("At least six 3D points are required")
    center = np.median(cloud, axis=0)
    centered = cloud - center
    covariance = centered.T @ centered / max(cloud.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    for column in range(2):
        dominant = int(np.argmax(np.abs(axes[:, column])))
        if axes[dominant, column] < 0:
            axes[:, column] *= -1.0
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    axes[:, 2] /= max(float(np.linalg.norm(axes[:, 2])), 1e-12)
    axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
    projected = centered @ axes
    extents = np.percentile(projected, 95, axis=0) - np.percentile(projected, 5, axis=0)
    return make_transform(axes, center), np.maximum(extents, 0.0)


def extract_object_geometry(
    observation: RGBDObservation,
    mask: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    mad_scale: float,
    self_mask_bottom_fraction: float,
) -> ObjectGeometry:
    requested = np.asarray(mask, dtype=bool)
    requested &= self_occlusion_mask(requested.shape, self_mask_bottom_fraction)
    valid, ratio = robust_depth_mask(
        observation.depth_m,
        requested,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        mad_scale=mad_scale,
    )
    points_camera = deproject_depth(observation.depth_m, observation.intrinsics, valid)
    T_camera_object, extents = estimate_object_frame(points_camera)
    T_base_object = observation.T_base_camera @ T_camera_object
    points_base = transform_points(observation.T_base_camera, points_camera)
    return ObjectGeometry(
        mask=valid,
        points_camera=points_camera,
        points_base=points_base,
        T_camera_object=T_camera_object,
        T_base_object=T_base_object,
        valid_depth_ratio=ratio,
        extents_m=extents,
    )
