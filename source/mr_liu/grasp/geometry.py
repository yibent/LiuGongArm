"""RGB-D cleanup, deprojection and local object-frame estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation
from mr_liu.grasp.transforms import invert_transform, make_transform, transform_points


@dataclass(frozen=True)
class ObjectGeometry:
    mask: np.ndarray
    points_camera: np.ndarray
    points_base: np.ndarray
    T_camera_object: np.ndarray
    T_base_object: np.ndarray
    valid_depth_ratio: float
    extents_m: np.ndarray


def recover_small_depth_holes(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    *,
    max_hole_area_px: int = 160,
    boundary_depth_span_m: float = 0.012,
) -> tuple[np.ndarray, int]:
    """Conservatively fill small specular holes inside a target mask.

    Only bounded connected holes with a coherent finite boundary are filled.
    Large missing regions remain invalid so reflective or transparent objects
    are rejected instead of receiving fabricated geometry.
    """
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    target = np.asarray(target_mask, dtype=bool)
    missing = target & (~np.isfinite(depth) | (depth <= 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    filled = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > int(max_hole_area_px):
            continue
        component = labels == label
        ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=2).astype(bool)
        ring &= ~component & target & np.isfinite(depth) & (depth > 0)
        boundary = np.asarray(depth[ring], dtype=np.float64)
        if boundary.size < 8:
            continue
        low, high = np.percentile(boundary, [10.0, 90.0])
        if float(high - low) > float(boundary_depth_span_m):
            continue
        depth[component] = float(np.median(boundary))
        filled += area
    return depth, filled


def support_completed_center(
    geometry: ObjectGeometry,
    *,
    table_height_m: float,
) -> np.ndarray:
    """Return a stable centre for a partially visible table-supported object.

    A wrist view first sees mostly the top face and later exposes side faces.
    The centroid of only visible points therefore drifts along the optical axis
    even when the object is motionless.  The robust top height and known support
    plane complete the hidden lower half without assuming a trained category.
    """
    center = np.asarray(geometry.T_base_object[:3, 3], dtype=np.float64).copy()
    top_z = float(np.percentile(geometry.points_base[:, 2], 95.0))
    height = top_z - float(table_height_m)
    if 0.004 < height < 0.25:
        center[2] = float(table_height_m) + 0.5 * height
    return center


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
    # A surface median is strongly view-dependent: a wrist camera transitioning
    # from a top view to a side view can shift it by several centimetres even
    # when the object is perfectly static.  The midpoint of robust base-frame
    # bounds is substantially more stable when multiple object faces are
    # visible and keeps the tracking translation independent of camera axes.
    lower, upper = np.percentile(points_base, [5.0, 95.0], axis=0)
    T_base_object = T_base_object.copy()
    T_base_object[:3, 3] = 0.5 * (lower + upper)
    T_camera_object = invert_transform(observation.T_base_camera) @ T_base_object
    return ObjectGeometry(
        mask=valid,
        points_camera=points_camera,
        points_base=points_base,
        T_camera_object=T_camera_object,
        T_base_object=T_base_object,
        valid_depth_ratio=ratio,
        extents_m=extents,
    )
