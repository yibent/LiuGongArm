"""RGB-D cleanup, deprojection and local object-frame estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2
from scipy.spatial import cKDTree

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


@dataclass(frozen=True)
class TranslationRegistration:
    """Robust current-object translation relative to a reference wrist cloud."""

    translation_base_m: np.ndarray
    residual_m: float
    inlier_ratio: float
    reliable: bool


@dataclass(frozen=True)
class RigidRegistration:
    """Bounded 6D correction from an expected pose to a fresh RGB-D cloud."""

    T_reference_current: np.ndarray
    residual_m: float
    inlier_ratio: float
    translation_m: float
    rotation_rad: float
    reliable: bool


def register_object_rigid(
    reference_points_base: np.ndarray,
    current_points_base: np.ndarray,
    *,
    max_points: int = 2048,
    max_correspondence_m: float = 0.012,
    min_inliers: int = 48,
    max_iterations: int = 10,
    max_translation_m: float = 0.012,
    max_rotation_rad: float = np.deg2rad(12.0),
) -> RigidRegistration:
    """Estimate a small rigid deviation around an already predicted pose.

    Placement supplies the object pose predicted from current Franka FK and
    the measured grasp attachment, so this routine is intentionally local. It
    detects translational and rotational slip; it is not a global pose finder.
    Current-to-reference matching tolerates a fresh view containing only part
    of the original measured surface.
    """
    reference = np.asarray(reference_points_base, dtype=np.float64).reshape(-1, 3)
    current = np.asarray(current_points_base, dtype=np.float64).reshape(-1, 3)
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    current = current[np.all(np.isfinite(current), axis=1)]
    identity = np.eye(4, dtype=np.float64)
    if (
        max_points < min_inliers
        or reference.shape[0] < min_inliers
        or current.shape[0] < min_inliers
    ):
        return RigidRegistration(identity, float("inf"), 0.0, 0.0, 0.0, False)

    def sampled(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        return points[indices]

    reference, current = sampled(reference), sampled(current)

    def has_observable_rotation(points: np.ndarray) -> bool:
        """Reject clouds whose rotation is underconstrained (point/line-like)."""
        centered = points - np.median(points, axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values.size < 2:
            return False
        # A plane is sufficient for a local point-to-point registration, but a
        # line cannot constrain rotation about itself.  The absolute floor also
        # prevents sensor noise around a line from masquerading as geometry.
        normalizer = max(float(np.sqrt(points.shape[0] - 1)), 1.0)
        principal_std = singular_values[:2] / normalizer
        return bool(
            principal_std[0] >= 5e-4
            and principal_std[1] >= max(5e-4, 0.01 * principal_std[0])
        )

    if not has_observable_rotation(reference) or not has_observable_rotation(current):
        return RigidRegistration(identity, float("inf"), 0.0, 0.0, 0.0, False)

    reference_center = np.mean(reference, axis=0)
    current_center = np.mean(current, axis=0)
    tree = cKDTree(reference)

    def solve(initial_translation: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool
    ]:
        R_current_reference = np.eye(3, dtype=np.float64)
        t_current_reference = np.asarray(initial_translation, dtype=np.float64).copy()
        correspondence_geometry_valid = False
        for _ in range(max_iterations):
            aligned = current @ R_current_reference.T + t_current_reference
            distances, indices = tree.query(aligned, k=1)
            inliers = distances <= max_correspondence_m
            if int(inliers.sum()) < min_inliers:
                break
            source = aligned[inliers]
            target = reference[indices[inliers]]
            correspondence_geometry_valid = (
                has_observable_rotation(source) and has_observable_rotation(target)
            )
            if not correspondence_geometry_valid:
                break
            source_center, target_center = source.mean(0), target.mean(0)
            U, _values, Vt = np.linalg.svd(
                (source - source_center).T @ (target - target_center),
                full_matrices=False,
            )
            rotation = Vt.T @ U.T
            if np.linalg.det(rotation) < 0:
                Vt[-1] *= -1
                rotation = Vt.T @ U.T
            translation = target_center - rotation @ source_center
            R_current_reference = rotation @ R_current_reference
            t_current_reference = rotation @ t_current_reference + translation
            rotation_step = np.arccos(
                np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
            )
            if np.linalg.norm(translation) < 1e-5 and rotation_step < np.deg2rad(0.02):
                break

        aligned = current @ R_current_reference.T + t_current_reference
        distances, indices = tree.query(aligned, k=1)
        inliers = distances <= max_correspondence_m
        if int(inliers.sum()) >= min_inliers:
            correspondence_geometry_valid = (
                has_observable_rotation(aligned[inliers])
                and has_observable_rotation(reference[indices[inliers]])
            )
        else:
            correspondence_geometry_valid = False
        return (
            R_current_reference,
            t_current_reference,
            distances,
            inliers,
            correspondence_geometry_valid,
        )

    # Identity is the expected local seed.  A second centre-aligned seed avoids
    # the well-known ICP failure where two overlapping dense objects settle on
    # the overlap instead of recovering even a modest translation.  Running
    # both retains the partial-view behaviour for which centroids can be biased.
    seeds = (np.zeros(3, dtype=np.float64), reference_center - current_center)
    solutions = [solve(seed) for seed in seeds]

    def solution_score(solution: tuple) -> tuple[float, float]:
        distances, inliers = solution[2], solution[3]
        count = int(inliers.sum())
        if count < min_inliers or not solution[4]:
            return (float("inf"), float("inf"))
        return (float(np.median(distances[inliers])), -float(count))

    (
        R_current_reference,
        t_current_reference,
        distances,
        inliers,
        correspondence_geometry_valid,
    ) = min(solutions, key=solution_score)
    count = int(inliers.sum())
    ratio = float(count / max(current.shape[0], 1))
    residual = float(np.median(distances[inliers])) if count else float("inf")
    # Invert the estimated current->reference transform.
    R_reference_current = R_current_reference.T
    t_reference_current = -R_reference_current @ t_current_reference
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R_reference_current
    transform[:3, 3] = t_reference_current
    # The homogeneous translation is origin-dependent: rotating an object at
    # x=0.5 m about its own centre can produce a large matrix translation even
    # though that centre did not move.  Bound the displacement at the observed
    # object centre instead, which is the physical slip quantity callers need.
    corrected_reference_center = (
        R_reference_current @ reference_center + t_reference_current
    )
    translation_m = float(np.linalg.norm(corrected_reference_center - reference_center))
    rotation_rad = float(np.arccos(np.clip(
        (np.trace(R_reference_current) - 1.0) * 0.5, -1.0, 1.0
    )))
    reliable = (
        correspondence_geometry_valid
        and count >= min_inliers
        and ratio >= 0.20
        and residual <= 0.004
        and translation_m <= max_translation_m
        and rotation_rad <= max_rotation_rad
    )
    return RigidRegistration(transform, residual, ratio, translation_m,
                             rotation_rad, reliable)


def register_object_translation(
    reference_points_base: np.ndarray,
    current_points_base: np.ndarray,
    *,
    initial_translation_m: np.ndarray | None = None,
    max_points: int = 2048,
    max_correspondence_m: float = 0.012,
    min_inliers: int = 48,
    max_iterations: int = 8,
) -> TranslationRegistration:
    """Estimate object motion while tolerating eye-in-hand self-occlusion.

    Both clouds are already in the robot base frame using their own fresh
    camera poses. Translation-only robust ICP is intentional here: it runs at
    servo frequency, does not pretend a partial wrist view observes a stable
    6D object pose, and leaves grasp-orientation changes to the slower model.
    """
    reference = np.asarray(reference_points_base, dtype=np.float64).reshape(-1, 3)
    current = np.asarray(current_points_base, dtype=np.float64).reshape(-1, 3)
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    current = current[np.all(np.isfinite(current), axis=1)]
    if reference.shape[0] < min_inliers or current.shape[0] < min_inliers:
        return TranslationRegistration(np.zeros(3), float("inf"), 0.0, False)

    def sampled(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        return points[indices]

    reference = sampled(reference)
    current = sampled(current)
    tree = cKDTree(reference)
    object_translation = (
        np.zeros(3, dtype=np.float64)
        if initial_translation_m is None
        else np.asarray(initial_translation_m, dtype=np.float64).reshape(3).copy()
    )
    current_to_reference = -object_translation
    inliers = np.zeros(current.shape[0], dtype=bool)
    distances = np.full(current.shape[0], np.inf, dtype=np.float64)
    for _ in range(max_iterations):
        aligned = current + current_to_reference
        distances, indices = tree.query(aligned, k=1)
        inliers = distances <= max_correspondence_m
        if int(inliers.sum()) < min_inliers:
            break
        residuals = reference[indices[inliers]] - aligned[inliers]
        correction = np.median(residuals, axis=0)
        current_to_reference += correction
        if float(np.linalg.norm(correction)) < 1e-5:
            break

    count = int(inliers.sum())
    ratio = float(count / max(current.shape[0], 1))
    residual = float(np.median(distances[inliers])) if count else float("inf")
    reliable = count >= min_inliers and ratio >= 0.20 and residual <= 0.004
    return TranslationRegistration(-current_to_reference, residual, ratio, reliable)


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
