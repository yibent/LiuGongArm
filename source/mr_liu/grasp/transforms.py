"""Rigid transform utilities with explicit ``T_destination_source`` naming."""

from __future__ import annotations

import math

import numpy as np


def assert_transform(T: np.ndarray, *, name: str = "transform", atol: float = 1e-5) -> np.ndarray:
    value = np.asarray(T, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError(f"{name} has an invalid homogeneous row")
    if not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=atol):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(value[:3, :3]) < 0.0:
        raise ValueError(f"{name} rotation is left-handed")
    return value


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return assert_transform(T)


def invert_transform(T_destination_source: np.ndarray) -> np.ndarray:
    T = assert_transform(T_destination_source)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def compose(*transforms: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ assert_transform(transform)
    return assert_transform(result)


def transform_points(T_destination_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    T = assert_transform(T_destination_source)
    points = np.asarray(points_source, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    return points @ T[:3, :3].T + T[:3, 3]


def quaternion_wxyz_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw rotation matrix (Rz @ Ry @ Rx)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rotation_angle_rad(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> tuple[np.ndarray, float]:
    current = assert_transform(T_current, name="T_current")
    target = assert_transform(T_target, name="T_target")
    return target[:3, 3] - current[:3, 3], rotation_angle_rad(current[:3, :3], target[:3, :3])


def interpolate_pose_step(
    T_current: np.ndarray,
    T_target: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> np.ndarray:
    """Bounded pose step using axis-angle interpolation for servo commands."""
    current = assert_transform(T_current, name="T_current")
    target = assert_transform(T_target, name="T_target")
    delta = target[:3, 3] - current[:3, 3]
    distance = float(np.linalg.norm(delta))
    translation_alpha = 1.0 if distance <= max_translation_m else max_translation_m / distance

    relative = current[:3, :3].T @ target[:3, :3]
    angle = rotation_angle_rad(np.eye(3), relative)
    rotation_alpha = 1.0 if angle <= max_rotation_rad or angle < 1e-9 else max_rotation_rad / angle
    if angle < 1e-9:
        rotation_step = np.eye(3)
    else:
        skew = (relative - relative.T) / (2.0 * math.sin(angle))
        axis = np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]])
        theta = angle * rotation_alpha
        K = np.asarray(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        rotation_step = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

    return make_transform(
        current[:3, :3] @ rotation_step,
        current[:3, 3] + translation_alpha * delta,
    )
