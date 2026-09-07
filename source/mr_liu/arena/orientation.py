"""Endpoint-constrained placement from observed geometry, without asset semantics."""
import numpy as np
from scipy.spatial.transform import Rotation


def align_vector(source, target):
    source = np.asarray(source, dtype=float); source = source / np.linalg.norm(source)
    target = np.asarray(target, dtype=float); target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross); cosine = np.clip(source @ target, -1., 1.)
    if sine > 1e-8:
        return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()
    if cosine > 0: return np.eye(3)
    perpendicular = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
    return Rotation.from_rotvec(perpendicular / np.linalg.norm(perpendicular) * np.pi).as_matrix()


def endpoint_vector(axis, endpoint, direction='up'):
    return np.asarray(axis) * (1 if endpoint == 1 else -1) * (1 if direction == 'up' else -1)


def placement_rotations(axis, tcp_rotation):
    """Keep endpoint up and search the unconstrained yaw for short wrist motion."""
    aligned = align_vector(axis, [0, 0, 1])
    proposals = [Rotation.from_euler('z', yaw).as_matrix() @ aligned for yaw in np.arange(0, 360, 30)]
    return sorted(proposals, key=lambda delta: Rotation.from_matrix(delta).magnitude())


def transformed_payload(points, tcp, delta):
    centre = np.quantile(points, [.02, .98], axis=0).mean(0)
    return (np.asarray(points)-centre) @ delta.T + centre


def placement_pose(points, tcp, delta, support, clearance=.003):
    """Rotate the grasp and its payload together; put measured bottom on support."""
    centre = np.quantile(points, [.02, .98], axis=0).mean(0)
    rotated = transformed_payload(points, tcp, delta)
    low, high = np.quantile(rotated, [.02, .98], axis=0)
    shift = np.r_[np.asarray(support)[:2]-(low[:2]+high[:2])/2, support[2]+clearance-low[2]]
    pose = np.asarray(tcp).copy()
    pose[:3,:3] = delta @ tcp[:3,:3]
    pose[:3,3] = delta @ (tcp[:3,3]-centre) + centre + shift
    return pose, rotated+shift


def endpoint_check(axis_object, quaternion_xyzw, endpoint, direction='up', tolerance_deg=12.):
    vector = Rotation.from_quat(quaternion_xyzw).apply(endpoint_vector(axis_object, endpoint, direction))
    angle = float(np.rad2deg(np.arccos(np.clip(vector[2]/np.linalg.norm(vector), -1., 1.))))
    return {'satisfied': angle <= tolerance_deg, 'angle_from_requested_deg': angle,
            'tolerance_deg': tolerance_deg, 'endpoint': endpoint, 'direction': direction}
