"""Bounded camera-view search, independent of grasp-pose generation."""
import numpy as np


def current_view_usable(sample, point):
    """Require an interior projection and matching measured depth, not just FOV."""
    if sample is None:
        return False
    p = np.linalg.inv(sample.T_world_camera) @ np.r_[point, 1.]
    if not .06 < p[2] < .55:
        return False
    uv = sample.intrinsics @ p[:3]
    u, v = uv[:2] / uv[2]
    h, w = sample.depth_m.shape
    if not (.12*w < u < .88*w and .12*h < v < .88*h):
        return False
    u, v = int(round(u)), int(round(v))
    depth = sample.depth_m[v-3:v+4, u-3:u+4]
    valid = np.isfinite(depth) & (depth > .01)
    return bool(valid.sum() >= 16 and abs(float(np.median(depth[valid])) - p[2]) < .025)


def observation_views(tool, camera, point):
    """40 optical viewpoints across heights, offsets, tilt and roll.

    These are observation poses, never geometric grasp proposals. The live
    collision-aware executor must validate each one before any movement.
    """
    hand_eye = np.linalg.inv(tool) @ camera
    poses = []
    for height, radius in ((.12, 0.), (.16, 0.), (.22, 0.), (.14, .08), (.20, .12)):
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            eye = np.asarray(point) + [radius*np.cos(angle), radius*np.sin(angle), height]
            z = np.asarray(point) - eye
            z /= np.linalg.norm(z)
            x = np.array([np.cos(angle), np.sin(angle), 0.])
            x -= np.dot(x, z)*z
            x /= np.linalg.norm(x)
            view = np.eye(4)
            view[:3, :3] = np.column_stack([x, np.cross(z, x), z])
            view[:3, 3] = eye
            poses.append(view @ np.linalg.inv(hand_eye))
    return sorted(poses, key=lambda pose: np.linalg.norm(pose[:3, 3]-tool[:3, 3])
                  + .03*np.linalg.norm(pose[:3, :3]-tool[:3, :3]))
