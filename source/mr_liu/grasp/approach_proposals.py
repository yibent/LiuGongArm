"""Bounded inclined proposal pool; every pose requires neural rescoring and IK.

Uses only the segmented cloud and calibrated robot/support frames, never object
ground truth. Coordinates are the same world frame used by live FineGrasp.
"""
import numpy as np


def inclined_proposals(points, robot_position, table_height, fingertip_depth):
    points = np.asarray(points, dtype=float)
    robot_position = np.asarray(robot_position, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid proposal cloud")
    if robot_position.shape != (3,) or not np.isfinite(robot_position).all():
        raise ValueError("Invalid robot frame")
    center = np.median(points, axis=0)
    top = float(np.percentile(points[:, 2], 95))
    if not table_height < top < table_height + .15:
        return np.empty((0, 4, 4), dtype=np.float32)
    center[2] = (top + table_height) * .5
    radial = center[:2] - robot_position[:2]
    if np.linalg.norm(radial) < .05:
        return np.empty((0, 4, 4), dtype=np.float32)
    azimuth = np.arctan2(radial[1], radial[0])
    poses = []
    for tilt in (25, 35, 45, 55, 65):
        for offset in (-9, -6, -3, 0, 3, 6, 9):
            yaw = azimuth + np.deg2rad(offset)
            radial = np.array([np.cos(yaw), np.sin(yaw), 0.])
            z = radial * np.sin(np.deg2rad(tilt))
            z[2] = -np.cos(np.deg2rad(tilt))
            for sign in (-1, 1):
                x = sign * np.array([-radial[1], radial[0], 0.])
                for dz in (-.004, -.002, 0.):
                    pose = np.eye(4)
                    pose[:3, :3] = np.column_stack([x, np.cross(z, x), z])
                    pose[:3, 3] = center + [0, 0, dz] - z * fingertip_depth
                    poses.append(pose)
    return np.asarray(poses, dtype=np.float32)
