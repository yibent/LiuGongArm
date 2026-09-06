"""Local observed-surface checks for a calibrated asymmetric two-finger tool.

This complements global robot/world planning; it is not a contact simulator.
The conservative boxes must be calibrated to the physical fingertips. Unknown
space is reported explicitly and is not evidence of a collision-free volume.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from mr_liu.grasp.transforms import invert_transform, transform_points


@dataclass(frozen=True)
class FingerGeometry:
    thickness_m: float = 0.009
    breadth_m: float = 0.018
    length_m: float = 0.045
    open_width_m: float = 0.075
    margin_m: float = 0.001
    symmetric: bool = False

    def boxes(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        half = np.asarray([self.thickness_m, self.breadth_m, self.length_m]) * 0.5
        if self.symmetric:
            x = (self.open_width_m + self.thickness_m) * .5
            return tuple((np.asarray([sign*x, 0., -self.length_m/2]), half) for sign in (-1, 1))
        # EE lies at the fixed finger centreline at the tip, +Z into object.
        return ((np.asarray([0.0, 0.0, -self.length_m / 2]), half),
                (np.asarray([-self.open_width_m - self.thickness_m, 0.0, -self.length_m / 2]), half))

    def collision_count(self, points_base: np.ndarray, pose: np.ndarray) -> int:
        local = transform_points(invert_transform(pose), points_base)
        occupied = np.zeros(len(local), dtype=bool)
        for center, half in self.boxes():
            occupied |= np.all(np.abs(local - center) < half + self.margin_m, axis=1)
        return int(occupied.sum())

    def path_collision_count(self, points_base: np.ndarray, start: np.ndarray,
                             end: np.ndarray) -> int:
        from scipy.spatial.transform import Rotation, Slerp
        length = np.linalg.norm(end[:3, 3] - start[:3, 3])
        rotations = Rotation.from_matrix(np.asarray([start[:3, :3], end[:3, :3]]))
        angle = (rotations[1] * rotations[0].inv()).magnitude()
        count = max(2, int(np.ceil(length / 0.002)), int(np.ceil(angle / np.deg2rad(2))))
        slerp = Slerp([0, 1], rotations)
        maximum = 0
        for fraction in np.linspace(0, 1, count + 1):
            pose = np.eye(4)
            pose[:3, :3] = slerp(fraction).as_matrix()
            pose[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * end[:3, 3]
            maximum = max(maximum, self.collision_count(points_base, pose))
        return maximum


def configured_finger_geometry(*, open_width_m=None) -> FingerGeometry:
    from mr_liu.config import robot_config
    values = dict(robot_config().get("finger_geometry", {}))
    if open_width_m is not None:
        values["open_width_m"] = float(open_width_m)
    return FingerGeometry(**values)
