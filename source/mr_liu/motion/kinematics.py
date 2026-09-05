"""URDF kinematics for bounded SO-101 Cartesian jogging (no Isaac imports)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class ArmKinematics:
    def __init__(self, urdf, tool_frame="gripper_frame_link"):
        root = ET.parse(urdf).getroot()
        self.limits = {}
        by_child = {}
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            limit = joint.find("limit")
            if limit is not None:
                self.limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            by_child[joint.find("child").attrib["link"]] = joint
        chain = []
        child = tool_frame
        while child in by_child:
            joint = by_child[child]
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis = joint.find("axis")
            chain.append((joint.attrib["name"], transform, None if axis is None else np.fromstring(axis.attrib["xyz"], sep=" ")))
            child = joint.find("parent").attrib["link"]
        self.chain = list(reversed(chain))
        self.names = [name for name, _, axis in self.chain if axis is not None]

    def forward(self, joints, base=None):
        transform = np.eye(4) if base is None else base.copy()
        for name, origin, axis in self.chain:
            transform = transform @ origin
            if axis is not None:
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * joints[name]).as_matrix()
                transform = transform @ turn
        return transform

    def solve(self, position, orientation, joints, base):
        seed = np.array([joints[name] for name in self.names])
        lower, upper = np.array([self.limits[name] for name in self.names]).T

        def residual(q):
            pose = self.forward(dict(zip(self.names, q)), base)
            error = (pose[:3, 3] - position) * 10
            if orientation is not None:
                error = np.r_[error, Rotation.from_matrix(orientation @ pose[:3, :3].T).as_rotvec()]
            return error

        result = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7), bounds=(lower, upper), max_nfev=180)
        pose = self.forward(dict(zip(self.names, result.x)), base)
        position_error = float(np.linalg.norm(pose[:3, 3] - position))
        angle_error = 0.0 if orientation is None else float(Rotation.from_matrix(orientation @ pose[:3, :3].T).magnitude())
        if position_error > 0.004 or angle_error > np.deg2rad(3):
            raise ValueError(f"NO_IK：五轴机械臂无法到达该位姿（位置误差 {position_error*1000:.1f} mm，姿态误差 {np.rad2deg(angle_error):.1f} 度）；未执行运动")
        return {**joints, **dict(zip(self.names, result.x))}
