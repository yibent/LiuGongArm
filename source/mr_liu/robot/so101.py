"""SO-101 Articulation wrapper. No planning or CV."""

from __future__ import annotations

import numpy as np
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.experimental.utils.backend import use_backend

from mr_liu.config import robot_config
from mr_liu.robot.joints import assert_joint_mapping


class So101Arm:
    def __init__(self, prim_path: str | None = None) -> None:
        cfg = robot_config()
        candidates = [
            prim_path,
            cfg.get("articulation_prim_path"),
            cfg["usd_prim_path"],
        ]
        last_error: Exception | None = None
        self._articulation: Articulation | None = None
        self._prim_path = ""
        for path in candidates:
            if not path:
                continue
            try:
                self._articulation = Articulation(path)
                self._prim_path = path
                break
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                last_error = exc
        if self._articulation is None:
            raise RuntimeError(f"Could not wrap SO-101 articulation: {last_error}")

    @property
    def articulation(self) -> Articulation:
        assert self._articulation is not None
        return self._articulation

    @property
    def prim_path(self) -> str:
        return self._prim_path

    @property
    def dof_names(self) -> list[str]:
        return list(self.articulation.dof_names)

    def validate_joint_mapping(self) -> None:
        assert_joint_mapping(usd_dof_names=self.dof_names)

    def apply_configured_pose(self) -> None:
        """Move the arm to configs/robot_so101.yaml default_joint_positions."""
        defaults = robot_config()["default_joint_positions"]
        names = self.dof_names
        positions = [float(defaults.get(name, 0.0)) for name in names]
        self.articulation.set_dof_positions(positions)
        self.articulation.set_dof_position_targets(positions)
        print(f"[mr_liu] Applied default pose: {dict(zip(names, positions))}")

    def configure_drives(self) -> dict:
        """Call after physics warm-up; preserve stiffness and explicit force limits."""
        cfg = robot_config()
        values = cfg.get("drive_damping")
        if values:
            self.articulation.set_dof_gains(dampings=[[float(values[name]) for name in self.dof_names]])
        self.articulation.set_dof_velocity_targets([[0.] * len(self.dof_names)])
        stiffness, damping = self.articulation.get_dof_gains()
        info = {name: {"stiffness_nm_rad": float(k), "damping_nm_s_rad": float(d)}
                for name, k, d in zip(self.dof_names, stiffness.numpy().reshape(-1), damping.numpy().reshape(-1))}
        print(f"[mr_liu] Runtime joint gains: {info}")
        return info

    def T_base_ee(self) -> np.ndarray:
        """World/base pose of the configured URDF planning tool frame."""
        # Keep the basic arm/follow-target entry points independent of the
        # optional grasp package and its perception dependencies.
        from mr_liu.grasp.transforms import make_transform, quaternion_wxyz_to_matrix, rpy_to_matrix

        cfg = robot_config()
        parent = getattr(self, "_tool_parent", None)
        if parent is None:
            parent = XformPrim(str(cfg["tool_parent_prim_path"]), reset_xform_op_properties=False)
            self._tool_parent = parent
        # GPU/Fabric simulation does not write moving link poses back to USD.
        # Never use the authored startup transform for online grasp feedback.
        with use_backend("fabric", raise_on_fallback=True):
            positions, orientations = parent.get_world_poses()
        position = _first_numpy(positions)
        orientation = _first_numpy(orientations)
        T_base_parent = make_transform(quaternion_wxyz_to_matrix(orientation), position)
        T_parent_ee = make_transform(
            rpy_to_matrix(np.asarray(cfg["tool_from_parent_rpy"], dtype=np.float64)),
            np.asarray(cfg["tool_from_parent_translation"], dtype=np.float64),
        )
        return T_base_parent @ T_parent_ee


def _first_numpy(value) -> np.ndarray:
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    result = np.asarray(array, dtype=np.float64)
    return result[0] if result.ndim > 1 else result
