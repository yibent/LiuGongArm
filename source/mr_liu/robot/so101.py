"""SO-101 Articulation wrapper. No planning or CV."""

from __future__ import annotations

from isaacsim.core.experimental.prims import Articulation

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
