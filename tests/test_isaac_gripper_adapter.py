"""Pure calibration tests for the Isaac SO-101 gripper adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.adapters.isaac_gripper import (  # noqa: E402
    IsaacGripperController,
    LinearGripperCalibration,
)


class _LaggedArticulation:
    dof_names = ["gripper"]

    def __init__(self, *, contact_joint: float | None = None) -> None:
        self.position = 1.70
        self.target = self.position
        self.contact_joint = contact_joint
        self.max_effort = None

    def get_dof_positions(self):
        return np.asarray([self.position])

    def get_dof_efforts(self, **_kwargs):
        return np.asarray([0.20])

    def set_dof_position_targets(self, positions, **_kwargs) -> None:
        self.target = float(np.asarray(positions).reshape(-1)[0])

    def set_dof_max_efforts(self, max_efforts, **_kwargs) -> None:
        self.max_effort = float(np.asarray(max_efforts).reshape(-1)[0])

    def advance(self) -> None:
        next_position = self.position + float(np.clip(self.target - self.position, -0.006, 0.006))
        if self.contact_joint is not None:
            next_position = max(next_position, self.contact_joint)
        self.position = next_position


class IsaacGripperCalibrationTests(unittest.TestCase):
    def test_width_joint_roundtrip(self) -> None:
        calibration = LinearGripperCalibration()
        for width in (0.0, 0.012, 0.037, 0.075):
            self.assertAlmostEqual(
                calibration.width_for_joint(calibration.joint_for_width(width)), width, places=8
            )

    def test_force_maps_to_revolute_effort(self) -> None:
        calibration = LinearGripperCalibration(jaw_lever_arm_m=0.04)
        self.assertAlmostEqual(calibration.joint_effort_for_force(12.0), 0.48)

    def test_close_allows_force_limited_drive_to_settle_after_ramp(self) -> None:
        articulation = _LaggedArticulation()
        controller = IsaacGripperController(
            articulation,
            advance_frame=articulation.advance,
            physics_dt_s=1.0 / 60.0,
        )
        self.assertTrue(controller.close(0.0, force_n=12.0, speed_mps=0.025))
        self.assertLess(abs(articulation.position + 0.05), controller.joint_tolerance_rad)
        self.assertAlmostEqual(articulation.max_effort, 0.48)

    def test_close_accepts_sustained_contact_stall(self) -> None:
        articulation = _LaggedArticulation(contact_joint=0.80)
        controller = IsaacGripperController(
            articulation,
            advance_frame=articulation.advance,
            physics_dt_s=1.0 / 60.0,
        )
        self.assertTrue(controller.close(0.0, force_n=6.0, speed_mps=0.025))
        self.assertTrue(controller.state().contact)


if __name__ == "__main__":
    unittest.main()
