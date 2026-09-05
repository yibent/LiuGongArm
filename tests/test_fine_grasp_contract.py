"""BusAgent facade and object-policy contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.contracts import ObjectProperties, TargetSpec  # noqa: E402
from mr_liu.grasp.policy import policy_for_target  # noqa: E402
from mr_liu.config import fine_grasp_config  # noqa: E402
from mr_liu.grasp.settings import FineGraspSettings  # noqa: E402


class FineGraspContractTests(unittest.TestCase):
    def test_baseline_does_not_enable_experimental_multiview(self) -> None:
        settings = FineGraspSettings.from_mapping(fine_grasp_config())
        self.assertFalse(settings.fusion.enabled)
        self.assertFalse(settings.active_views.enabled)

    def test_empty_object_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetSpec("  ")

    def test_fruit_policy_limits_force_and_speed(self) -> None:
        policy = policy_for_target(
            TargetSpec("fruit-1", "苹果"),
            default_force_n=15.0,
            default_speed_mps=0.03,
            table_clearance_m=0.006,
        )
        self.assertLessEqual(policy.force_n, 6.0)
        self.assertLessEqual(policy.close_speed_mps, 0.01)

    def test_thin_and_reflective_properties_change_safety_policy(self) -> None:
        policy = policy_for_target(
            TargetSpec(
                "part-1",
                "unknown",
                properties=ObjectProperties(thin=True, reflective=True),
            ),
            default_force_n=12.0,
            default_speed_mps=0.02,
            table_clearance_m=0.006,
        )
        self.assertTrue(policy.require_thin_object_clearance)
        self.assertTrue(policy.allow_depth_hole_recovery)
        self.assertGreaterEqual(policy.required_table_clearance_m, 0.012)


if __name__ == "__main__":
    unittest.main()
