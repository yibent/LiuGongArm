"""Phase 1: URDF / XRDF / config joint names must stay aligned."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.robot.joints import (  # noqa: E402
    assert_joint_mapping,
    expected_usd_joints,
    parse_urdf_joints,
    parse_xrdf_cspace,
    robot_dir,
)


class JointMappingTests(unittest.TestCase):
    def test_urdf_matches_config(self) -> None:
        self.assertEqual(parse_urdf_joints(), expected_usd_joints())

    def test_xrdf_cspace_is_arm_joints(self) -> None:
        cspace = parse_xrdf_cspace()
        self.assertEqual(
            cspace,
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        )
        self.assertTrue(set(cspace).issubset(set(expected_usd_joints())))

    def test_mapping_helper_passes(self) -> None:
        assert_joint_mapping()

    def test_robot_files_exist(self) -> None:
        directory = robot_dir()
        self.assertTrue((directory / "robot.urdf").is_file())
        self.assertTrue((directory / "robot.xrdf").is_file())
        self.assertTrue((directory / "rmp_flow.yaml").is_file())

    def test_urdf_has_tool_frame(self) -> None:
        text = (robot_dir() / "robot.urdf").read_text(encoding="utf-8")
        self.assertIn('link name="gripper_frame_link"', text)
        self.assertIn('joint name="gripper_frame_joint"', text)


if __name__ == "__main__":
    unittest.main()
