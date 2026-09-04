from __future__ import annotations

from isaacsim.robot_motion.cumotion import CumotionRobot, load_cumotion_robot

from mr_liu.config import robot_config
from mr_liu.paths import repo_root
from mr_liu.robot.joints import assert_joint_mapping


def load_so101_cumotion_robot() -> CumotionRobot:
    from isaacsim.core.experimental.utils.app import enable_extension

    enable_extension("isaacsim.robot_motion.cumotion")
    cfg = robot_config()
    directory = repo_root() / cfg["robot_dir"]
    robot = load_cumotion_robot(
        directory=directory,
        urdf_filename=cfg["urdf"],
        xrdf_filename=cfg["xrdf"],
    )
    assert_joint_mapping()
    expected = list(cfg["controlled_joints"])
    if robot.controlled_joint_names != expected:
        raise ValueError(
            f"cuMotion controlled joints {robot.controlled_joint_names} != config {expected}"
        )
    tool_frames = robot.robot_description.tool_frame_names()
    if cfg["tool_frame"] not in tool_frames:
        raise ValueError(f"tool_frame {cfg['tool_frame']!r} not in {tool_frames}")
    return robot
