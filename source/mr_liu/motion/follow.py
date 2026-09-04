"""RMPflow follow-target controller for SO-101."""

from __future__ import annotations

import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.experimental.prims import Articulation, GeomPrim, XformPrim
from isaacsim.robot_motion.cumotion import CumotionRobot, RmpFlowController

from mr_liu.config import motion_config, robot_config
from mr_liu.motion.cumotion_robot import load_so101_cumotion_robot
from mr_liu.motion.world import bind_world
from mr_liu.paths import repo_root


def _estimated_state(articulation: Articulation) -> mg.RobotState:
    names = articulation.dof_names
    return mg.RobotState(
        joints=mg.JointState.from_name(
            robot_joint_space=names,
            positions=(names, articulation.get_dof_positions()),
            velocities=(names, articulation.get_dof_velocities()),
        )
    )


def _setpoint_state(cumotion_robot: CumotionRobot, target: GeomPrim, *, track_orientation: bool) -> mg.RobotState:
    tool_frame = robot_config()["tool_frame"]
    site_space = cumotion_robot.robot_description.tool_frame_names()
    target_positions, target_orientations = target.get_world_poses()
    kwargs = {
        "spatial_space": site_space,
        "positions": ([tool_frame], target_positions),
    }
    if track_orientation:
        kwargs["orientations"] = ([tool_frame], target_orientations)
    return mg.RobotState(sites=mg.SpatialState.from_name(**kwargs))


def _robot_root_poses(articulation: Articulation):
    """World pose of the USD robot xform (base), not the articulation-root joint."""
    robot_path = robot_config()["usd_prim_path"]
    try:
        return XformPrim(robot_path).get_world_poses()
    except Exception:
        return articulation.get_world_poses()


class FollowTargetController:
    def __init__(self, articulation: Articulation, target: GeomPrim) -> None:
        self._articulation = articulation
        self._target = target
        self._cumotion_robot = load_so101_cumotion_robot()
        self._world = bind_world(articulation, root_poses=_robot_root_poses(articulation))
        cfg = robot_config()
        motion = motion_config()
        self._track_orientation = bool(motion.get("track_orientation", False))
        self._debug = bool(motion.get("debug_follow", False))
        rmp_path = repo_root() / cfg["robot_dir"] / cfg["rmp_flow"]
        site_space = self._cumotion_robot.robot_description.tool_frame_names()
        self._controller = RmpFlowController(
            cumotion_robot=self._cumotion_robot,
            cumotion_world_interface=self._world.get_world_interface(),
            robot_joint_space=articulation.dof_names,
            robot_site_space=site_space,
            rmp_flow_configuration_filename=str(rmp_path),
            tool_frame=cfg["tool_frame"],
        )
        rmp_cfg = self._controller.get_rmp_flow_config()
        rmp_cfg.set_param("cspace_target_rmp/metric_scalar", float(motion["cspace_target_metric_scalar"]))
        rmp_cfg.set_param("collision_rmp/metric_scalar", float(motion["collision_metric_scalar"]))
        for key, value in (motion.get("rmp_overrides") or {}).items():
            rmp_cfg.set_param(str(key), float(value))
        print(f"[mr_liu] follow-target orientation tracking: {self._track_orientation}")

    def reset(self, t: float = 0.0) -> None:
        estimated = _estimated_state(self._articulation)
        setpoint = _setpoint_state(
            self._cumotion_robot, self._target, track_orientation=self._track_orientation
        )
        if not self._controller.reset(estimated, setpoint, t=t):
            raise RuntimeError("RmpFlowController.reset failed for SO-101.")

    def step(self, t: float) -> None:
        self._world.get_world_interface().update_world_to_robot_root_transforms(_robot_root_poses(self._articulation))
        self._world.synchronize_transforms()
        estimated = _estimated_state(self._articulation)
        setpoint = _setpoint_state(
            self._cumotion_robot, self._target, track_orientation=self._track_orientation
        )
        desired = self._controller.forward(estimated, setpoint, t)
        if desired is not None and desired.joints.positions is not None:
            self._articulation.set_dof_position_targets(
                positions=desired.joints.positions,
                dof_indices=desired.joints.position_indices,
            )
        if self._debug:
            pos, _ = self._target.get_world_poses()
            print(f"[mr_liu] target xyz={pos.numpy()[0]}")
