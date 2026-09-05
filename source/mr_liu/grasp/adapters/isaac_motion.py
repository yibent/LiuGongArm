"""Collision-aware cuMotion executor for the model-independent grasp node."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import cumotion
import numpy as np
from scipy.optimize import least_squares
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.robot_motion.cumotion import GraphBasedMotionPlanner
from isaacsim.robot_motion.cumotion.impl.utils import isaac_sim_to_cumotion_translation

from mr_liu.config import robot_config
from mr_liu.grasp.contracts import GripperState, RobotState, SelectedGrasp
from mr_liu.grasp.transforms import (
    axis_alignment_error,
    invert_transform,
    make_transform,
    matrix_to_quaternion_wxyz,
    pose_error,
    quaternion_wxyz_to_matrix,
)
from mr_liu.motion.follow import FollowTargetController
from mr_liu.robot.so101 import So101Arm


@dataclass(frozen=True)
class _JointStepPlan:
    q_target: np.ndarray


@dataclass(frozen=True)
class _JointWaypointPath:
    """Small path façade compatible with Isaac's cuMotion Path API."""

    waypoints: tuple[np.ndarray, ...]

    def get_waypoints_count(self) -> int:
        return len(self.waypoints)

    def get_waypoint_by_index(self, index: int) -> np.ndarray:
        return self.waypoints[index]


class IsaacCumotionExecutor:
    """Use cuMotion graph planning for coarse moves and RMPflow for servo steps.

    The current SO-101 has five arm DOFs, so arbitrary 6D poses are not
    generally realizable. Planning and convergence therefore constrain XYZ and
    leave roll/pitch redundancy to cuMotion; the requested orientation remains
    in the stable contract for higher-DOF robot adapters.
    """

    def __init__(
        self,
        arm: So101Arm,
        *,
        advance_frame: Callable[[], None],
        physics_dt_s: float,
        target_object_paths: list[str] | None = None,
        obstacle_margin_overrides: dict[str, float] | None = None,
        target_prim_path: str = "/World/FineGraspMotionTarget",
        position_tolerance_m: float = 0.002,
        joint_tolerance_rad: float = 0.020,
        track_orientation: bool = False,
        max_move_steps: int = 480,
        max_servo_steps: int = 240,
        waypoint_settle_steps: int = 180,
        planner_max_iterations: int = 300,
        planner_max_sampling: int = 300,
        pinch_center_x_per_width: float = -0.5,
        fixed_finger_center_offset_m: float = 0.0045,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.arm = arm
        self.articulation = arm.articulation
        self.execution_guard = None
        self._guard_ticks = 0
        def guarded_advance():
            advance_frame()
            self._guard_ticks += 1
            # Optional coarse-loop guard only; FineGrasp keeps its original
            # observation-at-servo-boundary behavior when this callback is None.
            if self.execution_guard is not None and self._guard_ticks % 6 == 0:
                if not self.execution_guard():
                    self.stop()
        self.advance_frame = guarded_advance
        self.physics_dt_s = float(physics_dt_s)
        self.position_tolerance_m = float(position_tolerance_m)
        self.joint_tolerance_rad = float(joint_tolerance_rad)
        # Full orientation is useful for placing the open gripper at pregrasp.
        # The five arm joints cannot span arbitrary SE(3), so the outer visual
        # servo requests only bounded poses that remain on the selected grasp's
        # reachable manifold; each one is checked here with cuMotion.
        self.plan_orientation = bool(track_orientation)
        self.tracks_orientation = self.plan_orientation
        self.orientation_mode = "approach_axis" if self.plan_orientation else "none"
        self.max_move_steps = int(max_move_steps)
        self.max_servo_steps = int(max_servo_steps)
        self.waypoint_settle_steps = int(waypoint_settle_steps)
        self.clock = clock
        self.pinch_center_x_per_width = float(pinch_center_x_per_width)
        # ``gripper_frame_link`` lies on the fixed finger centreline, not its
        # inner contact face.  Half the 9 mm finger thickness must remain
        # outside the target silhouette during a top-down insertion.
        self.fixed_finger_center_offset_m = float(fixed_finger_center_offset_m)
        self._stopped = False
        self._time_s = 0.0
        # Candidate filtering asks about both pregrasp and contact poses.  Keep
        # every answer while the robot is stationary so move_to can reuse the
        # exact path that passed collision/IK checks.  A single-entry cache made
        # execution rerun a stochastic JtRRT query and could turn a 5 s check
        # into an effectively unbounded search for a 5-DOF boundary solution.
        self._plan_cache: dict[tuple[float, ...], object | None] = {}
        # IK solutions remain useful after a path is executed: near a wrist
        # singularity, warm-starting the next 6 mm solve only from the current
        # state can stay on a branch that ends above the object.  A previously
        # validated contact solution provides the alternative branch seed.
        self._ik_seed_cache: dict[tuple[float, ...], np.ndarray] = {}
        self._active_grasp_q: np.ndarray | None = None
        self._active_grasp_pose: np.ndarray | None = None
        self._transition_cache: dict[
            tuple[tuple[float, ...], tuple[float, ...]], list[np.ndarray] | None
        ] = {}
        self._grasp_center_specs: dict[
            tuple[float, ...], tuple[np.ndarray, float]
        ] = {}
        self.last_transition_rejection_reason = "approach_disconnected"
        self._active_waypoints: list[np.ndarray] = []
        self._active_waypoint_index = 0
        self._fk_reported = False
        self.last_plan_diagnostic = {}
        self.last_collision_reason = None

        stage = stage_utils.get_current_stage()
        if not stage.GetPrimAtPath(target_prim_path).IsValid():
            stage.DefinePrim(target_prim_path, "Xform")
        self.target = XformPrim(target_prim_path, reset_xform_op_properties=True)
        self.controller = FollowTargetController(
            self.articulation,
            self.target,
            track_orientation=self.plan_orientation,
            extra_exclude_prim_paths=[target_prim_path, *(target_object_paths or [])],
            **({"obstacle_margin_overrides":obstacle_margin_overrides} if obstacle_margin_overrides else {}),
        )
        self.planner = GraphBasedMotionPlanner(
            self.controller.cumotion_robot,
            self.controller.world_binding.get_world_interface(),
            tool_frame=str(robot_config()["tool_frame"]),
        )
        planner_config = self.planner.get_graph_planner_config()
        planner_config.set_param(
            "max_iterations",
            cumotion.MotionPlannerConfig.ParamValue(int(planner_max_iterations)),
        )
        planner_config.set_param(
            "max_sampling",
            cumotion.MotionPlannerConfig.ParamValue(int(planner_max_sampling)),
        )
        world_interface = self.controller.world_binding.get_world_interface()
        self._collision_inspector = cumotion.create_robot_world_inspector(
            self.controller.cumotion_robot.robot_description,
            world_interface.world_view,
        )
        names = list(self.articulation.dof_names)
        self.arm_dof_indices = [names.index(name) for name in self.controller.cumotion_robot.controlled_joint_names]

    def robot_state(self) -> RobotState:
        positions = self.articulation.get_dof_positions()
        if hasattr(positions, "numpy"):
            positions = positions.numpy()
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        return RobotState(
            timestamp_s=self.clock(),
            T_base_ee=self.arm.T_base_ee(),
            joint_positions={name: float(values[index]) for index, name in enumerate(self.articulation.dof_names)},
        )

    def kinematic_diagnostics(self) -> dict[str, object]:
        """Compare cuMotion FK with the live USD tool pose at the current q."""
        q = self._arm_values(self.articulation.get_dof_positions)
        kinematics = self.controller.cumotion_robot.kinematics
        tool_frame = str(robot_config()["tool_frame"])
        predicted = np.asarray(kinematics.position(q, tool_frame), dtype=np.float64)
        base_position, base_orientation = (
            self.controller.world_binding.get_world_interface().get_world_to_robot_base_transform()
        )
        actual = isaac_sim_to_cumotion_translation(
            self.arm.T_base_ee()[:3, 3], base_position, base_orientation
        )
        result: dict[str, object] = {
            "q": q.tolist(),
            "predicted_position_robot_m": predicted.tolist(),
            "actual_position_robot_m": np.asarray(actual).tolist(),
            "translation_error_m": float(np.linalg.norm(predicted - actual)),
        }
        if self._active_grasp_q is not None:
            result["active_grasp_q"] = self._active_grasp_q.tolist()
            result["active_grasp_position_robot_m"] = np.asarray(
                kinematics.position(self._active_grasp_q, tool_frame), dtype=np.float64
            ).tolist()
        return result

    def ee_pose_for_grasp(self, T_base_grasp_center: np.ndarray, width_m: float) -> np.ndarray:
        """Map a canonical pinch-centre pose to SO-101's fixed-finger EE frame.

        ``pinch_center_x_per_width=-0.5`` means the physical pinch centre is
        half the object width along -X from ``gripper_frame_link``.  Inverting
        that calibrated relation offsets the commanded EE along +X.
        """
        center = np.asarray(T_base_grasp_center, dtype=np.float64)
        T_ee_center = np.eye(4, dtype=np.float64)
        T_ee_center[0, 3] = (
            self.pinch_center_x_per_width * float(width_m)
            - self.fixed_finger_center_offset_m
        )
        T_base_ee = center @ invert_transform(T_ee_center)
        self._grasp_center_specs[self._pose_key(T_base_ee)] = (
            center.copy(),
            float(width_m),
        )
        if len(self._grasp_center_specs) > 256:
            self._grasp_center_specs.pop(next(iter(self._grasp_center_specs)))
        return T_base_ee

    def prepare_grasp(self, grasp: SelectedGrasp) -> bool:
        """Pin the candidate's validated endpoint for closed-loop execution."""
        pair = (self._pose_key(grasp.T_base_pregrasp), self._pose_key(grasp.T_base_grasp))
        if pair not in self._transition_cache and not self.is_grasp_transition_reachable(
            grasp.T_base_pregrasp, grasp.T_base_grasp
        ):
            return False
        waypoints = self._transition_cache.get(pair)
        if not waypoints:
            return False
        self._active_waypoints = [waypoint.copy() for waypoint in waypoints]
        self._active_waypoint_index = 1
        self._active_grasp_q = self._active_waypoints[-1].copy()
        self._active_grasp_pose = np.asarray(grasp.T_base_grasp, dtype=np.float64).copy()
        return True

    def realized_grasp_pose(self, T_base_ee: np.ndarray) -> np.ndarray | None:
        """Return the actual FK pose of the cached five-axis IK solution."""
        q_goal = self._ik_seed_cache.get(self._pose_key(T_base_ee))
        if q_goal is None:
            return None
        return self._world_ee_pose_from_joints(q_goal)

    def _world_ee_pose_from_joints(self, q_goal):
        kinematics = self.controller.cumotion_robot.kinematics
        tool_frame = str(robot_config()["tool_frame"])
        goal = kinematics.pose(q_goal, tool_frame)
        position_robot = np.asarray(goal.translation, dtype=np.float64)
        rotation_robot = np.asarray(goal.rotation.matrix(), dtype=np.float64)
        base_position, base_orientation = (
            self.controller.world_binding.get_world_interface().get_world_to_robot_base_transform()
        )
        base_position_np = np.asarray(
            base_position.numpy() if hasattr(base_position, "numpy") else base_position,
            dtype=np.float64,
        ).reshape(-1, 3)[0]
        base_quaternion = np.asarray(
            base_orientation.numpy() if hasattr(base_orientation, "numpy") else base_orientation,
            dtype=np.float64,
        ).reshape(-1, 4)[0]
        R_world_robot = quaternion_wxyz_to_matrix(base_quaternion)
        return make_transform(
            R_world_robot @ rotation_robot,
            base_position_np + R_world_robot @ position_robot,
        )

    def clear_grasp_plan(self):
        """Invalidate insertion/escape paths after an attempt, not calibration."""
        self._active_waypoints = []
        self._active_waypoint_index = 0
        self._active_grasp_q = None
        self._active_grasp_pose = None
        self._plan_cache.clear()
        self._transition_cache.clear()

    def is_observation_path_safe(self, target, points_base, finger):
        """Check fingers along the actual joint path in addition to world IK.

        This is an observed-surface test with the configured finger-box model,
        not a guarantee about unobserved space or unmodeled hardware.
        """
        self.last_observation_path_rejection = None
        plan = self._plan(target)
        if plan is None:
            self.last_observation_path_rejection = "ik_or_joint_path_infeasible"
            return False
        q0 = self._arm_values(self.articulation.get_dof_positions)
        if isinstance(plan, _JointStepPlan):
            waypoints = [q0, plan.q_target]
        elif hasattr(plan, "get_waypoints_count"):
            waypoints = [q0] + [np.asarray(plan.get_waypoint_by_index(i)).reshape(-1)
                                for i in range(plan.get_waypoints_count())]
        else:
            self.last_observation_path_rejection = "unsupported_plan_type"
            return False
        previous = self._world_ee_pose_from_joints(q0)
        for start, end in zip(waypoints, waypoints[1:]):
            steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / 0.005)))
            for alpha in np.linspace(0, 1, steps + 1):
                q = start + alpha * (end - start)
                if self._collision_inspector.in_self_collision(q):
                    self.last_observation_path_rejection = "self_collision"
                    return False
                if self._collision_inspector.in_collision_with_obstacle(q):
                    self.last_observation_path_rejection = "world_collision"
                    return False
                pose = self._world_ee_pose_from_joints(q)
                if finger.path_collision_count(points_base, previous, pose):
                    self.last_observation_path_rejection = "observed_surface_finger_collision"
                    return False
                previous = pose
        return True

    def is_grasp_transition_reachable(
        self, T_base_pregrasp: np.ndarray, T_base_grasp: np.ndarray
    ) -> bool:
        """Require one continuous collision-free branch through the approach."""
        self.last_transition_rejection_reason = "approach_disconnected"
        pair = (self._pose_key(T_base_pregrasp), self._pose_key(T_base_grasp))
        if pair in self._transition_cache:
            return self._transition_cache[pair] is not None
        if self._plan(T_base_pregrasp) is None or self._plan(T_base_grasp) is None:
            self._transition_cache[pair] = None
            return False
        q_pregrasp = self._ik_seed_cache.get(pair[0])
        q_grasp = self._ik_seed_cache.get(pair[1])
        if q_pregrasp is None or q_grasp is None:
            self._transition_cache[pair] = None
            return False
        center_spec = self._grasp_center_specs.get(pair[1])
        if center_spec is not None and self.orientation_mode == "approach_axis":
            desired_center_world, width_m = center_spec
            kinematics = self.controller.cumotion_robot.kinematics
            tool_frame = str(robot_config()["tool_frame"])
            goal_pose = kinematics.pose(q_grasp, tool_frame)
            goal_rotation = np.asarray(goal_pose.rotation.matrix(), dtype=np.float64)
            actual_center_base = np.asarray(goal_pose.translation, dtype=np.float64) + (
                goal_rotation[:, 0]
                * (
                    self.pinch_center_x_per_width * width_m
                    - self.fixed_finger_center_offset_m
                )
            )
            base_position, base_orientation = (
                self.controller.world_binding.get_world_interface().get_world_to_robot_base_transform()
            )
            desired_center_base = isaac_sim_to_cumotion_translation(
                desired_center_world[:3, 3], base_position, base_orientation
            )
            center_error_m = float(np.linalg.norm(actual_center_base - desired_center_base))
            # XYZ plus approach-axis IK leaves closing-axis yaw unconstrained on
            # a five-DOF arm.  Reject poses whose reachable yaw would move the
            # fixed-finger gripper's physical pinch centre off the object.
            if center_error_m > 0.006:
                self.last_transition_rejection_reason = "pinch_center_mismatch"
                self._transition_cache[pair] = None
                return False
        path = self._linear_cspace_path(q_pregrasp, q_grasp)
        if path is None:
            self._transition_cache[pair] = None
            return False
        waypoints = [
            np.asarray(path.get_waypoint_by_index(index), dtype=np.float64).reshape(-1)
            for index in range(path.get_waypoints_count())
        ]
        self._transition_cache[pair] = waypoints
        return True

    def is_reachable(self, T_base_ee: np.ndarray) -> bool:
        return self._plan(T_base_ee) is not None

    def initial_pose_proposal(self):
        """Check the configured arm home, retaining the current gripper opening."""
        self.controller.synchronize_world()
        q = self._arm_values(self.articulation.get_dof_positions)
        names = self.controller.cumotion_robot.controlled_joint_names
        defaults = robot_config()["default_joint_positions"]
        goal = np.array([defaults[name] for name in names], dtype=float)
        if np.max(np.abs(q-goal)) < .05:
            return None  # Already at home: do not prescribe a pointless reset.
        if self._linear_cspace_path(q, goal) is None:
            return None
        return dict(name="初始准备姿态", joints_deg=dict(zip(names, np.rad2deg(goal).tolist())),
                    gripper="保持当前开度", path_checked=True, auto_grasp=False)

    def move_to_initial_pose(self):
        """Recheck from actual joints after confirmation, never reuse an old path."""
        self._stopped = False
        self.controller.synchronize_world()
        q = self._arm_values(self.articulation.get_dof_positions)
        names = self.controller.cumotion_robot.controlled_joint_names
        goal = np.array([robot_config()["default_joint_positions"][n] for n in names])
        path = self._linear_cspace_path(q, goal)
        return path is not None and self._execute_plan(path, speed_scale=.6, max_steps=self.max_move_steps)

    def is_collision_free(
        self,
        T_base_ee: np.ndarray,
        *,
        margin_m: float,
        allow_target_contact: bool = False,
    ) -> bool:
        # The configured cuMotion world owns its safety tolerance. Target-object
        # contact is allowed by excluding only the selected object at binding.
        del margin_m, allow_target_contact
        return self._plan(T_base_ee) is not None

    def move_to(self, T_base_ee: np.ndarray, *, speed_scale: float) -> bool:
        self._stopped = False
        self.controller.set_track_orientation(self.plan_orientation)
        path = self._plan(T_base_ee)
        if path is None:
            return False
        # Graph waypoints guarantee collision-free global transit. Ramp each
        # one through the articulation drive rather than teleporting joints.
        if not self._execute_plan(path, speed_scale=speed_scale, max_steps=self.max_move_steps):
            return False
        self._plan_cache.clear()
        return self._target_reached(T_base_ee)

    def servo_to(self, T_base_ee: np.ndarray, *, speed_scale: float) -> bool:
        self._stopped = False
        self.controller.set_track_orientation(self.plan_orientation)
        # Resolve and execute only this (normally <= 6 mm) increment.  Returning
        # after the bounded segment is essential: GeneralGrasp captures a fresh
        # moving wrist frame before it is allowed to request the next segment.
        path = self._plan(T_base_ee)
        if path is None:
            return False
        follows_validated_branch = isinstance(path, _JointStepPlan)
        if not self._execute_plan(path, speed_scale=speed_scale, max_steps=self.max_servo_steps):
            return False
        self._plan_cache.clear()
        # A curved five-axis branch need not land on the virtual Cartesian
        # subtarget. The outer loop will measure the real error immediately;
        # success here means the bounded, prevalidated joint segment executed.
        return follows_validated_branch or self._target_reached(T_base_ee)

    def lift_from_grasp(self, T_base_ee: np.ndarray, *, speed_scale: float) -> bool:
        """Retrace the validated insertion branch before extending the lift.

        A grasped object changes the local collision/dynamics state and compact
        five-axis arms are often near a singularity at contact.  The reverse of
        the already checked pregrasp-to-grasp path is the safest deterministic
        escape.  Only the remaining lift distance is handed back to IK.
        """
        self._stopped = False
        self.controller.set_track_orientation(self.plan_orientation)
        if self._active_waypoints:
            q_current = self._arm_values(self.articulation.get_dof_positions)
            q_pregrasp = self._active_waypoints[0]
            reverse_path = self._linear_cspace_path(q_current, q_pregrasp)
            if reverse_path is None or not self._execute_plan(
                reverse_path,
                speed_scale=speed_scale,
                max_steps=self.max_move_steps,
            ):
                return False
            self._plan_cache.clear()
        if self._target_reached(T_base_ee):
            return True
        path = self._plan(T_base_ee)
        if path is None:
            return False
        completed = self._execute_plan(
            path, speed_scale=speed_scale, max_steps=self.max_move_steps
        )
        self._plan_cache.clear()
        return completed and self._target_reached(T_base_ee)

    def stop(self) -> None:
        self._stopped = True
        # Stop the arm, not the independent force-limited gripper. Replacing a
        # stalled jaw's commanded closing target by its measured position drops
        # the grip preload and can release the object during the next observation.
        # Gripper cancellation belongs to the gripper controller, not arm motion.
        positions = self._arm_values(self.articulation.get_dof_positions)
        self.articulation.set_dof_position_targets(positions,dof_indices=self.arm_dof_indices)

    def _plan(self, T_base_ee: np.ndarray):
        key = self._pose_key(T_base_ee)
        if key in self._plan_cache:
            return self._plan_cache[key]
        self.controller.synchronize_world()
        current = self.articulation.get_dof_positions()
        if hasattr(current, "numpy"):
            current = current.numpy()
        q = np.asarray(current, dtype=np.float64).reshape(-1)[self.arm_dof_indices]
        target = np.asarray(T_base_ee, dtype=np.float64)
        local_distance = float(np.linalg.norm(target[:3, 3] - self.arm.T_base_ee()[:3, 3]))
        if self._active_grasp_q is not None and local_distance <= 0.008:
            path = self._plan_active_grasp_step(q, target)
        elif self.orientation_mode == "approach_axis":
            path = self._plan_axis_constrained(q, target, key)
        elif self.plan_orientation:
            path = self.planner.plan_to_pose_target(
                q, target[:3, 3], matrix_to_quaternion_wxyz(target[:3, :3])
            )
        else:
            path = self.planner.plan_to_translation_target(q, target[:3, 3])
        self._plan_cache[key] = path
        return path

    @staticmethod
    def _pose_key(T_base_ee: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(T_base_ee, dtype=np.float64).reshape(-1), 5))

    def _plan_active_grasp_step(self, q: np.ndarray, command_target: np.ndarray):
        """Take one collision-checked step along the selected reachable branch."""
        assert self._active_grasp_q is not None
        if float(np.max(np.abs(self._active_grasp_q - q))) < 0.02:
            # The original model pose has been reached, but the latest wrist
            # observation can legitimately shift the target by a few mm. Extend
            # the validated branch with one fresh local IK/collision solve rather
            # than replaying the stale endpoint or forcing a full model replan.
            target_key = self._pose_key(command_target)
            correction = self._plan_axis_constrained(q, command_target, target_key)
            if correction is not None:
                self._active_grasp_q = np.asarray(
                    correction.get_waypoint_by_index(
                        correction.get_waypoints_count() - 1
                    ),
                    dtype=np.float64,
                ).reshape(-1)
            return correction
        # is_grasp_transition_reachable validated the entire straight c-space
        # segment densely.  Progress toward its endpoint directly; advancing a
        # sampled waypoint by joint-error threshold deadlocks near a singularity
        # where harmless servo sag can exceed that threshold indefinitely.
        joint_delta = self._active_grasp_q - q
        if float(np.max(np.abs(joint_delta))) < 1e-4:
            return _JointStepPlan(self._active_grasp_q.copy())
        # Keep each five-axis branch command small enough that the physical
        # joint drives can demonstrate progress before gravity sag consumes
        # the entire Cartesian step.  The outer loop deliberately trades a few
        # more camera frames for a fresher and more stable eye-in-hand path.
        alpha = min(1.0, 0.05 / max(float(np.max(np.abs(joint_delta))), 1e-9))
        kinematics = self.controller.cumotion_robot.kinematics
        tool_frame = str(robot_config()["tool_frame"])
        current_position_base = np.asarray(
            kinematics.position(q, tool_frame), dtype=np.float64
        )
        current_world = self.arm.T_base_ee()[:3, 3]
        requested_world_delta = command_target[:3, 3] - current_world
        requested_distance = float(np.linalg.norm(requested_world_delta))
        base_position, base_orientation = (
            self.controller.world_binding.get_world_interface().get_world_to_robot_base_transform()
        )
        base_quaternion = np.asarray(
            base_orientation.numpy() if hasattr(base_orientation, "numpy") else base_orientation,
            dtype=np.float64,
        ).reshape(-1, 4)[0]
        R_world_base = quaternion_wxyz_to_matrix(base_quaternion)
        requested_base_delta = R_world_base.T @ requested_world_delta

        q_step = q.copy()
        actual_base_delta = np.zeros(3, dtype=np.float64)
        for _ in range(10):
            q_step = q + alpha * joint_delta
            step_position = np.asarray(
                kinematics.position(q_step, tool_frame), dtype=np.float64
            )
            actual_base_delta = step_position - current_position_base
            if float(np.linalg.norm(actual_base_delta)) <= max(requested_distance * 1.2, 0.0015):
                break
            alpha *= 0.5
        actual_distance = float(np.linalg.norm(actual_base_delta))
        if actual_distance < 1e-5:
            actual_position_base = isaac_sim_to_cumotion_translation(
                current_world, base_position, base_orientation
            )
            goal_position_base = np.asarray(
                kinematics.position(self._active_grasp_q, tool_frame), dtype=np.float64
            )
            print(
                "[mr_liu] grasp branch exhausted before visual alignment: "
                f"fk_usd_error_m={np.linalg.norm(current_position_base - actual_position_base):.6f} "
                f"q_error_rad={np.max(np.abs(self._active_grasp_q - q)):.6f} "
                f"fk_goal_delta_m={np.linalg.norm(goal_position_base - current_position_base):.6f} "
                f"visual_command_delta_m={requested_distance:.6f}"
            )
            return None
        # A five-axis branch can curve sideways temporarily even though its
        # validated endpoint is the requested grasp. Object displacement is
        # handled by the outer observation/replan threshold; rejecting a safe
        # branch merely because one tangent is not a Cartesian straight line
        # causes repeated retreat/reapproach cycles near singularities.
        del requested_base_delta
        # q_step lies on the cuMotion path checked by
        # is_grasp_transition_reachable.  Do not ask stochastic RRT to rediscover
        # the same short subsegment at every camera frame.
        return _JointStepPlan(q_step)

    def _plan_axis_constrained(
        self, q: np.ndarray, target: np.ndarray, target_key: tuple[float, ...]
    ):
        """Solve five-axis IK, then collision-check the c-space path.

        SO-101 has five arm joints.  Constraining XYZ plus the directed tool Z
        axis uses exactly five task dimensions.  cuMotion supplies calibrated
        FK/Jacobians and performs the collision-aware graph plan; the small
        least-squares solve avoids pretending that a sixth wrist DOF exists.
        """
        self.last_plan_diagnostic = {"reason": "ik_no_solution"}
        self.controller.synchronize_world()
        base_position, base_orientation = (
            self.controller.world_binding.get_world_interface().get_world_to_robot_base_transform()
        )
        position_base = isaac_sim_to_cumotion_translation(
            target[:3, 3], base_position, base_orientation
        )
        base_quaternion = np.asarray(
            base_orientation.numpy() if hasattr(base_orientation, "numpy") else base_orientation,
            dtype=np.float64,
        ).reshape(-1, 4)[0]
        R_world_base = quaternion_wxyz_to_matrix(base_quaternion)
        approach_axis_base = R_world_base.T @ target[:3, 2]
        closing_axis_base = R_world_base.T @ target[:3, 0]
        kinematics = self.controller.cumotion_robot.kinematics
        tool_frame = str(robot_config()["tool_frame"])
        if not self._fk_reported:
            actual_position_base = isaac_sim_to_cumotion_translation(
                self.arm.T_base_ee()[:3, 3], base_position, base_orientation
            )
            fk_position_base = np.asarray(
                kinematics.position(q, tool_frame), dtype=np.float64
            )
            print(
                "[mr_liu] SO-101 FK parity: "
                f"error_m={np.linalg.norm(fk_position_base - actual_position_base):.6f} "
                f"cuMotion={fk_position_base.round(6).tolist()} "
                f"USD={actual_position_base.round(6).tolist()} q={q.round(6).tolist()}"
            )
            self._fk_reported = True
        lower = np.asarray(
            [kinematics.cspace_coord_limits(i).lower for i in range(len(q))], dtype=np.float64
        )
        upper = np.asarray(
            [kinematics.cspace_coord_limits(i).upper for i in range(len(q))], dtype=np.float64
        )
        regularization = np.asarray([0.02, 0.02, 0.02, 0.03, 0.25], dtype=np.float64)
        closing_axis_weight = 0.30

        def residual(q_trial: np.ndarray) -> np.ndarray:
            pose = kinematics.pose(q_trial, tool_frame)
            axis = np.asarray(pose.rotation.matrix(), dtype=np.float64)[:, 2]
            closing_axis = np.asarray(pose.rotation.matrix(), dtype=np.float64)[:, 0]
            return np.concatenate(
                (
                    (np.asarray(pose.translation) - position_base) / 0.01,
                    axis - approach_axis_base,
                    closing_axis_weight * (closing_axis - closing_axis_base),
                    regularization * (q_trial - q),
                )
            )

        def jacobian(q_trial: np.ndarray) -> np.ndarray:
            pose = kinematics.pose(q_trial, tool_frame)
            axis = np.asarray(pose.rotation.matrix(), dtype=np.float64)[:, 2]
            closing_axis = np.asarray(pose.rotation.matrix(), dtype=np.float64)[:, 0]
            position_jacobian = np.asarray(
                kinematics.position_jacobian(q_trial, tool_frame), dtype=np.float64
            )
            angular_jacobian = np.asarray(
                kinematics.orientation_jacobian(q_trial, tool_frame), dtype=np.float64
            )
            axis_jacobian = np.column_stack(
                [np.cross(angular_jacobian[:, index], axis) for index in range(len(q_trial))]
            )
            closing_axis_jacobian = np.column_stack(
                [
                    np.cross(angular_jacobian[:, index], closing_axis)
                    for index in range(len(q_trial))
                ]
            )
            return np.vstack(
                (
                    position_jacobian / 0.01,
                    axis_jacobian,
                    closing_axis_weight * closing_axis_jacobian,
                    np.diag(regularization),
                )
            )

        cached_seeds = sorted(
            self._ik_seed_cache.items(),
            key=lambda item: np.linalg.norm(np.asarray(item[0]).reshape(4, 4)[:3, 3] - target[:3, 3]),
        )
        self.last_axis_ik_diagnostics = []
        self._observation_pose_suggestions = []
        self._observation_suggestion_key = target_key
        defaults = robot_config()["default_joint_positions"]
        home_seed = np.array([defaults[name] for name in self.controller.cumotion_robot.controlled_joint_names])
        seeds = [q, *(seed for _key, seed in cached_seeds[:6]), home_seed]
        for seed in seeds:
            result = least_squares(
                residual,
                np.clip(seed, lower + 1e-6, upper - 1e-6),
                jac=jacobian,
                bounds=(lower, upper),
                max_nfev=100,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            q_goal = np.clip(result.x, lower + 1e-4, upper - 1e-4)
            goal_pose = kinematics.pose(q_goal, tool_frame)
            goal_position = np.asarray(goal_pose.translation, dtype=np.float64)
            goal_axis = np.asarray(goal_pose.rotation.matrix(), dtype=np.float64)[:, 2]
            self.last_axis_ik_diagnostics.append({
                "position_error_m": float(np.linalg.norm(goal_position - position_base)),
                "axis_error_deg": float(np.rad2deg(axis_alignment_error(goal_axis, approach_axis_base))),
            })
            if (np.linalg.norm(goal_position - position_base) <= .003
                    and axis_alignment_error(goal_axis, approach_axis_base) <= np.deg2rad(15)):
                self._observation_pose_suggestions.append(self._world_ee_pose_from_joints(q_goal))
            position_error = float(np.linalg.norm(goal_position-position_base))
            axis_error = float(axis_alignment_error(goal_axis, approach_axis_base))
            self.last_plan_diagnostic = dict(reason="ik_tolerance", position_error_mm=position_error*1000,
                                             axis_error_deg=float(np.rad2deg(axis_error)))
            if (
                float(np.linalg.norm(goal_position - position_base)) > 0.003
                or axis_alignment_error(goal_axis, approach_axis_base) > np.deg2rad(5.0)
            ):
                continue
            path = self._linear_cspace_path(q, q_goal)
            if path is not None:
                self.last_plan_diagnostic["reason"] = "reachable"
                self._ik_seed_cache[target_key] = q_goal
                if len(self._ik_seed_cache) > 64:
                    self._ik_seed_cache.pop(next(iter(self._ik_seed_cache)))
                return path
            self.last_plan_diagnostic["reason"] = self.last_collision_reason or "path_blocked"
        return None

    def propose_observation_pose(self, requested):
        """Offer an achieved IK orientation near an infeasible view request.

        This is only a candidate, never permission to move. The caller must
        plan/check the exact returned pose and perform a new visual handoff.
        FineGrasp never calls this: grasp orientation constraints are unchanged.
        """
        if getattr(self, "_observation_suggestion_key", None) != self._pose_key(requested):
            return None
        candidates = getattr(self, "_observation_pose_suggestions", [])
        return None if not candidates else min(candidates, key=lambda p:
            axis_alignment_error(p[:3, 2], requested[:3, 2])).copy()

    def _linear_cspace_path(
        self,
        q_initial: np.ndarray,
        q_final: np.ndarray,
        *,
        collision_step_rad: float = 0.02,
        waypoint_step_rad: float = 0.08,
    ) -> _JointWaypointPath | None:
        """Deterministically validate a short local path with cuMotion.

        FineGrasp starts near the object and evaluates several candidates.  A
        sampling RRT is the wrong primitive here: an infeasible candidate can
        occupy the control thread for minutes.  Dense interpolation through
        cuMotion's robot/world collision inspector is bounded, repeatable and
        sufficient for these local moves.  A blocked segment is rejected so a
        different candidate can be tried by the model-independent selector.
        """
        q0 = np.asarray(q_initial, dtype=np.float64).reshape(-1)
        self.last_collision_reason = None
        q1 = np.asarray(q_final, dtype=np.float64).reshape(-1)
        max_delta = float(np.max(np.abs(q1 - q0)))
        collision_samples = max(1, int(np.ceil(max_delta / collision_step_rad)))
        for index in range(collision_samples + 1):
            alpha = index / collision_samples
            q = q0 + alpha * (q1 - q0)
            if self._collision_inspector.in_self_collision(q):
                self.last_collision_reason = "self_collision"
                return None
            if self._collision_inspector.in_collision_with_obstacle(q):
                self.last_collision_reason = "obstacle_collision"
                return None
        waypoint_count = max(1, int(np.ceil(max_delta / waypoint_step_rad)))
        return _JointWaypointPath(
            tuple(q0 + index / waypoint_count * (q1 - q0) for index in range(waypoint_count + 1))
        )

    def _execute_plan(self, plan, *, speed_scale: float, max_steps: int) -> bool:
        if isinstance(plan, _JointStepPlan):
            return self._drive_joint_waypoint(
                plan.q_target,
                speed_scale=speed_scale,
                max_steps=max_steps,
                velocity_tolerance_rad_s=0.5,
                cartesian_tolerance_m=0.005,
                # A wrist-camera servo segment may be only 3--5 mm in FK.
                # Requiring 0.5 mm from every such segment deadlocks a loaded
                # five-axis arm near a singularity even while consecutive
                # observations show useful motion.  A 0.1 mm lower bound still
                # guarantees that the command was physically acted upon; the
                # outer loop owns the strict 5 mm / three-frame final test.
                minimum_cartesian_progress_m=0.0001,
                accept_partial_cartesian_progress=True,
            )
        if isinstance(plan, _JointWaypointPath):
            # This path was produced by dense collision checks of a straight
            # c-space segment.  _drive_joint_waypoint ramps the command along
            # that same segment, so stopping at every sampled collision point
            # only injects oscillation and latency.
            return self._drive_joint_waypoint(
                plan.waypoints[-1], speed_scale=speed_scale, max_steps=max_steps
            )
        if hasattr(plan, "get_waypoints_count"):
            for waypoint_index in range(1, plan.get_waypoints_count()):
                waypoint = plan.get_waypoint_by_index(waypoint_index)
                if not self._drive_joint_waypoint(
                    waypoint, speed_scale=speed_scale, max_steps=max_steps
                ):
                    return False
            return True

        duration = float(plan.duration)
        speed = max(min(float(speed_scale), 1.0), 0.05)
        trajectory_time = 0.0
        for _ in range(max_steps):
            if self._stopped:
                return False
            trajectory_time = min(duration, trajectory_time + self.physics_dt_s * speed)
            state = plan.get_target_state(trajectory_time)
            if state is None or state.joints.positions is None:
                return False
            self.articulation.set_dof_position_targets(
                positions=state.joints.positions,
                dof_indices=state.joints.position_indices,
            )
            self.advance_frame()
            if trajectory_time >= duration:
                endpoint = np.asarray(state.joints.positions.numpy(), dtype=np.float64).reshape(-1)
                return self._drive_joint_waypoint(
                    endpoint, speed_scale=speed_scale, max_steps=max_steps
                )
        return False

    def _target_reached(self, target: np.ndarray) -> bool:
        current = self.arm.T_base_ee()
        translation, _ = pose_error(current, target)
        if float(np.linalg.norm(translation)) > max(self.position_tolerance_m * 2.0, 0.003):
            return False
        if self.orientation_mode == "approach_axis":
            return axis_alignment_error(current[:3, 2], target[:3, 2]) <= np.deg2rad(5.0)
        return True

    def _set_target_pose(self, T_base_ee: np.ndarray) -> None:
        T = np.asarray(T_base_ee, dtype=np.float64)
        self.target.set_world_poses(
            positions=[T[:3, 3]],
            orientations=[matrix_to_quaternion_wxyz(T[:3, :3])],
        )

    def _rmp_refine(self, T_base_ee: np.ndarray, *, speed_scale: float, max_steps: int) -> bool:
        del speed_scale  # RMPflow's limits are configured centrally in motion.yaml.
        self._set_target_pose(T_base_ee)
        self.controller.reset(self._time_s)
        for _ in range(max_steps):
            if self._stopped:
                return False
            translation, _rotation = pose_error(self.arm.T_base_ee(), T_base_ee)
            if float(np.linalg.norm(translation)) <= self.position_tolerance_m:
                self._plan_cache.clear()
                return True
            self._time_s += self.physics_dt_s
            self.controller.step(self._time_s)
            self.advance_frame()
        return False

    def _drive_joint_waypoint(
        self,
        waypoint: np.ndarray,
        *,
        speed_scale: float,
        max_steps: int,
        tolerance_rad: float | None = None,
        velocity_tolerance_rad_s: float = 0.2,
        cartesian_tolerance_m: float | None = 0.0035,
        minimum_cartesian_progress_m: float = 0.0,
        accept_partial_cartesian_progress: bool = False,
    ) -> bool:
        target = np.asarray(waypoint, dtype=np.float64).reshape(-1)
        tolerance = self.joint_tolerance_rad if tolerance_rad is None else float(tolerance_rad)
        kinematics = self.controller.cumotion_robot.kinematics
        tool_frame = str(robot_config()["tool_frame"])
        target_xyz = np.asarray(kinematics.position(target, tool_frame), dtype=np.float64)
        speed = max(min(float(speed_scale), 1.0), 0.05)
        max_delta_per_step = 0.035 * speed
        initial = self.articulation.get_dof_positions()
        if hasattr(initial, "numpy"):
            initial = initial.numpy()
        command = np.asarray(initial, dtype=np.float64).reshape(-1)[self.arm_dof_indices]
        initial_xyz = np.asarray(kinematics.position(command, tool_frame), dtype=np.float64)
        initial_cartesian_error = float(np.linalg.norm(target_xyz - initial_xyz))
        progress_required = min(
            float(minimum_cartesian_progress_m), 0.5 * initial_cartesian_error
        )
        pose_stable_frames = 0
        # ``max_steps`` budgets the commanded motion.  The loaded arm may only
        # enter the Cartesian tolerance near the end of that budget, especially
        # during lift. Keep commanding the fixed endpoint for a bounded settling
        # window and require both pose and velocity convergence before success.
        for _ in range(int(max_steps) + self.waypoint_settle_steps):
            if self._stopped:
                return False
            current = self.articulation.get_dof_positions()
            if hasattr(current, "numpy"):
                current = current.numpy()
            q = np.asarray(current, dtype=np.float64).reshape(-1)[self.arm_dof_indices]
            delta = target - q
            velocity = self._arm_values(self.articulation.get_dof_velocities)
            current_xyz = np.asarray(kinematics.position(q, tool_frame), dtype=np.float64)
            cartesian_reached = (
                cartesian_tolerance_m is not None
                and float(np.linalg.norm(target_xyz - current_xyz)) <= cartesian_tolerance_m
            )
            cartesian_progress = initial_cartesian_error - float(
                np.linalg.norm(target_xyz - current_xyz)
            )
            pose_reached = float(np.max(np.abs(delta))) <= tolerance or cartesian_reached
            progress_reached = cartesian_progress + 1e-6 >= progress_required
            pose_stable_frames = pose_stable_frames + 1 if pose_reached and progress_reached else 0
            if (
                pose_reached
                and progress_reached
                # PhysX can report a non-zero instantaneous joint velocity at
                # a force-limited equilibrium even while the sampled Cartesian
                # pose remains inside tolerance. Accept a sustained in-tolerance
                # endpoint as the equivalent settling signal.
                and (
                    float(np.max(np.abs(velocity))) <= velocity_tolerance_rad_s
                    or pose_stable_frames >= 30
                )
            ):
                return True
            # Advance the commanded trajectory independently of tracking lag.
            # Re-anchoring every target to the measured q limits the position
            # error (and therefore PD torque) to one tiny increment, so a loaded
            # shoulder can never catch up even when its configured effort is
            # sufficient.
            command += np.clip(target - command, -max_delta_per_step, max_delta_per_step)
            self.articulation.set_dof_position_targets(command, dof_indices=self.arm_dof_indices)
            self.advance_frame()
        print(
            "[mr_liu] joint waypoint did not converge: "
            f"max_error_rad={np.max(np.abs(delta)):.6f} "
            f"target={target.round(5).tolist()} current={q.round(5).tolist()} "
            f"target_xyz={target_xyz.round(5).tolist()} "
            f"current_xyz={current_xyz.round(5).tolist()} "
            f"initial_cartesian_error_m={initial_cartesian_error:.6f} "
            f"cartesian_progress_m={cartesian_progress:.6f} "
            f"effort={self._arm_values(self.articulation.get_dof_efforts).round(4).tolist()} "
            f"velocity={self._arm_values(self.articulation.get_dof_velocities).round(4).tolist()}"
        )
        # Servo segments are deliberately bounded.  On a loaded compact arm,
        # gravity sag can leave the FK endpoint outside this segment's virtual
        # Cartesian tolerance even though the collision-checked command made
        # useful physical progress.  The caller will consume a *fresh* wrist
        # frame immediately, so accepting that progress is safer and more
        # faithful to visual servoing than discarding the observation and
        # replanning from stale geometry.  Global moves and lift never enable
        # this path and therefore still require endpoint convergence.
        if accept_partial_cartesian_progress:
            final_error = float(np.linalg.norm(target_xyz - current_xyz))
            final_progress = initial_cartesian_error - final_error
            if (
                np.isfinite(final_progress)
                and final_progress + 1e-6 >= progress_required
                and final_error < initial_cartesian_error
            ):
                print(
                    "[mr_liu] accepting bounded servo progress for fresh re-observation: "
                    f"progress_m={final_progress:.6f} residual_m={final_error:.6f}"
                )
                return True
        return False

    def _arm_values(self, getter: Callable[..., object]) -> np.ndarray:
        value = getter(dof_indices=self.arm_dof_indices)
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1)
