"""LeRobot SO-101 bridges for a calibrated wrist RGB-D camera and gripper.

The module intentionally depends only on LeRobot's public ``Robot`` shape
(``get_observation``/``send_action``).  Importing BusAgent therefore does not
require LeRobot or a connected serial device, and the same adapters can be
tested with a fake robot before any motor is enabled.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import numpy as np

from mr_liu.grasp.contracts import (
    CameraIntrinsics,
    GripperState,
    RGBDObservation,
    RobotState,
    TargetSpec,
)


@dataclass(frozen=True)
class SynchronizedLeRobotRGBD:
    """One driver-owned camera/joint sample in a shared clock domain."""

    observation: Mapping[str, object]
    rgb: np.ndarray
    depth: np.ndarray
    timestamp_s: float


class LeRobotLike(Protocol):
    def get_observation(self) -> Mapping[str, object]: ...

    def send_action(self, action: Mapping[str, float]) -> Mapping[str, float]: ...


class KinematicsCollisionPlanner(Protocol):
    """Robot-independent IK/collision boundary used by the real-arm adapter."""

    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray: ...

    def solve_ik(
        self, joint_positions_rad: np.ndarray, T_base_ee: np.ndarray
    ) -> np.ndarray | None: ...

    def is_path_collision_free(
        self,
        start_rad: np.ndarray,
        goal_rad: np.ndarray,
        *,
        margin_m: float,
        allow_target_contact: bool,
    ) -> bool: ...


class LeRobotWristCamera:
    """Make synchronized LeRobot RGB-D frames obey the eye-in-hand contract.

    ``ee_pose_from_observation`` must run forward kinematics from the joint
    values in the *same* LeRobot observation.  The fixed hand-eye transform is
    never inferred at runtime: it must come from a validated calibration.
    """

    def __init__(
        self,
        robot: LeRobotLike,
        *,
        camera_key: str,
        intrinsics: CameraIntrinsics,
        T_ee_camera: np.ndarray,
        ee_pose_from_observation: Callable[[Mapping[str, object]], np.ndarray],
        synchronized_snapshot: Callable[[], SynchronizedLeRobotRGBD] | None = None,
        require_synchronized_snapshot: bool = True,
        depth_scale_m: float = 0.001,
        color_order: str = "rgb",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot = robot
        self.camera_key = str(camera_key)
        self.intrinsics = intrinsics
        self.T_ee_camera = _validated_transform(T_ee_camera, "T_ee_camera")
        self.ee_pose_from_observation = ee_pose_from_observation
        self.synchronized_snapshot = synchronized_snapshot
        self.require_synchronized_snapshot = bool(require_synchronized_snapshot)
        self.depth_scale_m = float(depth_scale_m)
        if not np.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0:
            raise ValueError("depth_scale_m must be positive and finite")
        if color_order.casefold() not in {"rgb", "bgr"}:
            raise ValueError("color_order must be 'rgb' or 'bgr'")
        self.color_order = color_order.casefold()
        self.clock = clock
        self._sequence = 0
        self._lock = threading.Lock()

    def capture(self, target: TargetSpec) -> RGBDObservation | None:
        del target
        with self._lock:
            observation, rgb, raw_depth, source_timestamp = self._capture_snapshot()
            now = float(self.clock())
            timestamp = now if source_timestamp is None else float(source_timestamp)
            if not np.isfinite(timestamp) or timestamp > now + 0.05:
                raise ValueError("Wrist camera timestamp is outside the controller clock domain")

            rgb = np.asarray(rgb, dtype=np.uint8)
            depth = np.asarray(raw_depth)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            if rgb.shape != (self.intrinsics.height, self.intrinsics.width, 3):
                raise ValueError(f"Unexpected wrist RGB shape: {rgb.shape}")
            if depth.shape != (self.intrinsics.height, self.intrinsics.width):
                raise ValueError(f"Unexpected wrist depth shape: {depth.shape}")
            if self.color_order == "bgr":
                rgb = rgb[:, :, ::-1]
            depth_m = np.asarray(depth, dtype=np.float32) * self.depth_scale_m
            depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = np.nan

            T_base_ee = _validated_transform(
                self.ee_pose_from_observation(observation), "T_base_ee"
            )
            T_base_camera = T_base_ee @ self.T_ee_camera
            self._sequence += 1
            return RGBDObservation(
                sequence=self._sequence,
                timestamp_s=timestamp,
                rgb=np.ascontiguousarray(rgb),
                depth_m=np.ascontiguousarray(depth_m),
                intrinsics=self.intrinsics,
                T_base_camera=T_base_camera,
                T_base_ee=T_base_ee,
                T_ee_camera=self.T_ee_camera.copy(),
                metadata={
                    "adapter": "lerobot",
                    "camera_key": self.camera_key,
                    "capture_age_ms": max(0.0, (now - timestamp) * 1000.0),
                },
            )

    def _capture_snapshot(
        self,
    ) -> tuple[Mapping[str, object], np.ndarray, np.ndarray, float | None]:
        """Capture RGB, depth and FK inputs as one driver-owned transaction.

        Reading a camera object's ``latest_*`` buffers after calling
        ``get_observation`` is intentionally forbidden: the frame can advance
        while the joint values remain old.  Production RealSense integrations
        should inject a provider which holds the camera/joint snapshot lock and
        returns their shared timestamp.
        """
        if self.synchronized_snapshot is not None:
            snapshot = self.synchronized_snapshot()
            return (
                snapshot.observation,
                np.asarray(snapshot.rgb).copy(),
                np.asarray(snapshot.depth).copy(),
                float(snapshot.timestamp_s),
            )
        observation = self.robot.get_observation()
        rgb_key = self.camera_key
        depth_key = f"{self.camera_key}_depth"
        if rgb_key not in observation or depth_key not in observation:
            raise KeyError(
                f"LeRobot observation requires '{rgb_key}' and '{depth_key}'; "
                "configure the wrist camera with use_depth=True"
            )
        timestamp_key = f"{self.camera_key}_timestamp"
        timestamp = observation.get(timestamp_key)
        if timestamp is None and self.require_synchronized_snapshot:
            raise RuntimeError(
                "Strict eye-in-hand synchronization requires either synchronized_snapshot "
                f"or '{timestamp_key}' in the same LeRobot observation"
            )
        return (
            observation,
            np.asarray(observation[rgb_key]).copy(),
            np.asarray(observation[depth_key]).copy(),
            None if timestamp is None else float(timestamp),
        )


@dataclass(frozen=True)
class LeRobotGripperCalibration:
    closed_position: float = 0.0
    open_position: float = 100.0
    max_width_m: float = 0.075

    def position_for_width(self, width_m: float) -> float:
        alpha = float(np.clip(width_m / self.max_width_m, 0.0, 1.0))
        return self.closed_position + alpha * (self.open_position - self.closed_position)

    def width_for_position(self, position: float) -> float:
        span = self.open_position - self.closed_position
        alpha = (float(position) - self.closed_position) / span
        return float(np.clip(alpha, 0.0, 1.0) * self.max_width_m)


class LeRobotGripperController:
    """Ramped SO-101 gripper actions with explicit force-limit ownership.

    LeRobot's public robot action controls position but does not express force.
    A calibrated ``set_force_limit_n`` callback is therefore mandatory by
    default.  This avoids silently claiming force control while leaving the
    Feetech controller at an unknown torque limit.
    """

    def __init__(
        self,
        robot: LeRobotLike,
        *,
        set_force_limit_n: Callable[[float], None] | None,
        calibration: LeRobotGripperCalibration | None = None,
        require_force_limit: bool = True,
        control_period_s: float = 0.02,
        position_tolerance: float = 1.0,
        stall_position_delta: float = 0.03,
        stall_samples: int = 12,
        settle_samples: int = 40,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.set_force_limit_n = set_force_limit_n
        self.calibration = calibration or LeRobotGripperCalibration()
        self.require_force_limit = bool(require_force_limit)
        self.control_period_s = float(control_period_s)
        self.position_tolerance = float(position_tolerance)
        self.stall_position_delta = float(stall_position_delta)
        self.stall_samples = int(stall_samples)
        self.settle_samples = int(settle_samples)
        self.clock = clock
        self.sleep = sleep
        self._stalled = False
        self._commanded_force_n: float | None = None

    def open(self, width_m: float, *, speed_mps: float) -> bool:
        self._stalled = False
        self._commanded_force_n = None
        return self._move(self.calibration.position_for_width(width_m), speed_mps, False)

    def close(self, width_m: float, *, force_n: float, speed_mps: float) -> bool:
        if self.set_force_limit_n is None and self.require_force_limit:
            return False
        if self.set_force_limit_n is not None:
            self.set_force_limit_n(float(force_n))
        self._commanded_force_n = float(force_n)
        return self._move(self.calibration.position_for_width(width_m), speed_mps, True)

    def state(self) -> GripperState:
        position = self._position()
        return GripperState(
            timestamp_s=self.clock(),
            width_m=self.calibration.width_for_position(position),
            effort_n=self._commanded_force_n,
            contact=self._stalled,
            stalled=self._stalled,
        )

    def _move(self, target: float, speed_mps: float, accept_stall: bool) -> bool:
        current = self._position()
        width_rate = max(float(speed_mps), 1e-4)
        position_rate = width_rate / self.calibration.max_width_m * abs(
            self.calibration.open_position - self.calibration.closed_position
        )
        step = max(position_rate * self.control_period_s, 0.05)
        ideal_steps = int(np.ceil(abs(target - current) / step))
        max_steps = ideal_steps + max(self.settle_samples, self.stall_samples * 2)
        stagnant = 0
        command = current
        self._stalled = False
        for _ in range(max_steps):
            if abs(target - current) <= self.position_tolerance:
                return True
            command += float(np.clip(target - command, -step, step))
            self.robot.send_action({"gripper.pos": command})
            self.sleep(self.control_period_s)
            measured = self._position()
            loaded = abs(command - measured) > 4.0 * self.position_tolerance
            no_motion = abs(measured - current) < self.stall_position_delta
            stagnant = stagnant + 1 if loaded and no_motion else 0
            current = measured
            if stagnant >= self.stall_samples:
                self._stalled = True
                return accept_stall
        return abs(target - current) <= self.position_tolerance

    def _position(self) -> float:
        observation = self.robot.get_observation()
        try:
            return float(observation["gripper.pos"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("LeRobot observation has no numeric 'gripper.pos'") from exc


class FeetechTorqueLimitBridge:
    """Calibrated Newton-to-Feetech torque-limit callback for SO-101.

    The full-scale jaw force is installation-specific and must be measured.
    The default maximum register is kept at LeRobot's conservative 50% limit.
    """

    def __init__(
        self,
        robot: LeRobotLike,
        *,
        full_scale_force_n: float,
        max_register: int = 500,
    ) -> None:
        if full_scale_force_n <= 0:
            raise ValueError("full_scale_force_n must be calibrated and positive")
        self.robot = robot
        self.full_scale_force_n = float(full_scale_force_n)
        self.max_register = int(np.clip(max_register, 1, 500))

    def __call__(self, force_n: float) -> None:
        bus = getattr(self.robot, "bus", None)
        if bus is None or not hasattr(bus, "write"):
            raise RuntimeError("LeRobot SO-101 Feetech bus is unavailable")
        register = int(
            np.clip(round(float(force_n) / self.full_scale_force_n * 1000.0), 1, self.max_register)
        )
        bus.write("Max_Torque_Limit", "gripper", register)


class PlannedLeRobotMotionExecutor:
    """Execute externally planned IK solutions through LeRobot's public API.

    Neural grasp generation never controls joints directly.  A cuRobo, MoveIt,
    or equivalent deterministic planner is injected through
    :class:`KinematicsCollisionPlanner`; this bridge only performs bounded
    joint interpolation and feedback acknowledgement on the calibrated arm.
    """

    def __init__(
        self,
        robot: LeRobotLike,
        planner: KinematicsCollisionPlanner,
        *,
        joint_names: tuple[str, ...] = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ),
        observation_uses_degrees: bool = True,
        control_period_s: float = 0.02,
        max_joint_step_rad: float = 0.025,
        joint_tolerance_rad: float = 0.025,
        position_tolerance_m: float = 0.004,
        default_collision_margin_m: float = 0.006,
        max_steps: int = 400,
        pinch_center_x_per_width: float = -0.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.planner = planner
        self.joint_names = tuple(joint_names)
        self.observation_uses_degrees = bool(observation_uses_degrees)
        self.control_period_s = float(control_period_s)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.joint_tolerance_rad = float(joint_tolerance_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.default_collision_margin_m = float(default_collision_margin_m)
        self.max_steps = int(max_steps)
        self.pinch_center_x_per_width = float(pinch_center_x_per_width)
        self.clock = clock
        self.sleep = sleep
        self._stopped = False
        self._ik_cache: dict[tuple[float, ...], np.ndarray | None] = {}
        self._validated_paths: dict[tuple[float, ...], tuple[float, bool]] = {}

    def robot_state(self) -> RobotState:
        observation = self.robot.get_observation()
        q = self._joint_positions(observation)
        return RobotState(
            timestamp_s=self.clock(),
            T_base_ee=_validated_transform(self.planner.forward_kinematics(q), "T_base_ee"),
            joint_positions=dict(zip(self.joint_names, q, strict=True)),
        )

    def ee_pose_for_grasp(self, T_base_grasp_center: np.ndarray, width_m: float) -> np.ndarray:
        center = _validated_transform(T_base_grasp_center, "T_base_grasp_center")
        T_ee_center = np.eye(4, dtype=np.float64)
        T_ee_center[0, 3] = self.pinch_center_x_per_width * float(width_m)
        return center @ np.linalg.inv(T_ee_center)

    def is_reachable(self, T_base_ee: np.ndarray) -> bool:
        return self._goal_for_pose(T_base_ee) is not None

    def is_collision_free(
        self,
        T_base_ee: np.ndarray,
        *,
        margin_m: float,
        allow_target_contact: bool = False,
    ) -> bool:
        key = self._pose_key(T_base_ee)
        goal = self._goal_for_pose(T_base_ee)
        if goal is None:
            return False
        start = self._joint_positions(self.robot.get_observation())
        valid = self.planner.is_path_collision_free(
            start,
            goal,
            margin_m=float(margin_m),
            allow_target_contact=bool(allow_target_contact),
        )
        if valid:
            self._validated_paths[key] = (float(margin_m), bool(allow_target_contact))
        return bool(valid)

    def is_grasp_transition_reachable(
        self, T_base_pregrasp: np.ndarray, T_base_grasp: np.ndarray
    ) -> bool:
        pregrasp = self._goal_for_pose(T_base_pregrasp)
        grasp = self._goal_for_pose(T_base_grasp)
        if pregrasp is None or grasp is None:
            return False
        return bool(
            self.planner.is_path_collision_free(
                pregrasp,
                grasp,
                margin_m=self.default_collision_margin_m,
                allow_target_contact=True,
            )
        )

    def move_to(self, T_base_ee: np.ndarray, *, speed_scale: float) -> bool:
        return self._execute_pose(T_base_ee, speed_scale=speed_scale)

    def servo_to(self, T_base_ee: np.ndarray, *, speed_scale: float) -> bool:
        return self._execute_pose(T_base_ee, speed_scale=speed_scale)

    def stop(self) -> None:
        self._stopped = True
        q = self._joint_positions(self.robot.get_observation())
        self._send_joint_positions(q)

    def _execute_pose(self, T_base_ee: np.ndarray, *, speed_scale: float) -> bool:
        self._stopped = False
        key = self._pose_key(T_base_ee)
        goal = self._goal_for_pose(T_base_ee)
        if goal is None:
            return False
        start = self._joint_positions(self.robot.get_observation())
        margin, allow_target_contact = self._validated_paths.get(
            key, (self.default_collision_margin_m, False)
        )
        # Revalidate from the *current* measured state. A result obtained during
        # candidate filtering becomes stale after the pregrasp move.
        if not self.planner.is_path_collision_free(
            start,
            goal,
            margin_m=margin,
            allow_target_contact=allow_target_contact,
        ):
            return False

        speed = float(np.clip(speed_scale, 0.05, 1.0))
        step = max(self.max_joint_step_rad * speed, 1e-4)
        command = start.copy()
        for _ in range(self.max_steps):
            if self._stopped:
                return False
            current = self._joint_positions(self.robot.get_observation())
            if float(np.max(np.abs(goal - current))) <= self.joint_tolerance_rad:
                actual = self.planner.forward_kinematics(current)
                if np.linalg.norm(actual[:3, 3] - np.asarray(T_base_ee)[:3, 3]) <= self.position_tolerance_m:
                    self._ik_cache.clear()
                    self._validated_paths.clear()
                    return True
            command += np.clip(goal - command, -step, step)
            self._send_joint_positions(command)
            self.sleep(self.control_period_s)
        return False

    def _goal_for_pose(self, T_base_ee: np.ndarray) -> np.ndarray | None:
        target = _validated_transform(T_base_ee, "T_base_ee")
        key = self._pose_key(target)
        if key not in self._ik_cache:
            start = self._joint_positions(self.robot.get_observation())
            solution = self.planner.solve_ik(start, target)
            self._ik_cache[key] = (
                None if solution is None else np.asarray(solution, dtype=np.float64).reshape(-1)
            )
        result = self._ik_cache[key]
        if result is not None and result.shape != (len(self.joint_names),):
            raise ValueError("IK solution size does not match controlled LeRobot joints")
        return result

    def _joint_positions(self, observation: Mapping[str, object]) -> np.ndarray:
        try:
            values = np.asarray(
                [float(observation[f"{name}.pos"]) for name in self.joint_names],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("LeRobot observation is missing calibrated arm positions") from exc
        return np.deg2rad(values) if self.observation_uses_degrees else values

    def _send_joint_positions(self, q_rad: np.ndarray) -> None:
        values = np.rad2deg(q_rad) if self.observation_uses_degrees else q_rad
        action = {f"{name}.pos": float(value) for name, value in zip(self.joint_names, values, strict=True)}
        self.robot.send_action(action)

    @staticmethod
    def _pose_key(value: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(value, dtype=np.float64).reshape(-1), 5))


def _validated_transform(value: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(T[:3, :3]) < 0.999:
        raise ValueError(f"{name} rotation must be right-handed")
    return T.copy()
