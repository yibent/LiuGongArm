"""SO-101 Isaac gripper adapter with calibrated width/joint conversion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from mr_liu.grasp.contracts import GripperState


class _Articulation(Protocol):
    dof_names: list[str]

    def get_dof_positions(self): ...

    def get_dof_efforts(self, **kwargs): ...

    def set_dof_position_targets(self, positions, **kwargs) -> None: ...

    def set_dof_max_efforts(self, max_efforts, **kwargs) -> None: ...


@dataclass(frozen=True)
class LinearGripperCalibration:
    closed_joint_rad: float = -0.05
    # The Isaac USD exposes a wider mechanical range than LeRobot's nominal
    # 0..100 mapping. 1.70 rad is the calibrated clear-FOV/open endpoint; the
    # nominal ~1.30 rad pose places the moving jaw across the wrist lens.
    open_joint_rad: float = 1.70
    max_width_m: float = 0.075
    jaw_lever_arm_m: float = 0.040

    def joint_for_width(self, width_m: float) -> float:
        width = float(np.clip(width_m, 0.0, self.max_width_m))
        alpha = width / self.max_width_m
        return self.closed_joint_rad + alpha * (self.open_joint_rad - self.closed_joint_rad)

    def width_for_joint(self, joint_rad: float) -> float:
        span = self.open_joint_rad - self.closed_joint_rad
        alpha = (float(joint_rad) - self.closed_joint_rad) / span
        return float(np.clip(alpha, 0.0, 1.0) * self.max_width_m)

    def joint_effort_for_force(self, force_n: float) -> float:
        return max(0.0, float(force_n)) * self.jaw_lever_arm_m


class IsaacGripperController:
    """Position-ramped gripper control with force cap and stall/contact evidence.

    The SO-101 simulator exposes a revolute jaw, whereas BusAgent commands an
    opening in metres.  Closing accepts a stalled joint as a completed command;
    the node's independent RGB-D verifier decides whether that stall is a held
    object or an incidental collision.
    """

    def __init__(
        self,
        articulation: _Articulation,
        *,
        advance_frame: Callable[[], None],
        physics_dt_s: float,
        calibration: LinearGripperCalibration | None = None,
        observation_width_m: float = 0.075,
        joint_tolerance_rad: float = 0.018,
        stall_delta_rad: float = 1e-5,
        stall_frames: int = 30,
        settle_frames: int = 180,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.articulation = articulation
        self.advance_frame = advance_frame
        self.physics_dt_s = float(physics_dt_s)
        self.calibration = calibration or LinearGripperCalibration()
        self.observation_width_m = float(observation_width_m)
        self.joint_tolerance_rad = float(joint_tolerance_rad)
        self.stall_delta_rad = float(stall_delta_rad)
        self.stall_frames = int(stall_frames)
        self.settle_frames = int(settle_frames)
        self.clock = clock
        self.joint_index = list(articulation.dof_names).index("gripper")
        self._last_effort_n: float | None = None
        self._stalled = False

    def open(self, width_m: float, *, speed_mps: float) -> bool:
        # Keep the jaw outside the useful wrist-camera region while observing.
        visible_width = max(float(width_m), self.observation_width_m)
        self._stalled = False
        self._last_effort_n = None
        return self._move_joint(
            self.calibration.joint_for_width(visible_width),
            speed_mps=speed_mps,
            accept_stall=False,
        )

    def close(self, width_m: float, *, force_n: float, speed_mps: float) -> bool:
        effort = self.calibration.joint_effort_for_force(force_n)
        self.articulation.set_dof_max_efforts([effort], dof_indices=[self.joint_index])
        self._last_effort_n = float(force_n)
        return self._move_joint(
            self.calibration.joint_for_width(width_m),
            speed_mps=speed_mps,
            accept_stall=True,
        )

    def state(self) -> GripperState:
        joint = self._joint_position()
        effort_n = self._measured_force_n()
        return GripperState(
            timestamp_s=self.clock(),
            width_m=self.calibration.width_for_joint(joint),
            effort_n=effort_n,
            # This adapter has joint-stall evidence, not a target contact sensor.
            contact=None,
            stalled=self._stalled,
        )

    def _move_joint(self, target_rad: float, *, speed_mps: float, accept_stall: bool) -> bool:
        current = self._joint_position()
        span = abs(self.calibration.open_joint_rad - self.calibration.closed_joint_rad)
        rate_rad_s = max(float(speed_mps), 1e-4) * span / self.calibration.max_width_m
        command_step = max(rate_rad_s * self.physics_dt_s, 1e-4)
        # The position target is deliberately ramped, but a force-limited jaw
        # can lag well behind that ramp while squeezing an object.  Budget a
        # separate settling phase after the ideal command duration rather than
        # treating ordinary drive lag as a failed command.
        max_steps = int(np.ceil(abs(target_rad - current) / command_step)) + max(
            self.settle_frames, self.stall_frames * 2
        )
        stagnant = 0
        window_start = current
        self._stalled = False
        command = current
        for _ in range(max_steps):
            delta = target_rad - current
            if abs(delta) <= self.joint_tolerance_rad:
                return True
            command += float(np.clip(target_rad - command, -command_step, command_step))
            self.articulation.set_dof_position_targets([command], dof_indices=[self.joint_index])
            self.advance_frame()
            measured = self._joint_position()
            no_motion = abs(measured - current) < self.stall_delta_rad
            drive_has_loaded = abs(command - measured) > 4.0 * self.joint_tolerance_rad
            stagnant = stagnant + 1 if no_motion and drive_has_loaded else 0
            current = measured
            if stagnant >= self.stall_frames:
                self._stalled = True
                return accept_stall
            if (_ + 1) % self.stall_frames == 0:
                window_motion = abs(current - window_start)
                if drive_has_loaded and window_motion < self.stall_delta_rad * self.stall_frames:
                    self._stalled = True
                    return accept_stall
                window_start = current
        return abs(target_rad - current) <= self.joint_tolerance_rad

    def _joint_position(self) -> float:
        value = self.articulation.get_dof_positions()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[self.joint_index])

    def _measured_force_n(self) -> float | None:
        try:
            value = self.articulation.get_dof_efforts(dof_indices=[self.joint_index])
            if hasattr(value, "numpy"):
                value = value.numpy()
            torque_nm = abs(float(np.asarray(value, dtype=np.float64).reshape(-1)[0]))
            return torque_nm / max(self.calibration.jaw_lever_arm_m, 1e-6)
        except (AttributeError, RuntimeError, ValueError):
            return None


class IsaacParallelGripperController:
    """Two measured prismatic fingers; widths in metres, efforts in newtons."""

    def __init__(self, articulation, *, advance_frame, physics_dt_s, clock=time.monotonic):
        self.articulation = articulation
        self.advance_frame, self.physics_dt_s, self.clock = advance_frame, physics_dt_s, clock
        self.indices = [list(articulation.dof_names).index(name)
                        for name in ("panda_finger_joint1", "panda_finger_joint2")]
        self._stalled = False
        self.max_width_m = .08

    def _positions(self):
        values = self.articulation.get_dof_positions()
        values = values.numpy() if hasattr(values, "numpy") else values
        return np.asarray(values, float).reshape(-1)[self.indices]

    def state(self):
        effort = None
        try:
            values = self.articulation.get_dof_efforts(dof_indices=self.indices)
            values = values.numpy() if hasattr(values, "numpy") else values
            effort = float(np.min(np.abs(values)))
        except (AttributeError, RuntimeError, ValueError):
            pass
        return GripperState(self.clock(), float(self._positions().sum()), effort,
                            contact=None, stalled=self._stalled)

    def open(self, width_m, *, speed_mps):
        self.articulation.set_dof_max_efforts([20., 20.], dof_indices=self.indices)
        return self._move(width_m, speed_mps, False)

    def close(self, width_m, *, force_n, speed_mps):
        force = float(np.clip(force_n, 0., 20.))
        self.articulation.set_dof_max_efforts([force, force], dof_indices=self.indices)
        return self._move(width_m, speed_mps, True)

    def _move(self, width, speed, closing):
        target = float(np.clip(width, 0., self.max_width_m)) * .5
        current = self._positions()
        command = current.copy()
        step = max(float(speed), .001) * .5 * self.physics_dt_s
        steps = int(np.ceil(np.max(np.abs(target-current))/step)) + 240
        still = np.zeros(2, dtype=int)
        self._stalled = False
        for _ in range(steps):
            if np.max(np.abs(current-target)) <= .0002:
                return True
            command += np.clip(target-command, -step, step)
            self.articulation.set_dof_position_targets(command.tolist(), dof_indices=self.indices)
            self.advance_frame()
            measured = self._positions()
            loaded = np.abs(command-measured) > .0008
            still = np.where((np.abs(measured-current)<1e-6)&loaded, still+1, 0)
            current = measured
            if np.all(still>=30):
                self._stalled = True
                # Maintain preload after a stall. Future arm stops must leave
                # these independent drive targets intact until release.
                if closing:
                    self.articulation.set_dof_position_targets([target, target], dof_indices=self.indices)
                return bool(closing)
        return False


def create_isaac_gripper(articulation, **kwargs):
    from mr_liu.config import robot_config
    if robot_config().get("gripper_type") == "parallel_prismatic":
        return IsaacParallelGripperController(articulation, **kwargs)
    return IsaacGripperController(articulation, **kwargs)
