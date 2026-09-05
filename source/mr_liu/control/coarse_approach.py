"""Label -> live metric tracking -> guarded coarse transit -> FineGrasp handoff."""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from mr_liu.grasp.contact import FingerGeometry
from mr_liu.perception.semantic_target import TargetObservationError


class CoarseApproach:
    def __init__(self, tracker, motion, *, trace=lambda event: None, clock=time.monotonic,
                 timeout_s=100., clearance_m=0.060, step_m=0.040, speed_scale=0.65):
        self.tracker, self.motion, self.trace, self.clock = tracker, motion, trace, clock
        self.timeout_s, self.clearance_m, self.step_m, self.speed_scale = timeout_s, clearance_m, step_m, speed_scale
        self.guard_reason = None
        self.command_center = None
        self.guard_count = self.move_count = 0
        self.started = 0.
        self.finger = FingerGeometry(open_width_m=0.075)

    def _guard(self):
        self.guard_count += 1
        try:
            if self.clock() - self.started > self.timeout_s:
                raise TargetObservationError("coarse_timeout")
            evidence = self.tracker.observe()
            if np.linalg.norm(evidence.center_base_m - self.command_center) > 0.012:
                raise TargetObservationError("target_moved_during_transit")
        except TargetObservationError as exc:
            self.guard_reason = str(exc)
            self.trace({"phase": "coarse", "event": "coarse_guard_stop", "reason": str(exc)})
            return False
        return True

    def run(self):
        self.started = self.clock()
        self.motion.stop()
        self.motion.clear_grasp_plan()
        self.trace({"phase": "find", "event": "label_request", "label": self.tracker.target.semantic_label})
        evidence = self.tracker.find()
        initial_pose = self.motion.robot_state().T_base_ee.copy()
        self.trace({"phase": "coarse", "event": "coarse_start", "initial_T_base_ee": initial_pose.tolist(),
                    "initial_target_distance_m": float(np.linalg.norm(initial_pose[:3, 3] - evidence.center_base_m))})
        # Generic downward approach, not a per-object joint preset.
        x = initial_pose[:3, 0].copy()
        x[2] = 0
        if np.linalg.norm(x) < 1e-3:
            x = np.array([1., 0., 0.])
        x /= np.linalg.norm(x)
        z = np.array([0., 0., -1.])
        rotation = np.column_stack((x, np.cross(z, x), z))
        reached_xy = False
        try:
            for iteration in range(24):
                if self.clock() - self.started > self.timeout_s:
                    raise TargetObservationError("coarse_timeout")
                try:
                    evidence = self.tracker.observe()
                except TargetObservationError:
                    self.motion.stop()
                    if self.tracker.find_count >= 3:
                        raise
                    evidence = self.tracker.find()
                pose = self.motion.robot_state().T_base_ee.copy()
                center = np.eye(4)
                center[:3, :3] = rotation
                center[:3, 3] = evidence.center_base_m
                center[2, 3] = evidence.top_z_m + self.clearance_m
                goal = self.motion.ee_pose_for_grasp(center, 0.075)
                xy_error = float(np.linalg.norm(pose[:2, 3] - goal[:2, 3]))
                if xy_error < 0.008:
                    reached_xy = True
                if not reached_xy:
                    # Preserve a higher starting height, but don't invent a
                    # 12 cm corridor outside the small arm's workspace. Every
                    # segment still requires dense world + finger path checks.
                    goal[2, 3] = max(pose[2, 3], evidence.top_z_m + self.clearance_m)
                error = goal[:3, 3] - pose[:3, 3]
                if reached_xy and np.linalg.norm(error) < 0.005:
                    # Explicitly stop the coarse owner. FineGrasp must still
                    # validate a fresh eye-in-hand observation before acting.
                    self.motion.stop()
                    latest = self.tracker.observe()
                    if np.linalg.norm(latest.center_base_m - evidence.center_base_m) > 0.008:
                        continue
                    self.trace({"phase": "handoff", "event": "coarse_handoff", "sequence": latest.observation.sequence,
                        "center_base_m": latest.center_base_m.tolist(), "T_base_ee": pose.tolist(),
                        "tip_height_above_target_m": float(pose[2, 3] - latest.top_z_m),
                        "elapsed_s": self.clock()-self.started, "moves": self.move_count, "guard_checks": self.guard_count})
                    return replace(self.tracker.target, coarse_position_base_m=tuple(latest.center_base_m))
                command = goal.copy()
                command[:3, 3] = pose[:3, 3] + error * min(1., self.step_m / max(np.linalg.norm(error), 1e-6))
                if not self.motion.is_observation_path_safe(command, evidence.points_base, self.finger):
                    self.trace({"phase": "coarse", "event": "coarse_path_rejected",
                        "iteration": iteration, "command_T_base_ee": command.tolist(),
                        "reason": getattr(self.motion, "last_observation_path_rejection", None)})
                    raise TargetObservationError("coarse_path_infeasible")
                # IK may take seconds. Recheck the scene before executing its
                # cached path; a changed target invalidates the command.
                latest = self.tracker.observe()
                if np.linalg.norm(latest.center_base_m - evidence.center_base_m) > 0.008:
                    self.motion.clear_grasp_plan()
                    continue
                self.command_center = latest.center_base_m.copy()
                self.guard_reason = None
                self.motion.execution_guard = self._guard
                self.trace({"phase": "coarse", "event": "coarse_move", "iteration": iteration,
                            "sequence": latest.observation.sequence, "command_T_base_ee": command.tolist(),
                            "speed_scale": self.speed_scale})
                self.move_count += 1
                try:
                    completed = self.motion.move_to(command, speed_scale=self.speed_scale)
                finally:
                    self.motion.execution_guard = None
                if not completed:
                    self.motion.stop()
                    self.motion.clear_grasp_plan()
                    if self.guard_reason is not None:
                        continue
                    raise TargetObservationError("coarse_motion_not_converged")
            raise TargetObservationError("coarse_step_budget_exhausted")
        finally:
            self.motion.execution_guard = None
            self.motion.stop()
            self.motion.clear_grasp_plan()
