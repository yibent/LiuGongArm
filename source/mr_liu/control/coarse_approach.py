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
        if not 0.05 <= clearance_m <= 0.08 or not 0 < step_m <= 0.04 or not 0 < speed_scale <= 1:
            raise ValueError("Coarse approach requires 5-8 cm clearance, <=4 cm steps and bounded speed")
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
        clearance = self.clearance_m
        tried_inward_closing_axis = False
        tilt_options = iter((15., 30.))
        ik_view_adjustments = 0
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
                center[2, 3] = evidence.top_z_m + clearance
                goal = self.motion.ee_pose_for_grasp(center, 0.075)
                xy_error = float(np.linalg.norm(pose[:2, 3] - goal[:2, 3]))
                if xy_error < 0.008:
                    reached_xy = True
                if not reached_xy:
                    # Preserve a higher starting height, but don't invent a
                    # 12 cm corridor outside the small arm's workspace. Every
                    # segment still requires dense world + finger path checks.
                    goal[2, 3] = max(pose[2, 3], evidence.top_z_m + clearance)
                error = goal[:3, 3] - pose[:3, 3]
                axis_error = np.arccos(np.clip(np.dot(pose[:3, 2], goal[:3, 2]), -1., 1.))
                if reached_xy and np.linalg.norm(error) < 0.005 and axis_error <= np.deg2rad(5):
                    # Explicitly stop the coarse owner. FineGrasp must still
                    # validate a fresh eye-in-hand observation before acting.
                    self.motion.stop()
                    latest = self.tracker.observe()
                    if np.linalg.norm(latest.center_base_m - evidence.center_base_m) > 0.008:
                        continue
                    self.tracker.handoff_after_s = self.clock()
                    self.trace({"phase": "handoff", "event": "coarse_handoff", "sequence": latest.observation.sequence,
                        "center_base_m": latest.center_base_m.tolist(), "T_base_ee": pose.tolist(),
                        "tip_height_above_target_m": float(pose[2, 3] - latest.top_z_m),
                        "requested_clearance_m": clearance,
                        "elapsed_s": self.clock()-self.started, "moves": self.move_count, "guard_checks": self.guard_count})
                    return replace(self.tracker.target, coarse_position_base_m=tuple(latest.center_base_m))
                command = goal.copy()
                command[:3, 3] = pose[:3, 3] + error * min(1., self.step_m / max(np.linalg.norm(error), 1e-6))
                safe = self.motion.is_observation_path_safe(command, evidence.points_base, self.finger)
                if not safe:
                    self.trace({"phase": "coarse", "event": "coarse_path_rejected",
                        "iteration": iteration, "command_T_base_ee": command.tolist(),
                        "ik_diagnostics": getattr(self.motion, "last_axis_ik_diagnostics", None),
                        "reason": getattr(self.motion, "last_observation_path_rejection", None)})
                    proposer = getattr(self.motion, "propose_observation_pose", None)
                    suggestion = proposer(command) if proposer is not None and ik_view_adjustments < 3 else None
                    if (suggestion is not None and np.dot(suggestion[:3, 2], z) >= np.cos(np.deg2rad(30))
                            and np.linalg.norm(suggestion[:3, 3] - command[:3, 3]) <= .003):
                        safe = self.motion.is_observation_path_safe(suggestion, evidence.points_base, self.finger)
                        if safe:
                            command = suggestion
                            rotation = suggestion[:3, :3].copy()
                            reached_xy = False
                            ik_view_adjustments += 1
                            self.trace({"phase": "coarse", "event": "coarse_ik_view_adjustment",
                                        "T_base_ee": suggestion.tolist(), "adjustment": ik_view_adjustments})
                if not safe:
                    if clearance > 0.05001:
                        # Generic workspace adaptation, independent of class:
                        # try another handoff inside the 5-8 cm fine-entry band.
                        # This is a new proposal, never a safety-check bypass.
                        clearance = max(0.05, clearance - 0.01)
                        reached_xy = False
                        self.motion.clear_grasp_plan()
                        self.trace({"phase": "coarse", "event": "coarse_alternate_clearance",
                                    "clearance_m": clearance})
                        continue
                    if not tried_inward_closing_axis:
                        tried_inward_closing_axis = True
                        inward = initial_pose[:3, 3] - evidence.center_base_m
                        inward[2] = 0
                        if np.linalg.norm(inward) > 1e-3:
                            inward /= np.linalg.norm(inward)
                            # The asymmetric fixed-finger EE is offset from the
                            # open pinch centre. A safe inward closing axis can
                            # reduce reach without lowering clearance further.
                            rotation = np.column_stack((inward, np.cross(z, inward), z))
                            reached_xy = False
                            self.motion.clear_grasp_plan()
                            self.trace({"phase": "coarse", "event": "coarse_alternate_orientation",
                                        "closing_axis": inward.tolist()})
                            continue
                    tilt = next(tilt_options, None)
                    if tilt is not None:
                        # The coarse view need not impose a vertical grasp on
                        # a five-axis arm. Bounded slanted view proposals retain
                        # the same observed pinch-centre clearance and all checks.
                        inward = initial_pose[:3, 3] - evidence.center_base_m
                        inward[2] = 0
                        if np.linalg.norm(inward) > 1e-3:
                            inward /= np.linalg.norm(inward)
                            axis = -inward * np.sin(np.deg2rad(tilt)) + z * np.cos(np.deg2rad(tilt))
                            closing = inward - axis * np.dot(inward, axis)
                            closing /= np.linalg.norm(closing)
                            rotation = np.column_stack((closing, np.cross(axis, closing), axis))
                            reached_xy = False
                            self.motion.clear_grasp_plan()
                            self.trace({"phase": "coarse", "event": "coarse_alternate_tilt", "tilt_deg": tilt})
                            continue
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
