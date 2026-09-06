"""Bounded failure-driven active perception around the stable FineGrasp API."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
import numpy as np

from mr_liu.grasp.contracts import FailureCode, FineGraspPhase, FineGraspResult
from mr_liu.grasp.contact import FingerGeometry, configured_finger_geometry


@dataclass(frozen=True)
class RecoveryConfig:
    max_attempts: int = 2
    max_total_s: float = 110.0
    retreat_m: float = 0.04
    speed_scale: float = 0.25
    empty_width_m: float = 0.001
    max_state_age_s: float = 0.25

    def __post_init__(self):
        if not 1 <= self.max_attempts <= 3 or not 0 < self.retreat_m <= 0.06:
            raise ValueError("Invalid bounded recovery configuration")
        if (not all(math.isfinite(v) for v in (self.max_total_s, self.retreat_m, self.speed_scale,
                                              self.empty_width_m, self.max_state_age_s))
                or self.max_total_s <= 0 or not 0 < self.speed_scale <= 0.4
                or self.empty_width_m < 0 or self.max_state_age_s <= 0):
            raise ValueError("Invalid recovery safety limits")


class RecoveringGraspNode:
    """Factory(attempt_index, failed_grasps, target) makes a fresh attempt node.

    First attempt can use the baseline perception; subsequent attempts require
    a fresh external target measurement and active wrist views. A possibly held
    object is NEVER released or moved by the recovery supervisor.
    """
    RETRYABLE = {FailureCode.NO_OBSERVATION, FailureCode.STALE_OBSERVATION,
                 FailureCode.TARGET_NOT_VISIBLE, FailureCode.INSUFFICIENT_DEPTH,
                 FailureCode.GEOMETRY_UNCERTAIN, FailureCode.IDENTITY_UNCERTAIN,
                 FailureCode.NO_GRASP_CANDIDATES, FailureCode.SERVO_TIMEOUT,
                 FailureCode.TARGET_MOVED, FailureCode.GRASP_NOT_DETECTED,
                 FailureCode.LIFT_VERIFICATION_FAILED}

    def __init__(self, *, attempt_factory, observer, motion, gripper,
                 config=RecoveryConfig(), trace=None, clock=time.monotonic,
                 require_retry_authorization=False):
        self.attempt_factory, self.observer = attempt_factory, observer
        self.motion, self.gripper, self.config = motion, gripper, config
        self.trace, self.clock = trace, clock
        self.active_node = None
        self.attempt_results = []
        self._cancelled = False
        self.require_retry_authorization = require_retry_authorization
        self.pending_retry = None

    def retry(self, request_id):
        """Resume one failed attempt only after a new conversational command.

        All payload, fresh-scene and retreat checks run again; no saved path is
        trusted after waiting for the operator. Demo auto-recovery is unchanged.
        """
        if self.pending_retry is None:
            raise ValueError("没有可恢复的抓取上下文。")
        request, result = self.pending_retry
        self.pending_retry = None
        try:
            return self._execute(replace(request, request_id=request_id), previous_result=result)
        finally:
            self.active_node = None

    def cancel(self):
        self._cancelled = True
        if self.active_node:
            self.active_node.cancel()
        self.motion.stop()

    def _event(self, name, **values):
        if self.trace:
            self.trace({"timestamp_s": self.clock(), "phase": "recovery", "event": name, **values})

    def execute(self, request):
        self.pending_retry = None
        try:
            return self._execute(request)
        except Exception as exc:
            self.motion.stop()
            self._event("recovery_exception", error=repr(exc))
            return FineGraspResult(False, FineGraspPhase.FAILED, request.request_id,
                FailureCode.INTERNAL_ERROR, f"Recovery error: {type(exc).__name__}: {exc}",
                metrics={"recovery_attempts": len(self.attempt_results),
                         "recovery_stop_reason": "exception_no_further_motion"})
        finally:
            self.active_node = None

    def _execute(self, request, previous_result=None):
        self._cancelled = False
        self.attempt_results = []
        if previous_result is None:
            self.observer.reset()
        started = self.clock()
        deadline = started + min(request.timeout_s or self.config.max_total_s, self.config.max_total_s)
        target, failed = request.target, []
        initial_scene = self.observer.latest if previous_result is not None else (None if request.dry_run else self.observer.observe(target))
        result = None
        stop_reason = "attempt_limit"
        for attempt in range(self.config.max_attempts):
            if self._cancelled or self.clock() >= deadline:
                stop_reason = "cancelled" if self._cancelled else "deadline"
                break
            if attempt == 0 and previous_result is not None:
                result = previous_result
            else:
                self._event("attempt_start", attempt=attempt + 1, active_views=attempt > 0)
                self.active_node = self.attempt_factory(attempt, tuple(failed), target)
                if self._cancelled or self.clock() >= deadline:
                    stop_reason = "cancelled" if self._cancelled else "deadline"
                    break
                result = self.active_node.execute(replace(request, target=target,
                                                          timeout_s=max(0.001, deadline - self.clock())))
            self.attempt_results.append(result)
            self._event("attempt_end", attempt=attempt + 1, success=result.success,
                        failure=result.failure.value if result.failure else None,
                        close_commanded=result.metrics.get("close_commanded", 0))
            if result.success or request.dry_run:
                stop_reason = "succeeded" if result.success else "dry_run"
                break
            self.motion.stop()
            if result.failure not in self.RETRYABLE:
                stop_reason = "non_retryable_failure"
                break
            if attempt + 1 >= self.config.max_attempts:
                break
            if self._cancelled or self.clock() >= deadline:
                stop_reason = "cancelled" if self._cancelled else "deadline"
                break
            state = self.gripper.state()
            closed = bool(result.metrics.get("close_commanded", 0))
            if (not math.isfinite(state.timestamp_s)
                    or not -0.02 <= self.clock() - state.timestamp_s <= self.config.max_state_age_s
                    or state.width_m is None or not math.isfinite(state.width_m) or state.width_m < 0):
                stop_reason = "payload_uncertain_do_not_open" if closed else "gripper_state_unverified"
                break
            # Width alone cannot prove possession, but a non-empty/unknown width
            # after closing is sufficient to forbid release or lateral scanning.
            if closed and (state.width_m is None or state.width_m > self.config.empty_width_m
                           or state.contact is True or state.stalled is True or target.properties.thin):
                stop_reason = "payload_uncertain_do_not_open"
                self._event("recovery_blocked", reason=stop_reason)
                break
            reference = self.observer.latest or initial_scene
            fresh = self.observer.observe(target)
            if fresh is None:
                stop_reason = "target_identity_uncertain"
                break
            confirm = self.observer.observe(target, fresh.center_base_m)
            if confirm is None or np.linalg.norm(confirm.center_base_m - fresh.center_base_m) > 0.004:
                stop_reason = "target_not_stationary"
                break
            if closed and (reference is None or
                           abs(confirm.center_base_m[2] - reference.center_base_m[2]) > 0.012):
                stop_reason = "payload_location_uncertain"
                break
            # A completed failed close can exclude a grasp neighborhood. Do not
            # blacklist a candidate merely because a camera frame was missing.
            if closed and result.selected_grasp is not None:
                failed.append(result.selected_grasp)
            reset_plan = getattr(self.motion, "clear_grasp_plan", None)
            if callable(reset_plan):
                reset_plan()
            pose = self.motion.robot_state().T_base_ee.copy()
            retreat = pose.copy()
            direction = -pose[:3, 2] if result.selected_grasp is not None else np.array([0., 0., 1.])
            if direction[2] < 0.3:
                stop_reason = "no_safe_retreat_direction"
                break
            retreat[:3, 3] += direction * self.config.retreat_m
            checker = getattr(self.motion, "is_observation_path_safe", None)
            finger = configured_finger_geometry(open_width_m=state.width_m if state.width_m is not None else 0.075)
            if (not callable(checker) or not checker(retreat, confirm.points_base, finger)
                    or self._cancelled or self.clock() >= deadline):
                stop_reason = "retreat_path_unverified"
                break
            if self.require_retry_authorization and previous_result is None:
                stop_reason = "awaiting_retry_authorization"
                self.pending_retry = (request, result)
                self._event("retry_available", reason=stop_reason)
                break
            self._event("safe_retreat", xyz_m=retreat[:3, 3].tolist())
            if not self.motion.move_to(retreat, speed_scale=self.config.speed_scale):
                self.motion.stop()
                stop_reason = "retreat_motion_failed"
                break
            # No opening in the supervisor: the next node opens only after
            # selecting/validating its new grasp, from the safe observation pose.
            relocated = self.observer.observe(target, confirm.center_base_m)
            if relocated is None:
                stop_reason = "target_lost_after_retreat"
                break
            target = replace(target, coarse_position_base_m=tuple(relocated.center_base_m))
            self._event("target_reacquired", center_base_m=relocated.center_base_m.tolist())
        if result is None:
            result = FineGraspResult(False, FineGraspPhase.FAILED, request.request_id,
                                     FailureCode.ABORTED if self._cancelled else FailureCode.SERVO_TIMEOUT,
                                     "Recovery did not start an attempt")
        if self._cancelled:
            result = replace(result, success=False, phase=FineGraspPhase.FAILED,
                             failure=FailureCode.ABORTED, message="Recovery cancelled")
        self._event("recovery_finished", reason=stop_reason)
        metrics = {**dict(result.metrics), "elapsed_s": self.clock() - started,
                   "recovery_attempts": len(self.attempt_results), "recovery_stop_reason": stop_reason,
                   "first_attempt_node_success": int(bool(self.attempt_results and self.attempt_results[0].success)),
                   "grasp_attempts": sum(int(r.metrics.get("close_commanded", 0)) for r in self.attempt_results),
                   "total_servo_steps": sum(int(r.metrics.get("servo_steps", 0)) for r in self.attempt_results)}
        metrics["total_active_view_moves"] = sum(int(r.metrics.get("active_view_moves", 0)) for r in self.attempt_results)
        metrics["total_model_calls"] = sum(int(r.metrics.get("model_calls", 0)) for r in self.attempt_results)
        metrics["total_model_ms"] = sum(float(r.metrics.get("model_total_ms", 0)) for r in self.attempt_results)
        self.active_node = None
        return replace(result, request_id=request.request_id, metrics=metrics)
