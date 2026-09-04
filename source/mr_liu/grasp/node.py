"""Model-independent eye-in-hand FineGrasp closed loop."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Callable

import numpy as np

from mr_liu.grasp.contracts import (
    FailureCode,
    FineGraspPhase,
    FineGraspRequest,
    FineGraspResult,
    GraspBackend,
    GraspVerifier,
    GripperController,
    MotionExecutor,
    RGBDObservation,
    SelectedGrasp,
    TargetSegmenter,
    WristCamera,
)
from mr_liu.grasp.geometry import ObjectGeometry, extract_object_geometry
from mr_liu.grasp.policy import ExecutionPolicy, policy_for_target
from mr_liu.grasp.selection import SelectionResult, select_grasp
from mr_liu.grasp.servo import PoseServo
from mr_liu.grasp.settings import FineGraspSettings
from mr_liu.grasp.transforms import make_transform, pose_error


TraceCallback = Callable[[dict[str, object]], None]


class GeneralGraspNode:
    """Stable BusAgent node; concrete vision/model/robot backends are injected.

    Every correction is computed from a new wrist-camera observation. A grasp
    pose is never transformed with a camera pose from a different observation.
    """

    def __init__(
        self,
        *,
        camera: WristCamera,
        segmenter: TargetSegmenter,
        backend: GraspBackend,
        motion: MotionExecutor,
        gripper: GripperController,
        verifier: GraspVerifier,
        settings: FineGraspSettings,
        clock: Callable[[], float] = time.monotonic,
        trace: TraceCallback | None = None,
    ) -> None:
        self.camera = camera
        self.segmenter = segmenter
        self.backend = backend
        self.motion = motion
        self.gripper = gripper
        self.verifier = verifier
        self.settings = settings
        self.clock = clock
        self.trace = trace
        self._cancelled = False
        self._last_sequence = -1

    def cancel(self) -> None:
        self._cancelled = True
        self.motion.stop()

    def execute(self, request: FineGraspRequest) -> FineGraspResult:
        started = self.clock()
        deadline = started + min(
            request.timeout_s if request.timeout_s is not None else self.settings.loop.max_total_s,
            self.settings.loop.max_total_s,
        )
        metrics: dict[str, float | int | str] = {
            "backend": self.backend.name,
            "observations": 0,
            "replans": 0,
            "servo_steps": 0,
            "candidates": 0,
        }
        self._cancelled = False
        self._last_sequence = -1
        policy = policy_for_target(
            request.target,
            default_force_n=self.settings.gripper.default_force_n,
            default_speed_mps=self.settings.gripper.default_speed_mps,
            table_clearance_m=self.settings.table_clearance_m,
        )
        selected: SelectedGrasp | None = None

        try:
            for replan_index in range(self.settings.loop.max_replans + 1):
                metrics["replans"] = replan_index
                terminal = self._terminal_check(request, deadline, metrics, selected)
                if terminal is not None:
                    return terminal
                observed = self._observe(request, metrics, allow_same_sequence=False)
                if isinstance(observed, FineGraspResult):
                    return self._with_elapsed(observed, started, metrics)
                observation, geometry = observed
                self._emit(FineGraspPhase.GENERATE, observation_sequence=observation.sequence)
                generated_at = self.clock()
                candidates = list(
                    self.backend.generate(observation, geometry.points_camera, request.target)
                )
                metrics["model_latency_ms"] = round((self.clock() - generated_at) * 1000.0, 3)
                metrics["candidates"] = int(metrics["candidates"]) + len(candidates)
                selection = select_grasp(
                    candidates,
                    observation,
                    geometry,
                    self.motion,
                    policy,
                    self.settings.selection,
                )
                for reason, count in selection.rejected.items():
                    metrics[f"rejected_{reason}"] = int(metrics.get(f"rejected_{reason}", 0)) + count
                if selection.grasp is None:
                    return self._failure(
                        request,
                        self._selection_failure(selection),
                        "No candidate satisfied gripper, clearance, collision and IK constraints",
                        metrics,
                        started,
                    )
                selected = selection.grasp
                self._emit(
                    FineGraspPhase.PREGRASP,
                    score=selected.score,
                    width_m=selected.width_m,
                    backend=selected.backend,
                )
                if request.dry_run:
                    metrics["elapsed_s"] = self.clock() - started
                    return FineGraspResult(
                        success=True,
                        phase=FineGraspPhase.SUCCEEDED,
                        request_id=request.request_id,
                        message="Dry-run grasp proposal is feasible",
                        selected_grasp=selected,
                        metrics=metrics,
                    )

                open_width = min(
                    self.settings.gripper.max_width_m, selected.width_m + 0.006
                )
                if not self.gripper.open(open_width, speed_mps=policy.close_speed_mps):
                    return self._failure(
                        request,
                        FailureCode.GRIPPER_FAILURE,
                        "Gripper failed to open for pregrasp",
                        metrics,
                        started,
                        selected,
                    )
                if not self.motion.move_to(selected.T_base_pregrasp, speed_scale=policy.motion_speed_scale):
                    return self._failure(
                        request,
                        FailureCode.IK_UNREACHABLE,
                        "Motion backend failed to reach pregrasp",
                        metrics,
                        started,
                        selected,
                    )

                servo = PoseServo(self.settings.servo)
                previous_center = geometry.T_base_object[:3, 3].copy()
                replan_required = False
                before_close: RGBDObservation | None = None
                final_desired = selected.T_base_grasp
                for servo_index in range(self.settings.loop.servo_max_steps):
                    terminal = self._terminal_check(request, deadline, metrics, selected)
                    if terminal is not None:
                        return terminal
                    latest = self._observe(request, metrics, allow_same_sequence=False)
                    if isinstance(latest, FineGraspResult):
                        return self._with_elapsed(latest, started, metrics)
                    current_observation, current_geometry = latest
                    center = current_geometry.T_base_object[:3, 3]
                    jump = float(np.linalg.norm(center - previous_center))
                    metrics["max_observed_centroid_jump_m"] = max(
                        float(metrics.get("max_observed_centroid_jump_m", 0.0)), jump
                    )
                    if jump > self.settings.observation.max_centroid_jump_m:
                        self._emit(FineGraspPhase.OBSERVE, event="target_jump", jump_m=jump)
                        replan_required = True
                        break
                    previous_center = center.copy()
                    final_desired = current_geometry.T_base_object @ selected.T_object_grasp
                    translation_from_plan, rotation_from_plan = pose_error(
                        selected.T_base_grasp, final_desired
                    )
                    if (
                        self.clock() - generated_at >= self.settings.loop.model_period_s
                        and (
                            np.linalg.norm(translation_from_plan)
                            > self.settings.loop.model_replan_translation_m
                            or rotation_from_plan > self.settings.loop.model_replan_rotation_rad
                        )
                    ):
                        self._emit(
                            FineGraspPhase.GENERATE,
                            event="model_replan",
                            translation_m=float(np.linalg.norm(translation_from_plan)),
                            rotation_deg=math.degrees(rotation_from_plan),
                        )
                        replan_required = True
                        break
                    state = self.motion.robot_state()
                    update = servo.update(state.T_base_ee, final_desired)
                    metrics["servo_steps"] = int(metrics["servo_steps"]) + 1
                    metrics["final_translation_error_m"] = update.translation_error_m
                    metrics["final_rotation_error_deg"] = math.degrees(update.rotation_error_rad)
                    self._emit(
                        FineGraspPhase.SERVO,
                        step=servo_index,
                        translation_error_m=update.translation_error_m,
                        rotation_error_deg=math.degrees(update.rotation_error_rad),
                        observation_sequence=current_observation.sequence,
                    )
                    if update.aligned:
                        before_close = current_observation
                        break
                    if not self.motion.is_reachable(update.command_pose):
                        replan_required = True
                        break
                    if not self.motion.is_collision_free(
                        update.command_pose,
                        margin_m=self.settings.selection.collision_margin_m,
                        allow_target_contact=True,
                    ):
                        return self._failure(
                            request,
                            FailureCode.COLLISION,
                            "Servo command violates collision constraints",
                            metrics,
                            started,
                            selected,
                        )
                    if not self.motion.servo_to(
                        update.command_pose, speed_scale=policy.motion_speed_scale
                    ):
                        replan_required = True
                        break

                if replan_required:
                    continue
                if before_close is None:
                    return self._failure(
                        request,
                        FailureCode.SERVO_TIMEOUT,
                        "Visual servo did not converge",
                        metrics,
                        started,
                        selected,
                    )

                selected = replace(
                    selected,
                    T_base_grasp=final_desired,
                    T_base_pregrasp=final_desired
                    @ make_transform(
                        translation=np.asarray(
                            [0.0, 0.0, -self.settings.selection.pregrasp_distance_m]
                        )
                    ),
                )
                self._emit(FineGraspPhase.CLOSE, force_n=policy.force_n)
                if not self.gripper.close(0.0, force_n=policy.force_n, speed_mps=policy.close_speed_mps):
                    return self._failure(
                        request,
                        FailureCode.GRIPPER_FAILURE,
                        "Gripper close command failed",
                        metrics,
                        started,
                        selected,
                    )
                after_close_result = self._observe(request, metrics, allow_same_sequence=False)
                if isinstance(after_close_result, FineGraspResult):
                    return self._with_elapsed(after_close_result, started, metrics)
                after_close, _ = after_close_result
                close_check = self.verifier.verify_closed(
                    before_close, after_close, self.gripper.state()
                )
                metrics["close_confidence"] = close_check.confidence
                if not close_check.success:
                    return self._failure(
                        request,
                        FailureCode.GRASP_NOT_DETECTED,
                        close_check.reason,
                        {**metrics, **dict(close_check.metrics)},
                        started,
                        selected,
                    )

                self._emit(FineGraspPhase.LIFT, lift_m=self.settings.loop.lift_distance_m)
                state = self.motion.robot_state()
                lift_pose = make_transform(
                    state.T_base_ee[:3, :3],
                    state.T_base_ee[:3, 3]
                    + np.asarray([0.0, 0.0, self.settings.loop.lift_distance_m]),
                )
                if not self.motion.move_to(lift_pose, speed_scale=min(policy.motion_speed_scale, 0.4)):
                    return self._failure(
                        request,
                        FailureCode.IK_UNREACHABLE,
                        "Lift motion failed",
                        metrics,
                        started,
                        selected,
                    )
                after_lift_result = self._observe(request, metrics, allow_same_sequence=False)
                if isinstance(after_lift_result, FineGraspResult):
                    return self._with_elapsed(after_lift_result, started, metrics)
                after_lift, _ = after_lift_result
                lift_check = self.verifier.verify_lifted(
                    after_close, after_lift, self.gripper.state()
                )
                metrics["lift_confidence"] = lift_check.confidence
                metrics.update(lift_check.metrics)
                if not lift_check.success:
                    return self._failure(
                        request,
                        FailureCode.LIFT_VERIFICATION_FAILED,
                        lift_check.reason,
                        metrics,
                        started,
                        selected,
                    )
                metrics["elapsed_s"] = self.clock() - started
                self._emit(FineGraspPhase.SUCCEEDED)
                return FineGraspResult(
                    success=True,
                    phase=FineGraspPhase.SUCCEEDED,
                    request_id=request.request_id,
                    message="Grasp, lift and verification succeeded",
                    selected_grasp=selected,
                    metrics=metrics,
                )

            return self._failure(
                request,
                FailureCode.TARGET_MOVED,
                "Maximum grasp replans exceeded",
                metrics,
                started,
                selected,
            )
        except Exception as exc:  # noqa: BLE001 - node returns classified failures
            self.motion.stop()
            return self._failure(
                request,
                FailureCode.INTERNAL_ERROR,
                f"FineGrasp internal error: {type(exc).__name__}: {exc}",
                metrics,
                started,
                selected,
            )

    def _observe(
        self,
        request: FineGraspRequest,
        metrics: dict[str, float | int | str],
        *,
        allow_same_sequence: bool,
    ) -> tuple[RGBDObservation, ObjectGeometry] | FineGraspResult:
        last_failure = FailureCode.NO_OBSERVATION
        last_message = "Wrist camera returned no observation"
        for _ in range(self.settings.loop.max_observation_failures):
            self._emit(FineGraspPhase.OBSERVE)
            observation = self.camera.capture(request.target)
            if observation is None:
                continue
            metrics["observations"] = int(metrics["observations"]) + 1
            age = self.clock() - observation.timestamp_s
            metrics["max_observation_age_ms"] = max(
                float(metrics.get("max_observation_age_ms", 0.0)), age * 1000.0
            )
            if age > self.settings.observation.max_age_s:
                last_failure = FailureCode.STALE_OBSERVATION
                last_message = f"Wrist observation is stale by {age:.3f}s"
                continue
            if not allow_same_sequence and observation.sequence <= self._last_sequence:
                last_failure = FailureCode.STALE_OBSERVATION
                last_message = "Wrist camera did not provide a newer frame"
                continue
            mask = self.segmenter.segment(observation, request.target)
            if mask is None or int(np.asarray(mask, dtype=bool).sum()) < self.settings.observation.min_mask_pixels:
                last_failure = FailureCode.TARGET_NOT_VISIBLE
                last_message = "Target is not visible in the latest wrist frame"
                continue
            synchronized = replace(observation, target_mask=np.asarray(mask, dtype=bool))
            try:
                geometry = extract_object_geometry(
                    synchronized,
                    synchronized.target_mask,
                    min_depth_m=self.settings.observation.depth_min_m,
                    max_depth_m=self.settings.observation.depth_max_m,
                    mad_scale=self.settings.observation.depth_outlier_mad_scale,
                    self_mask_bottom_fraction=self.settings.observation.self_mask_bottom_fraction,
                )
            except ValueError as exc:
                last_failure = FailureCode.INSUFFICIENT_DEPTH
                last_message = str(exc)
                continue
            if len(geometry.points_camera) < self.settings.observation.min_valid_depth_pixels:
                last_failure = FailureCode.INSUFFICIENT_DEPTH
                last_message = "Too few valid target depth samples"
                continue
            if geometry.valid_depth_ratio < self.settings.observation.min_valid_depth_ratio:
                last_failure = FailureCode.INSUFFICIENT_DEPTH
                last_message = (
                    f"Target valid-depth ratio {geometry.valid_depth_ratio:.3f} is below threshold"
                )
                continue
            self._last_sequence = observation.sequence
            return synchronized, geometry
        return FineGraspResult(
            success=False,
            phase=FineGraspPhase.FAILED,
            request_id=request.request_id,
            failure=last_failure,
            message=last_message,
            metrics=metrics,
        )

    def _terminal_check(
        self,
        request: FineGraspRequest,
        deadline: float,
        metrics: dict[str, float | int | str],
        selected: SelectedGrasp | None,
    ) -> FineGraspResult | None:
        if self._cancelled:
            return FineGraspResult(
                success=False,
                phase=FineGraspPhase.FAILED,
                request_id=request.request_id,
                failure=FailureCode.ABORTED,
                message="FineGrasp was cancelled",
                selected_grasp=selected,
                metrics=metrics,
            )
        if self.clock() > deadline:
            self.motion.stop()
            return FineGraspResult(
                success=False,
                phase=FineGraspPhase.FAILED,
                request_id=request.request_id,
                failure=FailureCode.SERVO_TIMEOUT,
                message="FineGrasp deadline exceeded",
                selected_grasp=selected,
                metrics=metrics,
            )
        return None

    @staticmethod
    def _selection_failure(selection: SelectionResult) -> FailureCode:
        if not selection.rejected:
            return FailureCode.NO_GRASP_CANDIDATES
        dominant = max(selection.rejected, key=selection.rejected.get)  # type: ignore[arg-type]
        if dominant == "gripper_width":
            return FailureCode.GRIPPER_WIDTH_INFEASIBLE
        if dominant == "table_clearance":
            return FailureCode.TABLE_CLEARANCE
        if "collision" in dominant:
            return FailureCode.COLLISION
        if dominant == "ik_unreachable":
            return FailureCode.IK_UNREACHABLE
        return FailureCode.NO_GRASP_CANDIDATES

    def _failure(
        self,
        request: FineGraspRequest,
        failure: FailureCode,
        message: str,
        metrics: dict[str, float | int | str],
        started: float,
        selected: SelectedGrasp | None = None,
    ) -> FineGraspResult:
        self.motion.stop()
        metrics["elapsed_s"] = self.clock() - started
        self._emit(FineGraspPhase.FAILED, failure=failure.value, message=message)
        return FineGraspResult(
            success=False,
            phase=FineGraspPhase.FAILED,
            request_id=request.request_id,
            failure=failure,
            message=message,
            selected_grasp=selected,
            metrics=metrics,
        )

    def _with_elapsed(
        self,
        result: FineGraspResult,
        started: float,
        metrics: dict[str, float | int | str],
    ) -> FineGraspResult:
        metrics["elapsed_s"] = self.clock() - started
        return replace(result, metrics=metrics)

    def _emit(self, phase: FineGraspPhase, **fields: object) -> None:
        if self.trace is not None:
            self.trace({"timestamp_s": self.clock(), "phase": phase.value, **fields})
