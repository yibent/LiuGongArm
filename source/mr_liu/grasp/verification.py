"""Vision plus gripper-state grasp verification."""

from __future__ import annotations

import numpy as np

from mr_liu.grasp.contracts import GripperState, RGBDObservation, VerificationResult
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import transform_points


def _target_center_camera(observation: RGBDObservation) -> np.ndarray | None:
    if observation.target_mask is None:
        return None
    mask = (
        np.asarray(observation.target_mask, dtype=bool)
        & np.isfinite(observation.depth_m)
        & (observation.depth_m > 0)
    )
    points = deproject_depth(observation.depth_m, observation.intrinsics, mask)
    if len(points) < 8:
        return None
    return np.median(points, axis=0)


def _target_center_base(observation: RGBDObservation) -> np.ndarray | None:
    center_camera = _target_center_camera(observation)
    if center_camera is None:
        return None
    return transform_points(
        observation.T_base_camera, np.asarray([center_camera], dtype=np.float64)
    )[0]


class VisionGripperVerifier:
    def __init__(
        self,
        *,
        min_lift_m: float = 0.035,
        min_held_width_m: float = 0.002,
        empty_width_tolerance_m: float = 1e-5,
        max_follow_error_m: float = 0.025,
    ) -> None:
        self.min_lift_m = float(min_lift_m)
        self.min_held_width_m = float(min_held_width_m)
        self.empty_width_tolerance_m = float(empty_width_tolerance_m)
        self.max_follow_error_m = float(max_follow_error_m)

    def verify_closed(
        self, before: RGBDObservation, after: RGBDObservation, gripper: GripperState
    ) -> VerificationResult:
        del before
        visual = _target_center_base(after) is not None
        holding_signal = gripper.contact is True or gripper.stalled is True
        nonempty_width = gripper.width_m is None or gripper.width_m > self.min_held_width_m
        success = bool(nonempty_width and (holding_signal or visual))
        confidence = 0.85 if holding_signal and visual else 0.65 if holding_signal else 0.45 if visual else 0.0
        return VerificationResult(
            success=success,
            confidence=confidence,
            reason="contact_or_visual_target_present" if success else "no_contact_or_target_evidence",
            metrics={"visual_target": int(visual), "holding_signal": int(holding_signal)},
        )

    def verify_lifted(
        self, before: RGBDObservation, after: RGBDObservation, gripper: GripperState
    ) -> VerificationResult:
        before_center = _target_center_base(before)
        after_center = _target_center_base(after)
        holding_signal = gripper.contact is True or gripper.stalled is True
        not_fully_closed = (
            gripper.width_m is None
            or gripper.width_m > self.empty_width_tolerance_m
        )
        if before_center is None or after_center is None:
            success = bool(holding_signal and not_fully_closed)
            return VerificationResult(
                success=success,
                confidence=0.45 if success else 0.0,
                reason=(
                    "occluded_target_with_loaded_nonempty_gripper"
                    if success
                    else "target_missing_or_gripper_fully_closed_after_lift"
                ),
                metrics={
                    "holding_signal": int(holding_signal),
                    "gripper_not_fully_closed": int(not_fully_closed),
                },
            )
        target_delta = after_center - before_center
        dz = float(target_delta[2])
        follow_error_m: float | None = None
        if before.T_base_ee is not None and after.T_base_ee is not None:
            ee_delta = np.asarray(after.T_base_ee, dtype=np.float64)[:3, 3] - np.asarray(
                before.T_base_ee, dtype=np.float64
            )[:3, 3]
            follow_error_m = float(np.linalg.norm(target_delta - ee_delta))
        follows_wrist = follow_error_m is None or follow_error_m <= self.max_follow_error_m
        # A calibrated zero-width reading is useful negative evidence, but it
        # is not an absolute veto: thin tool handles can drive the inexpensive
        # SO-101 linkage to its nominal endpoint while still being physically
        # carried. In that case require visual 3-D motion to follow the wrist;
        # if vision is unavailable, a non-empty gripper remains mandatory.
        visual_follow_evidence = follow_error_m is not None and follows_wrist
        payload_evidence = not_fully_closed or visual_follow_evidence
        success = bool(
            dz >= self.min_lift_m
            and holding_signal
            and payload_evidence
            and follows_wrist
        )
        if not follows_wrist:
            reason = "target_did_not_follow_wrist"
        elif dz < self.min_lift_m:
            reason = "target_did_not_follow_lift"
        elif not holding_signal:
            reason = "no_gripper_contact_after_lift"
        else:
            reason = (
                "target_lifted_visual_follow_with_closed_gripper"
                if not not_fully_closed
                else "target_lifted_with_gripper"
            )
        metrics: dict[str, float | int] = {
            "target_lift_m": dz,
            "target_translation_m": float(np.linalg.norm(target_delta)),
            "holding_signal": int(holding_signal),
            "gripper_not_fully_closed": int(not_fully_closed),
        }
        if follow_error_m is not None:
            metrics["target_wrist_follow_error_m"] = follow_error_m
        return VerificationResult(
            success=success,
            confidence=(
                0.72
                if success and not not_fully_closed
                else 0.9
                if success and holding_signal
                else 0.7
                if success
                else 0.0
            ),
            reason=reason,
            metrics=metrics,
        )
