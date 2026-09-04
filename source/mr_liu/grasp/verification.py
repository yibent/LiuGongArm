"""Vision plus gripper-state grasp verification."""

from __future__ import annotations

import numpy as np

from mr_liu.grasp.contracts import GripperState, RGBDObservation, VerificationResult
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import transform_points


def _target_center_base(observation: RGBDObservation) -> np.ndarray | None:
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
    return np.median(transform_points(observation.T_base_camera, points), axis=0)


class VisionGripperVerifier:
    def __init__(self, *, min_lift_m: float = 0.035, min_held_width_m: float = 0.002) -> None:
        self.min_lift_m = float(min_lift_m)
        self.min_held_width_m = float(min_held_width_m)

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
        if before_center is None or after_center is None:
            success = bool(gripper.contact is True or gripper.stalled is True)
            return VerificationResult(
                success=success,
                confidence=0.55 if success else 0.0,
                reason="gripper_only_no_visual_center" if success else "target_missing_after_lift",
            )
        dz = float(after_center[2] - before_center[2])
        holding_signal = gripper.contact is True or gripper.stalled is True
        success = dz >= self.min_lift_m and (holding_signal or gripper.width_m is None)
        return VerificationResult(
            success=success,
            confidence=0.9 if success and holding_signal else 0.7 if success else 0.0,
            reason="target_lifted_with_gripper" if success else "target_did_not_follow_lift",
            metrics={"target_lift_m": dz, "holding_signal": int(holding_signal)},
        )
