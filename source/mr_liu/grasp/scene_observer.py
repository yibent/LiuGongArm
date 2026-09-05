"""Target-specific external RGB-D evidence, without simulator object truth."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
import cv2
import numpy as np

from mr_liu.grasp.contracts import RGBDObservation, TargetSpec, VerificationResult
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import transform_points
from mr_liu.grasp.verification import VisionGripperVerifier


@dataclass(frozen=True)
class SceneTargetEvidence:
    observation: RGBDObservation
    points_base: np.ndarray
    center_base_m: np.ndarray


class SceneTargetObserver:
    """Track one instance by current metric ROI and a fixed appearance anchor.

    This geometric/color baseline can return unknown for similarly colored
    neighbors or poor depth. It is not an open-vocabulary detector or an oracle.
    The expected center is a search hypothesis; only measured pixels count.
    """
    def __init__(self, camera, *, table_height_m, recorder=None, clock=time.monotonic,
                 search_radius_m=0.08, color_tolerance=0.13, min_pixels=16,
                 max_age_s=0.25, require_robot_mask=False):
        self.camera, self.recorder, self.clock = camera, recorder, clock
        self.table_height_m = table_height_m
        self.search_radius_m, self.color_tolerance = search_radius_m, color_tolerance
        self.min_pixels, self.max_age_s = min_pixels, max_age_s
        self.require_robot_mask = require_robot_mask
        self.reset()

    def reset(self):
        self.object_id = None
        self.color = None
        self.latest = None
        self.last_sequence = -1
        self.last_diagnostic = {}

    def observe(self, target: TargetSpec, expected_center=None):
        if self.object_id != target.object_id:
            self.reset()
            self.object_id = target.object_id
        obs = self.camera.capture(target)
        if obs is None:
            self.last_diagnostic = {"scene_reason": "no_frame"}
            return None
        age = self.clock() - obs.timestamp_s
        if not -0.02 <= age <= self.max_age_s or obs.sequence <= self.last_sequence:
            self.last_diagnostic = {"scene_reason": "stale_frame"}
            return None
        self.last_sequence = obs.sequence
        center = expected_center
        if center is None:
            center = self.latest.center_base_m if self.latest is not None else target.coarse_position_base_m
        if center is None:
            return None
        center = np.asarray(center, dtype=float)
        valid = np.isfinite(obs.depth_m) & (obs.depth_m > 0)
        robot = obs.metadata.get("robot_self_mask")
        if robot is None and self.require_robot_mask:
            self.last_diagnostic = {"scene_reason": "robot_mask_unavailable"}
            return None
        if robot is not None:
            valid &= ~np.asarray(robot, dtype=bool)
        rows, cols = np.nonzero(valid)
        points = transform_points(obs.T_base_camera, deproject_depth(obs.depth_m, obs.intrinsics, valid))
        # A known support plane rejects table pixels; missing/reflective depth
        # does not become geometry. The XYZ ROI follows observations, not a
        # permanently fixed upstream hint.
        in_roi = ((np.linalg.norm(points[:, :2] - center[:2], axis=1) < self.search_radius_m)
                  & (np.abs(points[:, 2] - center[2]) < self.search_radius_m)
                  & (points[:, 2] > self.table_height_m + 0.004))
        colors = obs.rgb[rows, cols, :3].astype(float)
        colors /= np.maximum(colors.sum(axis=1, keepdims=True), 12)
        if self.color is not None:
            in_roi &= np.linalg.norm(colors - self.color, axis=1) <= self.color_tolerance
        mask = np.zeros(valid.shape, bool)
        mask[rows[in_roi], cols[in_roi]] = True
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        candidates = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < self.min_pixels:
                continue
            component = labels[rows, cols] == label
            cloud = points[component]
            local_center = np.median(cloud, axis=0)
            candidates.append((float(np.linalg.norm(local_center - center)), label, component, local_center))
        candidates.sort(key=lambda item: item[0])
        reason = "no_target_component"
        if candidates:
            # Comparable independent components are ambiguous, never choose one
            # solely because it has more pixels.
            if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.012:
                mask[:] = False
                reason = "ambiguous_components"
            else:
                _, label, component, local_center = candidates[0]
                mask = labels == label
                if self.color is None:
                    self.color = np.median(colors[component], axis=0)
                selected = replace(obs, target_mask=mask)
                evidence = SceneTargetEvidence(selected, points[component], local_center)
                self.latest = evidence
                self.last_diagnostic = {"scene_reason": "measured", "scene_pixels": int(mask.sum()),
                                        "scene_center_base_m": local_center.tolist()}
                if self.recorder:
                    self.recorder.record_segmentation(obs, mask, target)
                return evidence
        self.last_diagnostic = {"scene_reason": reason, "scene_pixels": 0}
        if self.recorder:
            self.recorder.record_segmentation(obs, None, target)
        return None


class SceneAssistedVerifier:
    """Supplement wrist verification with the *same external view* before/after.

    No arm motion is issued here. A missing external observation is unknown,
    never success. A clearly stationary target vetoes a wrist false positive.
    """
    def __init__(self, observer, motion, target, *, wrist=None):
        self.observer, self.motion, self.target = observer, motion, target
        self.wrist = wrist or VisionGripperVerifier()
        self.before = None
        self.ee_before = None
        self.last_evidence = None

    def before_close(self, observation):
        self.before = self.observer.observe(self.target)
        self.ee_before = self.motion.robot_state().T_base_ee.copy()

    def verify_closed(self, before, after, gripper):
        return self.wrist.verify_closed(before, after, gripper)

    def verify_lifted(self, before, after, gripper):
        primary = self.wrist.verify_lifted(before, after, gripper)
        if self.before is None or self.ee_before is None:
            return primary
        delta_ee = self.motion.robot_state().T_base_ee[:3, 3] - self.ee_before[:3, 3]
        expected = self.before.center_base_m + delta_ee
        evidence = self.observer.observe(self.target, expected)
        self.last_evidence = evidence
        metrics = {**dict(primary.metrics), "wrist_verification_success": int(primary.success)}
        if evidence is None:
            return replace(primary, metrics={**metrics, "scene_verification": "unknown"})
        measured = evidence.center_base_m - self.before.center_base_m
        error = float(np.linalg.norm(measured - delta_ee))
        nonempty = gripper.width_m is not None and gripper.width_m > self.wrist.min_held_width_m
        loaded = gripper.contact is True or gripper.stalled is True
        metrics.update(scene_target_lift_m=float(measured[2]), scene_follow_error_m=error)
        if measured[2] >= self.wrist.min_lift_m and error <= 0.015 and nonempty and loaded:
            return VerificationResult(True, 0.85, "external_target_follows_lift",
                                      {**metrics, "scene_verification": "confirmed"})
        if np.linalg.norm(measured) <= 0.01 and np.linalg.norm(delta_ee) >= self.wrist.min_lift_m:
            return VerificationResult(False, 0.0, "external_target_stayed_on_support",
                                      {**metrics, "scene_verification": "contradiction"})
        return replace(primary, metrics={**metrics, "scene_verification": "inconclusive"})
