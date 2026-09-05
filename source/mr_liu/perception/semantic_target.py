"""Label-conditioned metric target from real RGB-D pixels, never object truth."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import time

import cv2
import numpy as np
import requests

from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import transform_points
from mr_liu.perception.optical_flow import OpticalFlowTracker


class TargetObservationError(RuntimeError):
    pass


@dataclass
class MetricTarget:
    observation: object
    mask: np.ndarray
    points_base: np.ndarray
    center_base_m: np.ndarray
    top_z_m: float
    color: np.ndarray


class LocalSemanticLocator:
    def __init__(self, port=5570, mode="florence_yoloe", timeout_s=45):
        self.url = f"http://127.0.0.1:{int(port)}/locate"
        self.mode, self.timeout_s = mode, timeout_s
        self.session = requests.Session()
        self.session.trust_env = False

    def locate(self, observation, label):
        rgb = observation.rgb.copy()
        robot = observation.metadata.get("robot_self_mask")
        if robot is not None:
            rgb[np.asarray(robot, bool)] = 0
        ok, data = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise TargetObservationError("image_encode_failed")
        response = self.session.post(self.url, json={"sequence": observation.sequence, "label": label,
            "mode": self.mode, "jpeg": base64.b64encode(data).decode()}, timeout=self.timeout_s)
        response.raise_for_status()
        result = response.json()
        if result.get("sequence") != observation.sequence or result.get("protocol") != 1:
            raise TargetObservationError("detector_frame_mismatch")
        return result


def metric_component(observation, roi, table_height_m, *, min_pixels=10):
    valid = (np.asarray(roi, bool) & np.isfinite(observation.depth_m)
             & (observation.depth_m > 0.02) & (observation.depth_m < 3.0))
    robot = observation.metadata.get("robot_self_mask")
    if robot is not None:
        valid &= ~np.asarray(robot, bool)
    rows, cols = np.nonzero(valid)
    points = transform_points(observation.T_base_camera, deproject_depth(observation.depth_m, observation.intrinsics, valid))
    above = (points[:, 2] > table_height_m + 0.004) & (points[:, 2] < table_height_m + 0.4)
    mask = np.zeros(valid.shape, np.uint8)
    mask[rows[above], cols[above]] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = sorted((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
                        if stats[i, cv2.CC_STAT_AREA] >= min_pixels)
    if not components:
        raise TargetObservationError("insufficient_target_depth")
    if len(components) > 1 and components[-2][0] > 0.5 * components[-1][0]:
        raise TargetObservationError("ambiguous_depth_components")
    selected = labels == components[-1][1]
    cloud = points[selected[rows, cols]]
    top = float(np.percentile(cloud[:, 2], 95))
    center = np.median(cloud, axis=0)
    center[2] = (table_height_m + top) * 0.5
    colors = observation.rgb[selected].astype(float)
    colors /= np.maximum(colors.sum(axis=1, keepdims=True), 12)
    return MetricTarget(observation, selected, cloud, center, top, np.median(colors, axis=0))


class SemanticFlowTarget:
    def __init__(self, camera, locator, target, table_height_m, *, clock=time.monotonic, trace=lambda event: None, recorder=None):
        self.camera, self.locator, self.target = camera, locator, target
        self.table_height_m, self.clock, self.trace = table_height_m, clock, trace
        self.recorder = recorder
        self.flow = OpticalFlowTracker()
        self.latest = None
        self.anchor_color = None
        self.last_sequence = -1
        self.find_count = self.flow_count = 0
        self.handoff_after_s = None
        self.last_wrist_sequence = -1

    def capture(self):
        obs = self.camera.capture(self.target)
        if obs is None:
            raise TargetObservationError("camera_unavailable")
        if obs.sequence <= self.last_sequence or not -0.02 <= self.clock() - obs.timestamp_s <= 0.35:
            raise TargetObservationError("stale_rgbd")
        self.last_sequence = obs.sequence
        return obs

    def find(self):
        obs = self.capture()
        result = self.locator.locate(obs, self.target.semantic_label)
        self.find_count += 1
        self.trace({"phase": "find", "event": "semantic_find", "sequence": obs.sequence, "response": result})
        choices = []
        for item in result["boxes"]:
            box = np.asarray(item["xyxy"], float)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            x1, y1, x2, y2 = box.astype(int)
            roi = np.zeros(obs.depth_m.shape, bool)
            roi[max(0, y1):max(0, min(roi.shape[0], y2)), max(0, x1):max(0, min(roi.shape[1], x2))] = True
            try:
                candidate = metric_component(obs, roi, self.table_height_m)
            except TargetObservationError:
                continue
            if self.anchor_color is not None:
                if np.linalg.norm(candidate.color - self.anchor_color) > 0.18:
                    continue
                if np.linalg.norm(candidate.center_base_m - self.latest.center_base_m) > 0.08:
                    continue
            if not any(np.linalg.norm(c.center_base_m - candidate.center_base_m) < 0.012 for c in choices):
                choices.append(candidate)
        if len(choices) != 1:
            raise TargetObservationError("semantic_target_not_found" if not choices else "ambiguous_label_instances")
        selected = choices[0]
        if self.recorder:
            self.recorder.record_segmentation(obs, selected.mask, self.target)
        if self.anchor_color is None:
            self.anchor_color = selected.color.copy()
        if not self.flow.initialize(obs.rgb, selected.mask):
            raise TargetObservationError("insufficient_optical_flow_features")
        self.latest = selected
        # Model latency makes its original frame stale. Fresh LK + depth must
        # succeed before this FIND can authorize a movement.
        return self.observe()

    def observe(self):
        obs = self.capture()
        processing_started = time.perf_counter()
        mask = self.flow.update(obs.rgb)
        if mask is None:
            raise TargetObservationError(self.flow.diagnostic.get("flow_reason", "flow_lost"))
        evidence = metric_component(obs, cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)), self.table_height_m)
        if np.linalg.norm(evidence.color - self.anchor_color) > 0.18:
            raise TargetObservationError("appearance_identity_changed")
        self.flow_count += 1
        self.latest = evidence
        processing_ms = 1000 * (time.perf_counter() - processing_started)
        if self.recorder and self.flow_count % 5 == 0:
            self.recorder.record_segmentation(obs, evidence.mask, self.target)
        self.trace({"phase": "track", "event": "optical_flow", "sequence": obs.sequence,
                    "center_base_m": evidence.center_base_m.tolist(), "top_z_m": evidence.top_z_m,
                    "timestamp_s": obs.timestamp_s,
                    "flow_and_depth_ms": processing_ms,
                    **self.flow.diagnostic})
        return evidence

    def validate_wrist_handoff(self, observation, mask):
        """Cross-view identity/geometry gate, not merely a nonempty seed mask."""
        if (observation is None or self.latest is None or self.handoff_after_s is None
                or observation.timestamp_s <= self.handoff_after_s
                or observation.sequence <= self.last_wrist_sequence
                or not -0.02 <= self.clock() - observation.timestamp_s <= 0.35):
            raise TargetObservationError("stale_wrist_handoff")
        if mask is None or np.count_nonzero(mask) < 80:
            raise TargetObservationError("wrist_handoff_insufficient_mask")
        evidence = metric_component(observation, mask, self.table_height_m, min_pixels=80)
        distance = float(np.linalg.norm(evidence.center_base_m - self.latest.center_base_m))
        color_error = float(np.linalg.norm(evidence.color - self.anchor_color))
        # Different surfaces are visible from the two cameras. This is a
        # coarse association gate; FineGrasp still refines the wrist geometry.
        if distance > 0.03 or color_error > 0.18:
            raise TargetObservationError("wrist_handoff_identity_mismatch")
        self.last_wrist_sequence = observation.sequence
        return {"center_disagreement_m": distance, "color_disagreement": color_error,
                "mask_pixels": int(evidence.mask.sum()), "sequence": observation.sequence}
