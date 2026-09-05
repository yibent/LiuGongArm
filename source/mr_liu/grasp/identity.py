"""Image-mask identity tracking grounded in the current wrist camera pose."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree
from mr_liu.grasp.contracts import TargetSpec, RGBDObservation
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import invert_transform, transform_points


class GeometricIdentitySegmenter:
    """Coarse location initializes once; subsequent prompts use tracked geometry.

    The wrapped segmenter can be depth-based or a learned promptable segmenter.
    A tracking hint during lift is a search hypothesis, never verification.
    """
    def __init__(self, segmenter, max_displacement_m=0.035, color_tolerance=0.18):
        self.segmenter = segmenter
        self.max_displacement_m = max_displacement_m
        self.color_tolerance = color_tolerance
        self.reset()

    def reset(self):
        self.object_id = None
        self.points = None
        self.color = None
        self.last_diagnostic = {}

    def segment(self, observation: RGBDObservation, target: TargetSpec):
        if target.object_id != self.object_id:
            self.reset()
            self.object_id = target.object_id
        vision_target = target
        expected = None
        if self.points is not None:
            expected = np.asarray(observation.metadata.get("tracking_hint_base_m",
                                   np.median(self.points, axis=0)), dtype=float)
            vision_target = replace(target, coarse_position_base_m=tuple(expected))
            predicted = self.points + expected - np.median(self.points, axis=0)
            pc = transform_points(invert_transform(observation.T_base_camera), predicted)
            z = pc[:, 2]
            k = observation.intrinsics
            u = np.rint(k.fx * pc[:, 0] / np.maximum(z, 1e-6) + k.cx).astype(int)
            v = np.rint(k.fy * pc[:, 1] / np.maximum(z, 1e-6) + k.cy).astype(int)
            inside = (z > 0) & (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)
            u, v, z = u[inside], v[inside], z[inside]
            depth = observation.depth_m[v, u]
            rgb = observation.rgb[v, u, :3].astype(float)
            colors = rgb / np.maximum(rgb.sum(axis=1, keepdims=True), 12)
            supported = (np.isfinite(depth) & (np.abs(depth - z) <= 0.008)
                         & (np.linalg.norm(colors - self.color, axis=1) <= self.color_tolerance))
            if int(supported.sum()) < 8:
                self.last_diagnostic = {"identity": "uncertain", "reason": "no_visible_reference_surface"}
                return None
            us, vs = u[supported], v[supported]
            index = int(np.argmin((us - np.median(us))**2 + (vs - np.median(vs))**2))
            observation = replace(observation, metadata={**observation.metadata,
                "target_prompt_box": [int(u.min()), int(v.min()), int(u.max()), int(v.max())],
                "target_prompt_point": [int(us[index]), int(vs[index])]})
        mask = self.segmenter.segment(observation, vision_target)
        if mask is None:
            self.last_diagnostic = {"identity": "uncertain", "reason": "no_mask",
                                    **getattr(self.segmenter, "last_diagnostic", {})}
            return None
        mask = np.asarray(mask, bool).copy()
        robot_mask = observation.metadata.get("robot_self_mask")
        if robot_mask is not None:
            mask &= ~np.asarray(robot_mask, bool)
        valid = mask & np.isfinite(observation.depth_m) & (observation.depth_m > 0)
        if valid.sum() < 32:
            return None
        points = transform_points(observation.T_base_camera,
                                  deproject_depth(observation.depth_m, observation.intrinsics, valid))
        rgb = observation.rgb[valid, :3].astype(float)
        color = np.median(rgb / np.maximum(rgb.sum(axis=1, keepdims=True), 12), axis=0)
        if self.points is not None:
            predicted = self.points + expected - np.median(self.points, axis=0)
            distance, _ = cKDTree(predicted).query(points)
            support = float(np.mean(distance < self.max_displacement_m))
            color_error = float(np.linalg.norm(color - self.color))
            self.last_diagnostic = {"identity_support": support, "identity_color_error": color_error}
            if support < 0.50 or color_error > self.color_tolerance:
                self.last_diagnostic["identity"] = "uncertain"
                return None
        # Keep a fixed appearance anchor; recursive updates can learn the table.
        if self.color is None:
            self.color = color
        self.points = points[::max(1, len(points) // 2048)].copy()
        self.last_diagnostic["identity"] = "consistent"
        return mask


class RemotePromptSegmenter:
    """SAM 2 masks and optional open-vocabulary initialization via model RPC.

    Uses a geometric seed when available. Grounding DINO is requested only for
    initialization without a seed, never as a substitute for metric depth.
    """
    def __init__(self, transport, seed_segmenter):
        self.transport = transport
        self.seed_segmenter = seed_segmenter
        self.last_diagnostic = {}

    def segment(self, observation, target):
        seed_mask = self.seed_segmenter.segment(observation, target)
        box = None
        point = None
        if seed_mask is not None and seed_mask.any():
            rows, cols = np.nonzero(seed_mask)
            box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
            # Pick an actual foreground pixel nearest the median, not a point
            # in a mug/handle concavity.
            index = int(np.argmin((cols - np.median(cols))**2 + (rows - np.median(rows))**2))
            point = [int(cols[index]), int(rows[index])]
        response = self.transport._request({"action": "segment", "rgb": observation.rgb,
                                            "box": observation.metadata.get("target_prompt_box", box),
                                            "point": observation.metadata.get("target_prompt_point", point),
                                            "text": target.semantic_label or "object"})
        self.last_diagnostic = {k: v for k, v in response.items() if k != "mask"}
        mask = response.get("mask")
        return None if mask is None else np.asarray(mask, bool)
