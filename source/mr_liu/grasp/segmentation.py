"""Local wrist-view segmentation adapters."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from mr_liu.grasp.contracts import RGBDObservation, TargetSegmenter, TargetSpec
from mr_liu.grasp.transforms import invert_transform, transform_points


class ObservationMaskSegmenter:
    """Use a synchronized target mask supplied by a camera/simulator adapter."""

    def segment(self, observation: RGBDObservation, target: TargetSpec) -> np.ndarray | None:
        del target
        if observation.target_mask is None:
            return None
        return np.asarray(observation.target_mask, dtype=bool).copy()


class SeededDepthSegmenter:
    """Segment around the upstream coarse target using the *current* depth.

    This is a local geometric fallback for unknown objects. It is not a fixed
    overhead-camera projection: the coarse base point is projected through the
    latest eye-in-hand pose on every observation, and connected depth growing is
    then performed in the wrist image.
    """

    def __init__(
        self,
        *,
        depth_tolerance_m: float = 0.025,
        seed_window_px: int = 7,
        max_radius_px: int = 140,
    ) -> None:
        self.depth_tolerance_m = float(depth_tolerance_m)
        self.seed_window_px = int(seed_window_px)
        self.max_radius_px = int(max_radius_px)

    def segment(self, observation: RGBDObservation, target: TargetSpec) -> np.ndarray | None:
        if observation.target_mask is not None:
            return np.asarray(observation.target_mask, dtype=bool).copy()
        if target.coarse_position_base_m is None:
            return None
        point_base = np.asarray(target.coarse_position_base_m, dtype=np.float64).reshape(1, 3)
        point_camera = transform_points(invert_transform(observation.T_base_camera), point_base)[0]
        if point_camera[2] <= 0:
            return None
        K = observation.intrinsics
        u = int(round(K.fx * point_camera[0] / point_camera[2] + K.cx))
        v = int(round(K.fy * point_camera[1] / point_camera[2] + K.cy))
        if not (0 <= u < K.width and 0 <= v < K.height):
            return None

        radius = max(self.seed_window_px // 2, 1)
        y0, y1 = max(v - radius, 0), min(v + radius + 1, K.height)
        x0, x1 = max(u - radius, 0), min(u + radius + 1, K.width)
        patch = np.asarray(observation.depth_m[y0:y1, x0:x1], dtype=np.float64)
        valid_patch = patch[np.isfinite(patch) & (patch > 0)]
        seed_depth = float(np.median(valid_patch)) if valid_patch.size else float(point_camera[2])
        allowed = np.isfinite(observation.depth_m) & (
            np.abs(observation.depth_m - seed_depth) <= self.depth_tolerance_m
        )
        yy, xx = np.ogrid[: K.height, : K.width]
        allowed &= (xx - u) ** 2 + (yy - v) ** 2 <= self.max_radius_px**2
        return _connected_component_from_nearest_seed(allowed, u, v, radius)


class AppearanceDepthTrackerSegmenter:
    """Bootstrap from current-pose depth, then track the target's appearance.

    Camera-pose reprojection supplies only the first local seed. Subsequent
    frames use the target appearance learned in that wrist view, which remains
    useful during rapid camera motion and tolerates a renderer/pose timestamp
    offset better than repeatedly trusting the upstream coarse point.
    """

    def __init__(
        self,
        seed_segmenter: TargetSegmenter | None = None,
        *,
        chromaticity_tolerance: float = 0.16,
        min_component_pixels: int = 80,
        morphology_kernel_px: int = 5,
    ) -> None:
        self.seed_segmenter = seed_segmenter or SeededDepthSegmenter()
        self.chromaticity_tolerance = float(chromaticity_tolerance)
        self.min_component_pixels = int(min_component_pixels)
        self.kernel = np.ones((morphology_kernel_px, morphology_kernel_px), dtype=np.uint8)
        self._object_id: str | None = None
        self._reference_chromaticity: np.ndarray | None = None
        self._last_center_uv: np.ndarray | None = None

    def reset(self) -> None:
        self._object_id = None
        self._reference_chromaticity = None
        self._last_center_uv = None

    def segment(self, observation: RGBDObservation, target: TargetSpec) -> np.ndarray | None:
        if target.object_id != self._object_id:
            self.reset()
            self._object_id = target.object_id
        rgb = np.asarray(observation.rgb[:, :, :3], dtype=np.float32)
        chromaticity = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 12.0)
        if self._reference_chromaticity is None:
            initial = self.seed_segmenter.segment(observation, target)
            if initial is None or int(np.asarray(initial, dtype=bool).sum()) < self.min_component_pixels:
                return None
            mask = np.asarray(initial, dtype=bool)
            self._update_reference(chromaticity, mask, replace=True)
            return mask

        distance = np.linalg.norm(chromaticity - self._reference_chromaticity.reshape(1, 1, 3), axis=2)
        valid_depth = np.isfinite(observation.depth_m) & (observation.depth_m > 0)
        allowed = (distance <= self.chromaticity_tolerance) & valid_depth & (rgb.sum(axis=2) > 30.0)
        cleaned = cv2.morphologyEx(allowed.astype(np.uint8), cv2.MORPH_OPEN, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        candidates: list[tuple[float, int]] = []
        diagonal = float(np.hypot(observation.intrinsics.width, observation.intrinsics.height))
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_pixels:
                continue
            center = centers[label]
            travel = 0.0 if self._last_center_uv is None else float(np.linalg.norm(center - self._last_center_uv))
            # Area supports large close-range targets; the soft travel term
            # preserves identity without preventing large eye-in-hand motion.
            score = np.log1p(area) - 1.5 * travel / max(diagonal, 1.0)
            candidates.append((score, label))
        if not candidates:
            return None
        selected_label = max(candidates)[1]
        mask = labels == selected_label
        self._update_reference(chromaticity, mask, replace=False)
        return mask

    def _update_reference(self, chromaticity: np.ndarray, mask: np.ndarray, *, replace: bool) -> None:
        values = chromaticity[mask]
        reference = np.median(values, axis=0)
        if replace or self._reference_chromaticity is None:
            self._reference_chromaticity = reference
        else:
            self._reference_chromaticity = 0.9 * self._reference_chromaticity + 0.1 * reference
        rows, columns = np.nonzero(mask)
        self._last_center_uv = np.asarray([np.median(columns), np.median(rows)], dtype=np.float64)


def _connected_component_from_nearest_seed(
    allowed: np.ndarray, seed_u: int, seed_v: int, search_radius: int
) -> np.ndarray | None:
    height, width = allowed.shape
    start: tuple[int, int] | None = None
    for radius in range(search_radius + 1):
        for y in range(max(0, seed_v - radius), min(height, seed_v + radius + 1)):
            for x in range(max(0, seed_u - radius), min(width, seed_u + radius + 1)):
                if allowed[y, x]:
                    start = (y, x)
                    break
            if start is not None:
                break
        if start is not None:
            break
    if start is None:
        return None

    component = np.zeros_like(allowed, dtype=bool)
    queue: deque[tuple[int, int]] = deque([start])
    component[start] = True
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and allowed[ny, nx] and not component[ny, nx]:
                component[ny, nx] = True
                queue.append((ny, nx))
    return component
