"""Local wrist-view segmentation adapters."""

from __future__ import annotations

import math
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
        chromaticity_tolerance: float = 0.18,
        seed_window_px: int = 7,
        max_radius_px: int = 140,
        coarse_search_radius_px: int = 48,
    ) -> None:
        self.depth_tolerance_m = float(depth_tolerance_m)
        self.chromaticity_tolerance = float(chromaticity_tolerance)
        self.seed_window_px = int(seed_window_px)
        self.max_radius_px = int(max_radius_px)
        self.coarse_search_radius_px = int(coarse_search_radius_px)

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
        depth_tolerance = min(
            self.depth_tolerance_m,
            0.004 if bool(target.properties.thin) else self.depth_tolerance_m,
        )
        # The coarse 3-D point may project into a hole or concavity (mug mouth,
        # open wrench jaw, key bow). A tiny patch then measures the table and
        # grows a large circular background mask. Search a bounded wrist-local
        # area and choose its robust near-depth mode; foreground object surfaces
        # are closer to the wrist than their support plane.
        search_radius = min(self.max_radius_px, max(self.coarse_search_radius_px, radius))
        sy0, sy1 = max(v - search_radius, 0), min(v + search_radius + 1, K.height)
        sx0, sx1 = max(u - search_radius, 0), min(u + search_radius + 1, K.width)
        search_depth = np.asarray(
            observation.depth_m[sy0:sy1, sx0:sx1], dtype=np.float64
        )
        plausible = (
            np.isfinite(search_depth)
            & (search_depth > 0)
            & (np.abs(search_depth - float(point_camera[2])) <= 0.060)
        )
        plausible_depth = search_depth[plausible]
        growth_tolerance = depth_tolerance
        if plausible_depth.size:
            # Pick the nearest supported depth mode rather than a percentile:
            # a thin wrench jaw can occupy <5% of the search crop, while a few
            # isolated depth outliers must not become seeds.
            bin_width = max(0.001, min(0.004, depth_tolerance * 0.25))
            depth_min = float(plausible_depth.min())
            depth_bins = np.floor((plausible_depth - depth_min) / bin_width).astype(
                np.int64
            )
            counts = np.bincount(depth_bins)
            support_threshold = max(8, int(math.ceil(plausible_depth.size * 0.003)))
            supported = np.flatnonzero(counts >= support_threshold)
            if supported.size:
                nearest_bin = int(supported[0])
                seed_depth = float(np.median(plausible_depth[depth_bins == nearest_bin]))
                # If a second dense mode is separated by a genuine histogram
                # valley, it is normally the support surface behind a low or
                # concave object. Keep region growing on the foreground side
                # of that valley. Curved objects have continuous occupied bins
                # and therefore retain the configured tolerance.
                strong_threshold = max(
                    support_threshold * 4,
                    int(math.ceil(counts[nearest_bin] * 0.12)),
                )
                valley_threshold = max(
                    support_threshold,
                    int(math.ceil(counts[nearest_bin] * 0.03)),
                )
                for candidate_bin in range(nearest_bin + 2, len(counts)):
                    valley = counts[nearest_bin + 1 : candidate_bin]
                    if (
                        counts[candidate_bin] >= strong_threshold
                        and valley.size
                        and int(valley.max()) <= valley_threshold
                    ):
                        mode_separation = (candidate_bin - nearest_bin) * bin_width
                        growth_tolerance = min(
                            growth_tolerance,
                            max(0.0015, mode_separation * 0.40),
                        )
                        break
            else:
                seed_depth = float(np.percentile(plausible_depth, 5.0))
        else:
            patch = np.asarray(observation.depth_m[y0:y1, x0:x1], dtype=np.float64)
            valid_patch = patch[np.isfinite(patch) & (patch > 0)]
            seed_depth = (
                float(np.median(valid_patch))
                if valid_patch.size
                else float(point_camera[2])
            )
        allowed = np.isfinite(observation.depth_m) & (
            np.abs(observation.depth_m - seed_depth) <= growth_tolerance
        )
        rgb = np.asarray(observation.rgb[:, :, :3], dtype=np.float32)
        chromaticity = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 12.0)
        search_chromaticity = chromaticity[sy0:sy1, sx0:sx1]
        foreground = plausible & (np.abs(search_depth - seed_depth) <= growth_tolerance)
        if foreground.any():
            seed_chromaticity = np.median(search_chromaticity[foreground], axis=0)
        else:
            seed_chromaticity = np.median(
                chromaticity[y0:y1, x0:x1].reshape(-1, 3), axis=0
            )
        allowed &= (
            np.linalg.norm(chromaticity - seed_chromaticity.reshape(1, 1, 3), axis=2)
            <= self.chromaticity_tolerance
        )
        yy, xx = np.ogrid[: K.height, : K.width]
        allowed &= (xx - u) ** 2 + (yy - v) ** 2 <= self.max_radius_px**2
        return _connected_component_from_nearest_seed(
            allowed, u, v, search_radius
        )


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
        max_area_growth_per_frame: float = 4.0,
        max_depth_change_m: float = 0.10,
    ) -> None:
        self.seed_segmenter = seed_segmenter or SeededDepthSegmenter()
        self.chromaticity_tolerance = float(chromaticity_tolerance)
        self.min_component_pixels = int(min_component_pixels)
        self.kernel = np.ones((morphology_kernel_px, morphology_kernel_px), dtype=np.uint8)
        self.max_area_growth_per_frame = float(max_area_growth_per_frame)
        self.max_depth_change_m = float(max_depth_change_m)
        self._object_id: str | None = None
        self._reference_chromaticity: np.ndarray | None = None
        self._last_center_uv: np.ndarray | None = None
        self._last_area_px: int | None = None
        self._last_depth_m: float | None = None

    def reset(self) -> None:
        self._object_id = None
        self._reference_chromaticity = None
        self._last_center_uv = None
        self._last_area_px = None
        self._last_depth_m = None

    def segment(self, observation: RGBDObservation, target: TargetSpec) -> np.ndarray | None:
        if target.object_id != self._object_id:
            self.reset()
            self._object_id = target.object_id
        rgb = np.asarray(observation.rgb[:, :, :3], dtype=np.float32)
        chromaticity = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 12.0)
        seeded = self.seed_segmenter.segment(observation, target)
        if self._reference_chromaticity is None:
            if seeded is None or int(np.asarray(seeded, dtype=bool).sum()) < self.min_component_pixels:
                return None
            mask = np.asarray(seeded, dtype=bool)
            self._update_reference(chromaticity, observation.depth_m, mask, replace=True)
            return mask

        # Reprojection is not an open-loop mask: it uses the current wrist pose
        # only to find a seed, then regrows the component from the latest depth
        # and RGB.  Prefer it while the upstream point still falls on a region
        # consistent with the preceding observation.  This prevents pastel or
        # reflective objects from merging with a similarly coloured table.
        if seeded is not None:
            seeded_mask = np.asarray(seeded, dtype=bool)
            if self._is_consistent(seeded_mask, observation.depth_m):
                self._update_reference(
                    chromaticity, observation.depth_m, seeded_mask, replace=False
                )
                return seeded_mask

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
            area_ratio = 1.0 if self._last_area_px is None else area / max(self._last_area_px, 1)
            if not 1.0 / self.max_area_growth_per_frame <= area_ratio <= self.max_area_growth_per_frame:
                continue
            component = labels == label
            component_depth = np.asarray(observation.depth_m[component], dtype=np.float64)
            component_depth = component_depth[np.isfinite(component_depth) & (component_depth > 0)]
            median_depth = float(np.median(component_depth)) if component_depth.size else None
            if (
                median_depth is not None
                and self._last_depth_m is not None
                and abs(median_depth - self._last_depth_m) > self.max_depth_change_m
            ):
                continue
            # Preserve identity under eye-in-hand scale change.  A large area
            # alone is not evidence of being the target: it is commonly the
            # table, gripper, or a reflection-connected background region.
            score = (
                -2.0 * travel / max(diagonal, 1.0)
                - abs(float(np.log(max(area_ratio, 1e-6))))
                + 0.03 * float(np.log1p(area))
            )
            candidates.append((score, label))
        if not candidates:
            return None
        selected_label = max(candidates)[1]
        mask = labels == selected_label
        self._update_reference(chromaticity, observation.depth_m, mask, replace=False)
        return mask

    def _is_consistent(self, mask: np.ndarray, depth_m: np.ndarray) -> bool:
        area = int(mask.sum())
        if area < self.min_component_pixels:
            return False
        area_ratio = 1.0 if self._last_area_px is None else area / max(self._last_area_px, 1)
        if not 1.0 / self.max_area_growth_per_frame <= area_ratio <= self.max_area_growth_per_frame:
            return False
        depths = np.asarray(depth_m[mask], dtype=np.float64)
        depths = depths[np.isfinite(depths) & (depths > 0)]
        if depths.size == 0:
            return False
        median_depth = float(np.median(depths))
        return self._last_depth_m is None or (
            abs(median_depth - self._last_depth_m) <= self.max_depth_change_m
        )

    def _update_reference(
        self,
        chromaticity: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        *,
        replace: bool,
    ) -> None:
        values = chromaticity[mask]
        reference = np.median(values, axis=0)
        if replace or self._reference_chromaticity is None:
            self._reference_chromaticity = reference
        else:
            self._reference_chromaticity = 0.9 * self._reference_chromaticity + 0.1 * reference
        rows, columns = np.nonzero(mask)
        self._last_center_uv = np.asarray([np.median(columns), np.median(rows)], dtype=np.float64)
        self._last_area_px = int(mask.sum())
        depths = np.asarray(depth_m[mask], dtype=np.float64)
        depths = depths[np.isfinite(depths) & (depths > 0)]
        self._last_depth_m = float(np.median(depths)) if depths.size else None


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
