"""Local wrist-view segmentation adapters."""

from __future__ import annotations

from collections import deque

import numpy as np

from mr_liu.grasp.contracts import RGBDObservation, TargetSpec
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
