"""Isaac Sim scene / wrist RGB capture. Imports isaacsim only on use."""

from __future__ import annotations

from typing import Any

import numpy as np

from mr_liu.config import cameras_config


class SceneCamera:
    """Thin wrapper around isaacsim.sensors.camera.Camera."""

    def __init__(self, which: str = "scene") -> None:
        cfg = cameras_config()
        block = cfg.get(which) or cfg["scene"]
        self.prim_path = str(block["prim_path"])
        self.resolution = tuple(int(v) for v in block["resolution"])
        self._cam: Any = None

    def spawn(self, position: list[float], target: list[float]) -> None:
        from isaacsim.sensors.camera import Camera

        width, height = self.resolution
        self._cam = Camera(
            prim_path=self.prim_path,
            frequency=30,
            resolution=(width, height),
            position=np.array(position, dtype=np.float32),
        )
        self._cam.set_focal_length(1.8)
        self._cam.set_clipping_range(0.01, 10.0)
        self._cam.initialize()
        try:
            self._cam.set_local_pose(translation=np.array(position, dtype=np.float32))
        except Exception:
            pass
        print(f"[mr_liu] Camera {self.prim_path} {self.resolution} look-at {target}")

    def rgb_bgr(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        rgba = self._cam.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        frame = np.asarray(rgba)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        bgr = frame[:, :, :3][:, :, ::-1].copy()
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)
        return bgr
