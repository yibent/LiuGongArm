"""Config-driven Isaac Sim RGB and RGBD cameras."""

from __future__ import annotations

from typing import Any

import numpy as np

from mr_liu.config import cameras_config


class SceneCamera:
    """Thin wrapper around ``isaacsim.sensors.camera.Camera``.

    The ``scene`` camera is world-mounted over the table. The ``wrist`` camera
    is authored below the gripper prim, so its pose follows the articulation.
    """

    def __init__(self, which: str = "scene") -> None:
        cfg = cameras_config()
        block = cfg.get(which) or cfg["scene"]
        self.which = which
        self.config = block
        self.prim_path = str(block["prim_path"])
        self.resolution = tuple(int(v) for v in block["resolution"])
        self.annotators = tuple(str(v) for v in block.get("annotators", ["rgb"]))
        self._cam: Any = None

    @property
    def has_depth(self) -> bool:
        return "distance_to_image_plane" in self.annotators

    def spawn(
        self,
        position: list[float] | None = None,
        target: list[float] | None = None,
    ) -> None:
        from isaacsim.sensors.camera import Camera

        width, height = self.resolution
        kwargs: dict[str, Any] = {
            "prim_path": self.prim_path,
            "name": f"{self.which}_camera",
            "frequency": int(self.config.get("frequency", 30)),
            "resolution": (width, height),
        }
        orientation = self.config.get("orientation_wxyz")
        if orientation is not None:
            kwargs["orientation"] = np.asarray(orientation, dtype=np.float32)
        translation = self.config.get("translation")
        world_position = position or self.config.get("position")
        if translation is not None:
            self._require_parent()
            kwargs["translation"] = np.asarray(translation, dtype=np.float32)
        elif world_position is not None:
            kwargs["position"] = np.asarray(world_position, dtype=np.float32)
        else:
            raise ValueError(f"Camera {self.which!r} needs position or translation")

        self._cam = Camera(**kwargs)
        self._cam.set_focal_length(float(self.config.get("focal_length", 1.8)))
        near, far = self.config.get("clipping_range", [0.01, 10.0])
        self._cam.set_clipping_range(float(near), float(far))
        self._cam.initialize(attach_rgb_annotator="rgb" in self.annotators)
        if self.has_depth:
            self._cam.add_distance_to_image_plane_to_frame()

        target = target or self.config.get("target")
        mount = "local" if translation is not None else "world"
        suffix = f" target={target}" if target is not None else ""
        print(
            f"[mr_liu] {self.which} camera {self.prim_path} "
            f"{self.resolution} {self.annotators} mount={mount}{suffix}"
        )

    def _require_parent(self) -> None:
        import omni.usd

        parent_path = str(self.config.get("parent_prim_path") or self.prim_path.rsplit("/", 1)[0])
        parent = omni.usd.get_context().get_stage().GetPrimAtPath(parent_path)
        if not parent.IsValid():
            raise RuntimeError(f"Camera parent prim does not exist: {parent_path}")

    def rgb_rgba(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        rgba = self._cam.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        frame = np.asarray(rgba)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        return frame

    def depth_m(self) -> np.ndarray | None:
        """Return distance-to-image-plane depth in metres when configured."""
        if self._cam is None or not self.has_depth:
            return None
        depth = self._cam.get_depth()
        if depth is None or getattr(depth, "size", 0) == 0:
            return None
        frame = np.asarray(depth, dtype=np.float32)
        if frame.ndim != 2:
            return None
        return frame

    def world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cam is None:
            raise RuntimeError(f"Camera {self.which!r} has not been spawned")
        return self._cam.get_world_pose(camera_axes="world")

    def rgb_bgr(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        frame = self.rgb_rgba()
        if frame is None:
            return None
        bgr = frame[:, :, :3][:, :, ::-1].copy()
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)
        return bgr


def spawn_configured_cameras() -> dict[str, SceneCamera]:
    """Spawn every enabled camera after the table and robot are on stage."""
    cfg = cameras_config()
    if not bool(cfg.get("enabled", True)):
        return {}
    cameras: dict[str, SceneCamera] = {}
    for which in ("scene", "wrist"):
        if which not in cfg:
            continue
        camera = SceneCamera(which)
        camera.spawn()
        cameras[which] = camera
    return cameras
