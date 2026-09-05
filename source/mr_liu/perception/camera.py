"""Config-driven Isaac Sim RGB and RGBD cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mr_liu.config import cameras_config


@dataclass(frozen=True)
class CameraRGBDFrame:
    """One simulator acquisition, with metric optical-Z depth and render pose.

    T_world_camera maps OpenCV optical coordinates into the simulator world,
    not necessarily the physical robot base. Independent cameras may have
    different rendering times; callers must check freshness before fusion.
    """

    rgba: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    rendering_time_s: float
    rendering_frame: int | None
    # Isaac 6 returns a rational ReferenceTime instead of an integer frame ID.
    render_reference: tuple[int, int] | None = None

    @property
    def acquisition_id(self) -> int | tuple[int, int] | None:
        return self.render_reference if self.render_reference is not None else self.rendering_frame


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
            self._cam.attach_annotator("camera_params")
        if 'semantic_segmentation' in self.annotators:
            self._cam.add_semantic_segmentation_to_frame({'colorize': False})

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

    def rgbd_frame(self) -> CameraRGBDFrame | None:
        """Copy RGB, depth and pose from a single acquisition callback.

        Does not advance simulation or promise a newer frame on each call.
        Never pairs a buffered image with a separately queried live pose.
        """
        if self._cam is None or not self.has_depth:
            return None
        frame = self._cam.get_current_frame()
        rgb, depth = frame.get("rgb"), frame.get("distance_to_image_plane")
        params = frame.get("camera_params")
        stamp, number = frame.get("rendering_time"), frame.get("rendering_frame")
        if rgb is None or depth is None or not isinstance(params, dict):
            return None
        if stamp is None or number is None or "cameraViewTransform" not in params:
            return None
        reference = None
        if isinstance(number, dict):
            numerator = number.get("referenceTimeNumerator")
            denominator = number.get("referenceTimeDenominator")
            if numerator is None or denominator is None or int(denominator) <= 0:
                return None
            reference = (int(numerator), int(denominator))
            number = None
        rgb, depth = np.asarray(rgb), np.asarray(depth, dtype=np.float32)
        width, height = self.resolution
        if (rgb.ndim != 3 or rgb.shape[:2] != (height, width) or rgb.shape[2] < 3
                or depth.shape != (height, width)):
            return None
        if not np.isfinite(float(stamp)):
            return None
        view = np.asarray(params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
        # USD view transforms use row vectors, +Y up and -Z forward.
        pose = np.linalg.inv(view.T) @ np.diag([1., -1., -1., 1.])
        intrinsics = self.intrinsics_matrix()
        if not np.isfinite(pose).all() or not np.isfinite(intrinsics).all():
            return None
        return CameraRGBDFrame(
            rgba=rgb.copy(), depth_m=depth.copy(), intrinsics=intrinsics.copy(),
            T_world_camera=pose, rendering_time_s=float(stamp),
            rendering_frame=None if number is None else int(number), render_reference=reference,
        )

    def world_pose(self, camera_axes: str = "world") -> tuple[np.ndarray, np.ndarray]:
        if self._cam is None:
            raise RuntimeError(f"Camera {self.which!r} has not been spawned")
        return self._cam.get_world_pose(camera_axes=camera_axes)

    def intrinsics_matrix(self) -> np.ndarray:
        if self._cam is None:
            raise RuntimeError(f"Camera {self.which!r} has not been spawned")
        return np.asarray(self._cam.get_intrinsics_matrix(device="cpu"), dtype=np.float64)

    def enable_instance_segmentation(self, *, leaf_paths: bool = False) -> None:
        """Attach a non-colorized ground-truth instance annotator for sim tests."""
        if self._cam is None:
            raise RuntimeError(f"Camera {self.which!r} has not been spawned")
        if leaf_paths:
            self._cam.add_instance_id_segmentation_to_frame({"colorize": False})
        else:
            self._cam.add_instance_segmentation_to_frame({"colorize": False})

    def instance_mask(self, object_id: str, *, leaf_paths: bool = False) -> np.ndarray | None:
        """Return the synchronized simulator mask whose label/path matches ``object_id``."""
        if self._cam is None:
            return None
        payload = self.instance_frame(leaf_paths=leaf_paths)
        if not isinstance(payload, dict):
            return None
        data = np.asarray(payload.get("data"))
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        if data.ndim != 2:
            return None
        labels = (payload.get("info") or {}).get("idToLabels") or {}
        matching: list[int] = []
        needle = str(object_id).casefold()
        for raw_id, value in labels.items():
            text = str(value).casefold()
            if needle in text:
                try:
                    matching.append(int(raw_id))
                except (TypeError, ValueError):
                    continue
        if not matching:
            return np.zeros(data.shape, dtype=bool) if leaf_paths else None
        return np.isin(data, matching)

    def instance_frame(self, *, leaf_paths: bool = False):
        if self._cam is None:
            return None
        frame = self._cam.get_current_frame()
        # Isaac 6 normalizes annotator names by removing the _fast suffix.
        key = "instance_id_segmentation" if leaf_paths else "instance_segmentation"
        payload = frame.get(key)
        return payload if payload is not None else frame.get(key + "_fast")

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if self._cam is None:
            raise RuntimeError('Camera is not initialized')
        return self._cam.get_world_points_from_image_coords(pixels, depth, device='cpu')

    def semantic_frame(self):
        return self._cam.get_current_frame().get('semantic_segmentation') if self._cam else None

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
