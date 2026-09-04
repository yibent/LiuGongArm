"""Isaac Sim wrist RGB-D source with explicit eye-in-hand calibration chain."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation, TargetSpec
from mr_liu.grasp.transforms import (
    invert_transform,
    make_transform,
    quaternion_wxyz_to_matrix,
)
from mr_liu.perception.camera import SceneCamera


PoseProvider = Callable[[], np.ndarray]
AdvanceFrame = Callable[[], None]
MaskProvider = Callable[[TargetSpec], np.ndarray | None]


def T_world_opencv_from_ros_pose(
    position_world: np.ndarray, quaternion_world_ros_wxyz: np.ndarray
) -> np.ndarray:
    """Convert Isaac's ROS camera pose (+Y up,+Z forward) to OpenCV optical axes.

    OpenCV point clouds use +X right, +Y down and +Z forward. Isaac's documented
    ROS camera axes use +X left, +Y up and +Z forward, so both X and Y flip.
    """
    R_world_ros = quaternion_wxyz_to_matrix(quaternion_world_ros_wxyz)
    R_ros_opencv = np.diag([-1.0, -1.0, 1.0])
    return make_transform(R_world_ros @ R_ros_opencv, np.asarray(position_world).reshape(3))


class IsaacWristCamera:
    def __init__(
        self,
        camera: SceneCamera,
        *,
        ee_pose_provider: PoseProvider,
        advance_frame: AdvanceFrame,
        mask_provider: MaskProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_extrinsic_translation_drift_m: float = 0.001,
        max_extrinsic_rotation_drift: float = 0.002,
    ) -> None:
        if camera.which != "wrist":
            raise ValueError("IsaacWristCamera requires the configured wrist camera")
        self.camera = camera
        self.ee_pose_provider = ee_pose_provider
        self.advance_frame = advance_frame
        self.mask_provider = mask_provider
        self.clock = clock
        self.max_extrinsic_translation_drift_m = float(max_extrinsic_translation_drift_m)
        self.max_extrinsic_rotation_drift = float(max_extrinsic_rotation_drift)
        self._sequence = 0
        self._reference_T_ee_camera: np.ndarray | None = None

    def capture(self, target: TargetSpec) -> RGBDObservation | None:
        self.advance_frame()
        rgb = self.camera.rgb_rgba()
        depth = self.camera.depth_m()
        if rgb is None or depth is None:
            return None
        K = self.camera.intrinsics_matrix()
        width, height = self.camera.resolution
        position, quaternion = self.camera.world_pose(camera_axes="ros")
        T_base_camera = T_world_opencv_from_ros_pose(position, quaternion)
        T_base_ee = np.asarray(self.ee_pose_provider(), dtype=np.float64)
        T_ee_camera = invert_transform(T_base_ee) @ T_base_camera
        self._validate_static_extrinsic(T_ee_camera)
        self._sequence += 1
        mask = self.mask_provider(target) if self.mask_provider is not None else None
        frame = getattr(self.camera, "_cam", None)
        sim_frame = frame.get_current_frame() if frame is not None else {}
        return RGBDObservation(
            sequence=self._sequence,
            timestamp_s=self.clock(),
            rgb=np.asarray(rgb[:, :, :3]).copy(),
            depth_m=np.asarray(depth, dtype=np.float32).copy(),
            intrinsics=CameraIntrinsics(
                width=width,
                height=height,
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
            ),
            T_base_camera=T_base_camera,
            T_base_ee=T_base_ee,
            T_ee_camera=T_ee_camera,
            target_mask=None if mask is None else np.asarray(mask, dtype=bool).copy(),
            metadata={"sim_rendering_time": sim_frame.get("rendering_time")},
        )

    @property
    def calibrated_T_ee_camera(self) -> np.ndarray | None:
        return None if self._reference_T_ee_camera is None else self._reference_T_ee_camera.copy()

    def _validate_static_extrinsic(self, current: np.ndarray) -> None:
        if self._reference_T_ee_camera is None:
            self._reference_T_ee_camera = current.copy()
            return
        translation_drift = float(
            np.linalg.norm(current[:3, 3] - self._reference_T_ee_camera[:3, 3])
        )
        rotation_drift = float(
            np.linalg.norm(current[:3, :3] - self._reference_T_ee_camera[:3, :3], ord="fro")
        )
        if (
            translation_drift > self.max_extrinsic_translation_drift_m
            or rotation_drift > self.max_extrinsic_rotation_drift
        ):
            raise RuntimeError(
                "Eye-in-hand extrinsic changed: "
                f"translation drift={translation_drift:.6f}m rotation_fro={rotation_drift:.6f}"
            )
