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
    assert_transform,
)
from mr_liu.perception.camera import SceneCamera


PoseProvider = Callable[[], np.ndarray]
AdvanceFrame = Callable[[], None]
MaskProvider = Callable[[TargetSpec], np.ndarray | None]


def T_world_opencv_from_ros_pose(
    position_world: np.ndarray, quaternion_world_ros_wxyz: np.ndarray
) -> np.ndarray:
    """Build the OpenCV optical pose returned by Isaac's ``ros`` camera axes.

    Isaac's implementation names this the ROS/computer-vision convention and
    already converts USD's ``+X right, +Y up, -Z forward`` axes with
    ``diag(1, -1, -1)``.  The result is the OpenCV optical convention used by
    pinhole back-projection: ``+X right, +Y down, +Z forward``.  Applying a
    second X/Y flip here mirrors image coordinates and breaks base-frame
    reconstruction as soon as the wrist rotates away from its initial pose.
    """
    R_world_ros = quaternion_wxyz_to_matrix(quaternion_world_ros_wxyz)
    return make_transform(R_world_ros, np.asarray(position_world).reshape(3))


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
        render_settle_frames: int = 2,
        model_seed: int = 0,
        robot_mask_provider: Callable[[], np.ndarray | None] | None = None,
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
        self.render_settle_frames = max(int(render_settle_frames), 1)
        self._sequence = 0
        self.model_seed = int(model_seed)
        self.robot_mask_provider = robot_mask_provider
        self._reference_T_ee_camera: np.ndarray | None = None
        self._last_render_time = -float("inf")

    def capture(self, target: TargetSpec) -> RGBDObservation | None:
        # RTX annotators are delivered one application update behind the
        # articulation/USD pose on Isaac Sim 6.0.  Advancing twice pairs the
        # image with the pose used to render it; a single update produced one
        # large false object jump immediately after every robot motion.
        sample = None
        # Read RGB, depth and camera parameters from one acquisition callback.
        # get_rgba/get_depth individually read annotators that can be at a
        # different pipeline stage after physics-only motion stepping.
        for attempt in range(16):
            self.advance_frame()
            sample = self.camera.rgbd_frame()
            if (sample is None or attempt + 1 < self.render_settle_frames
                    or sample.rendering_time_s <= self._last_render_time):
                continue
            render_pose = sample.T_world_camera
            position, quaternion = self.camera.world_pose(camera_axes="ros")
            live_pose = T_world_opencv_from_ros_pose(position, quaternion)
            if (np.linalg.norm(render_pose[:3, 3] - live_pose[:3, 3]) <= 0.001
                    and np.linalg.norm(render_pose[:3, :3] - live_pose[:3, :3]) <= 0.005):
                break
        else:
            return None
        rgb, depth, K = sample.rgba, sample.depth_m, sample.intrinsics
        width, height = self.camera.resolution
        position, quaternion = self.camera.world_pose(camera_axes="ros")
        T_base_camera = T_world_opencv_from_ros_pose(position, quaternion)
        T_base_ee = np.asarray(self.ee_pose_provider(), dtype=np.float64)
        T_ee_camera = invert_transform(T_base_ee) @ T_base_camera
        self._validate_static_extrinsic(T_ee_camera)
        T_base_camera = render_pose
        T_base_ee = T_base_camera @ invert_transform(T_ee_camera)
        self._last_render_time = sample.rendering_time_s
        self._sequence += 1
        mask = self.mask_provider(target) if self.mask_provider is not None else None
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
            metadata={"sim_rendering_time": sample.rendering_time_s, "model_seed": self.model_seed,
                      "sim_acquisition_id": sample.acquisition_id,
                      "pose_source": "render_camera_params",
                      "robot_self_mask": self.robot_mask_provider() if self.robot_mask_provider else None,
                      "render_updates": attempt + 1},
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


class IsaacSceneCamera:
    """Fixed eye-to-hand RGB-D; never fabricate a wrist hand-eye chain.

    T_base_world must come from the host's calibrated world/base relation.
    In this Isaac scene the application's base coordinates are world metres.
    """
    def __init__(self, camera, *, T_base_world, advance_frame, clock=time.monotonic,
                 robot_mask_provider=None):
        if camera.which not in {"scene", "placement"} or not camera.has_depth:
            raise ValueError("External observer requires a configured fixed RGB-D camera")
        self.camera, self.advance_frame, self.clock = camera, advance_frame, clock
        self.T_base_world = assert_transform(T_base_world, name="T_base_world").copy()
        self.robot_mask_provider = robot_mask_provider
        self._last_id = None
        self._pose = None
        self._sequence = 0
        self._last_render_time = -float("inf")

    def capture(self, target):
        del target
        for step in range(16):
            self.advance_frame()
            sample = self.camera.rgbd_frame()
            if (sample is None or step < 1 or sample.acquisition_id == self._last_id
                    or sample.rendering_time_s <= self._last_render_time):
                continue
            if self._pose is not None and not np.allclose(self._pose, sample.T_world_camera, atol=1e-5):
                raise RuntimeError("Fixed overhead camera moved; recalibration required")
            self._pose = sample.T_world_camera.copy()
            self._last_id = sample.acquisition_id
            self._last_render_time = sample.rendering_time_s
            self._sequence += 1
            k = sample.intrinsics
            width, height = self.camera.resolution
            return RGBDObservation(
                sequence=self._sequence, timestamp_s=self.clock(), rgb=sample.rgba[:, :, :3],
                depth_m=sample.depth_m, intrinsics=CameraIntrinsics(width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2]),
                T_base_camera=self.T_base_world @ sample.T_world_camera,
                camera_frame=f"{self.camera.which}_camera",
                metadata={"mount_type": "fixed", "sim_rendering_time": sample.rendering_time_s,
                          "robot_self_mask": self.robot_mask_provider() if self.robot_mask_provider else None},
            )
        return None
