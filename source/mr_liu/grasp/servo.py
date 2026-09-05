"""High-rate model-free pose servo for the last few centimetres."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mr_liu.grasp.transforms import align_rotation_axis, axis_alignment_error, interpolate_pose_step, pose_error


@dataclass(frozen=True)
class ServoConfig:
    translation_tolerance_m: float
    rotation_tolerance_rad: float
    max_translation_step_m: float
    max_rotation_step_rad: float
    stable_frames: int


@dataclass(frozen=True)
class ServoUpdate:
    command_pose: np.ndarray
    translation_error_m: float
    rotation_error_rad: float
    within_tolerance: bool
    aligned: bool


class PoseServo:
    def __init__(
        self,
        config: ServoConfig,
        *,
        track_orientation: bool = True,
        orientation_mode: str | None = None,
    ) -> None:
        self.config = config
        self.orientation_mode = orientation_mode or ("full" if track_orientation else "none")
        if self.orientation_mode not in {"full", "approach_axis", "none"}:
            raise ValueError(f"Unsupported servo orientation mode: {self.orientation_mode}")
        self._stable_count = 0

    def reset(self) -> None:
        self._stable_count = 0

    def update(self, T_base_ee: np.ndarray, T_base_desired: np.ndarray) -> ServoUpdate:
        translation, rotation = pose_error(T_base_ee, T_base_desired)
        desired = np.asarray(T_base_desired, dtype=np.float64)
        if self.orientation_mode == "none":
            rotation = 0.0
            desired = desired.copy()
            desired[:3, :3] = np.asarray(T_base_ee)[:3, :3]
        elif self.orientation_mode == "approach_axis":
            current_rotation = np.asarray(T_base_ee, dtype=np.float64)[:3, :3]
            target_rotation = desired[:3, :3]
            rotation = axis_alignment_error(current_rotation[:, 2], target_rotation[:, 2])
            desired = desired.copy()
            desired[:3, :3] = align_rotation_axis(current_rotation, target_rotation[:, 2])
        translation_error = float(np.linalg.norm(translation))
        within = (
            translation_error <= self.config.translation_tolerance_m
            and rotation <= self.config.rotation_tolerance_rad
        )
        self._stable_count = self._stable_count + 1 if within else 0
        command = interpolate_pose_step(
            T_base_ee,
            desired,
            max_translation_m=self.config.max_translation_step_m,
            max_rotation_rad=self.config.max_rotation_step_rad,
        )
        return ServoUpdate(
            command_pose=command,
            translation_error_m=translation_error,
            rotation_error_rad=rotation,
            within_tolerance=within,
            aligned=self._stable_count >= self.config.stable_frames,
        )
