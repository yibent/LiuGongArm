"""High-rate model-free pose servo for the last few centimetres."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mr_liu.grasp.transforms import interpolate_pose_step, pose_error


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
    aligned: bool


class PoseServo:
    def __init__(self, config: ServoConfig) -> None:
        self.config = config
        self._stable_count = 0

    def reset(self) -> None:
        self._stable_count = 0

    def update(self, T_base_ee: np.ndarray, T_base_desired: np.ndarray) -> ServoUpdate:
        translation, rotation = pose_error(T_base_ee, T_base_desired)
        translation_error = float(np.linalg.norm(translation))
        within = (
            translation_error <= self.config.translation_tolerance_m
            and rotation <= self.config.rotation_tolerance_rad
        )
        self._stable_count = self._stable_count + 1 if within else 0
        command = interpolate_pose_step(
            T_base_ee,
            T_base_desired,
            max_translation_m=self.config.max_translation_step_m,
            max_rotation_rad=self.config.max_rotation_step_rad,
        )
        return ServoUpdate(
            command_pose=command,
            translation_error_m=translation_error,
            rotation_error_rad=rotation,
            aligned=self._stable_count >= self.config.stable_frames,
        )
