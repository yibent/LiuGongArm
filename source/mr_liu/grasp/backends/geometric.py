"""Deterministic antipodal/PCA fallback and integration-test backend.

This is intentionally not presented as the learned unseen-object model. It is
kept as a safe no-weight fallback, a simulator smoke-test backend and a way to
separate closed-loop failures from neural inference failures.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mr_liu.grasp.contracts import GraspCandidate, RGBDObservation, TargetSpec
from mr_liu.grasp.geometry import estimate_object_frame
from mr_liu.grasp.transforms import make_transform


class GeometricAntipodalBackend:
    name = "geometric_antipodal"

    def __init__(self, *, width_margin_m: float = 0.008) -> None:
        self.width_margin_m = float(width_margin_m)

    def generate(
        self,
        observation: RGBDObservation,
        object_points_camera: np.ndarray,
        target: TargetSpec,
    ) -> Sequence[GraspCandidate]:
        T_camera_object, extents = estimate_object_frame(object_points_camera)
        axes = T_camera_object[:3, :3]
        center = T_camera_object[:3, 3]
        candidates: list[GraspCandidate] = []

        # A single depth view makes the unseen thickness axis look artificially
        # tiny. Never close across an axis parallel to the viewing/approach ray;
        # doing so would propose a vertical pinch against the support surface.
        camera_approach = np.asarray([0.0, 0.0, 1.0])
        lateral_axes = [
            index for index in range(3) if abs(float(np.dot(axes[:, index], camera_approach))) < 0.72
        ]
        if not lateral_axes:
            lateral_axes = list(np.argsort(extents)[-2:])
        lateral_axes.sort(key=lambda index: float(extents[index]))
        for rank, closing_index in enumerate(lateral_axes[:2]):
            closing = axes[:, closing_index]
            surface_normal_index = int(np.argmin(extents))
            surface_normal = axes[:, surface_normal_index]
            if float(np.dot(surface_normal, camera_approach)) < 0.0:
                surface_normal *= -1.0
            approach_options = [camera_approach, surface_normal]
            for approach_rank, raw_approach in enumerate(approach_options):
                approach = raw_approach - closing * float(np.dot(raw_approach, closing))
                norm = float(np.linalg.norm(approach))
                if norm < 1e-6:
                    continue
                approach /= norm
                lateral = np.cross(approach, closing)
                lateral /= max(float(np.linalg.norm(lateral)), 1e-9)
                rotation = np.column_stack((closing, lateral, approach))
                if np.linalg.det(rotation) < 0:
                    lateral *= -1.0
                    rotation = np.column_stack((closing, lateral, approach))
                width = float(extents[closing_index] + self.width_margin_m)
                score = 0.65 - rank * 0.08 - approach_rank * 0.05
                candidates.append(
                    GraspCandidate(
                        T_camera_grasp=make_transform(rotation, center),
                        width_m=width,
                        score=score,
                        observation_sequence=observation.sequence,
                        backend=self.name,
                        metadata={
                            "closing_extent_axis": int(closing_index),
                            "fallback": True,
                            "support_surface_completion": True,
                        },
                    )
                )
        return candidates
