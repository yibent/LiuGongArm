"""Coordinated FIND/TRACK pipelines for independent camera views."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .florence_finder import FlorenceFinder
from .memory import ObjectMemoryStore
from .pipeline import FindTrackPipeline
from .types import FrameResult, RuntimeConfig


class MultiViewFindTrackPipeline:
    """Share one Florence model while keeping one YOLOE tracker per view.

    Florence has no temporal state, so sharing saves GPU memory. YOLOE/BYTETrack
    state must remain independent because camera coordinates and motion differ.
    """

    def __init__(
        self,
        yoloe_weights: str | Path | None = None,
        *,
        views: tuple[str, ...] = ("scene", "wrist"),
        florence_id: str | None = None,
        device: str | None = None,
        pipelines: Mapping[str, FindTrackPipeline] | None = None,
        memory_store: ObjectMemoryStore | None = None,
    ) -> None:
        if pipelines is not None:
            self.pipelines = dict(pipelines)
            if not self.pipelines:
                raise ValueError("At least one camera pipeline is required")
            return
        if yoloe_weights is None:
            raise ValueError("yoloe_weights is required when pipelines are not supplied")

        shared_finder = FlorenceFinder(model_id=florence_id, device=device)
        self.pipelines = {
            view: FindTrackPipeline(
                yoloe_weights=yoloe_weights,
                florence_id=florence_id,
                device=device,
                finder=shared_finder,
                memory_store=memory_store,
            )
            for view in views
        }

    @property
    def loaded(self) -> bool:
        return all(pipeline.loaded for pipeline in self.pipelines.values())

    def load(self, backend: str = "yoloe") -> None:
        for pipeline in self.pipelines.values():
            pipeline.load(backend)

    def reset(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.reset()

    def process_frames(
        self,
        frames_bgr: Mapping[str, np.ndarray],
        cfg: RuntimeConfig,
    ) -> dict[str, FrameResult]:
        missing = set(self.pipelines) - set(frames_bgr)
        if missing:
            raise KeyError(f"Missing camera frames: {sorted(missing)}")
        results = {}
        for view, pipeline in self.pipelines.items():
            view_cfg = RuntimeConfig(**{**cfg.__dict__, "memory_view": cfg.memory_view or view})
            results[view] = pipeline.process_frame(frames_bgr[view], view_cfg)
        return results

    def status(self) -> dict[str, dict]:
        return {view: pipeline.status() for view, pipeline in self.pipelines.items()}
