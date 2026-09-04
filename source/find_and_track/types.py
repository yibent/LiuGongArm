from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

PathName = Literal["idle", "slow", "fast"]
BackendName = Literal["yoloe", "cv"]


@dataclass
class Detection:
    xyxy: np.ndarray
    label: str
    score: float = 1.0
    track_id: int | None = None
    source: Literal["slow", "fast"] = "slow"
    class_id: int = 0

    def clip(self, width: int, height: int) -> "Detection":
        x1, y1, x2, y2 = self.xyxy.astype(np.float32)
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 1, width))
        y2 = float(np.clip(y2, 1, height))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        self.xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
        return self

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, float(x2 - x1) * float(y2 - y1))

    def as_xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy.astype(int)
        return int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))


@dataclass
class RuntimeConfig:
    prompt: str = "person"
    prompt_version: int = 0
    fast_backend: BackendName = "yoloe"
    slow_interval: int = 45
    conf: float = 0.25
    imgsz: int = 640
    lost_patience: int = 12
    find_epoch: int = 0
    stop: bool = False


@dataclass
class FrameStats:
    frame_idx: int = 0
    path: PathName = "idle"
    fps: float = 0.0
    slow_ms: float = 0.0
    fast_ms: float = 0.0
    prompt: str = ""
    backend: BackendName = "yoloe"
    n_slow: int = 0
    n_fast: int = 0
    lost: bool = False
    message: str = ""


@dataclass
class FrameResult:
    image: np.ndarray
    slow: list[Detection] = field(default_factory=list)
    fast: list[Detection] = field(default_factory=list)
    stats: FrameStats = field(default_factory=FrameStats)
