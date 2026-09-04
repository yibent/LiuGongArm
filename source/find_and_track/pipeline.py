from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .cv_tracker import CvFeatureTracker
from .florence_finder import FlorenceFinder, parse_prompts
from .types import Detection, FrameResult, FrameStats, RuntimeConfig
from .visualize import draw
from .yoloe_tracker import YoloeVisualTracker

LOGGER = logging.getLogger("find_and_track")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def imread_unicode(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class FindTrackPipeline:
    def __init__(
        self,
        yoloe_weights: str | Path,
        florence_id: str | None = None,
        device: str | None = None,
    ):
        self.florence = FlorenceFinder(model_id=florence_id, device=device)
        self.yoloe = YoloeVisualTracker(weights=yoloe_weights, device=device)
        self.cv = CvFeatureTracker()
        self._frame_idx = 0
        self._seen_version = -1
        self._seen_epoch = -1
        self._has_target = False
        self._lost_frames = 0
        self._last_slow = -10_000
        self._fps_ema = 0.0
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, backend: str = "yoloe") -> None:
        self.florence.load()
        if backend == "yoloe":
            self.yoloe.load()
        self._loaded = True

    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "florence": self.florence.ready,
            "florence_id": self.florence.loaded_id,
            "yoloe": self.yoloe.ready,
            "has_target": self._has_target,
            "device": str(self.florence.device),
        }

    def reset(self) -> None:
        self._frame_idx = 0
        self._seen_version = -1
        self._seen_epoch = -1
        self._has_target = False
        self._lost_frames = 0
        self._last_slow = -10_000
        self._fps_ema = 0.0
        self.yoloe.reset()
        self.cv.reset()

    def process_frame(self, frame_bgr: np.ndarray, cfg: RuntimeConfig) -> FrameResult:
        t0 = time.perf_counter()
        prompt_text = cfg.prompt
        prompts = parse_prompts(prompt_text)
        prompt_changed = cfg.prompt_version != self._seen_version
        find_requested = cfg.find_epoch != self._seen_epoch
        if prompt_changed or find_requested:
            self._seen_version = cfg.prompt_version
            self._seen_epoch = cfg.find_epoch
            self._has_target = False
            self.yoloe.reset()
            self.cv.reset()

        due_interval = (
            cfg.slow_interval > 0
            and self._frame_idx > 0
            and (self._frame_idx - self._last_slow) >= cfg.slow_interval
        )
        lost = self._lost_frames >= cfg.lost_patience
        need_slow = bool(prompts) and (
            (not self._has_target)
            or prompt_changed
            or find_requested
            or lost
            or due_interval
        )

        slow_dets: list[Detection] = []
        slow_ms = 0.0
        path = "idle"
        message = ""

        if cfg.fast_backend == "yoloe" and not self.yoloe.ready:
            self.yoloe.load()

        if need_slow:
            t_slow = time.perf_counter()
            slow_dets = self.florence.find(frame_bgr, prompts)
            slow_ms = (time.perf_counter() - t_slow) * 1000
            self._last_slow = self._frame_idx
            if slow_dets:
                self.cv.init(frame_bgr, slow_dets)
                if cfg.fast_backend == "yoloe":
                    self.yoloe.imgsz = cfg.imgsz
                    self.yoloe.conf = cfg.conf
                    self.yoloe.set_visual_prompt(frame_bgr, slow_dets, conf=cfg.conf)
                self._has_target = True
                self._lost_frames = 0
                path = "slow"
                message = f"Florence found {len(slow_dets)} target(s)"
            else:
                message = "Florence found nothing"
                if not self._has_target:
                    path = "slow"

        fast_dets: list[Detection] = []
        t_fast = time.perf_counter()
        if self._has_target:
            if cfg.fast_backend == "yoloe" and self.yoloe.has_prompt:
                if path == "slow":
                    fast_dets = self.yoloe.detect(frame_bgr, conf=cfg.conf)
                else:
                    fast_dets = self.yoloe.track(frame_bgr, conf=cfg.conf, persist=True)
                    path = "fast"
            else:
                fast_dets = self.cv.update(frame_bgr)
                if path != "slow":
                    path = "fast"
        fast_ms = (time.perf_counter() - t_fast) * 1000

        if path == "slow" and not fast_dets:
            # On the find frame, surface Florence boxes as the current tracks too.
            for det in slow_dets:
                clone = Detection(
                    xyxy=det.xyxy.copy(),
                    label=det.label,
                    score=det.score,
                    track_id=det.track_id,
                    source="fast",
                    class_id=det.class_id,
                )
                fast_dets.append(clone)

        if not fast_dets:
            self._lost_frames += 1
        else:
            self._lost_frames = 0

        dt = time.perf_counter() - t0
        inst_fps = 1.0 / dt if dt > 1e-6 else 0.0
        self._fps_ema = inst_fps if self._fps_ema == 0 else 0.2 * inst_fps + 0.8 * self._fps_ema

        stats = FrameStats(
            frame_idx=self._frame_idx,
            path=path,  # type: ignore[arg-type]
            fps=self._fps_ema,
            slow_ms=slow_ms,
            fast_ms=fast_ms,
            prompt=prompt_text,
            backend=cfg.fast_backend,
            n_slow=len(slow_dets),
            n_fast=len(fast_dets),
            lost=self._lost_frames >= cfg.lost_patience,
            message=message,
        )
        vis = draw(frame_bgr, slow_dets, fast_dets, stats)
        self._frame_idx += 1
        return FrameResult(image=vis, slow=slow_dets, fast=fast_dets, stats=stats)

    def process_source(
        self,
        source: str | Path | int,
        cfg: RuntimeConfig,
        output: str | Path | None = None,
    ) -> Iterator[FrameResult]:
        if not self._loaded:
            self.load(cfg.fast_backend)
        self.reset()
        path = None if isinstance(source, int) else Path(source)
        writer = None
        try:
            if path is not None and path.suffix.lower() in IMAGE_EXTS:
                frame = imread_unicode(path)
                if frame is None:
                    raise FileNotFoundError(f"Cannot read image: {path}")
                result = self.process_frame(frame, cfg)
                if output:
                    self._write_image(output, result.image)
                yield result
                return

            cap = cv2.VideoCapture(source if isinstance(source, int) else str(path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open source: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            while not cfg.stop:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                result = self.process_frame(frame, cfg)
                if output and writer is None:
                    writer = self._open_writer(output, result.image.shape, fps)
                if writer is not None:
                    writer.write(result.image)
                yield result
            cap.release()
        finally:
            if writer is not None:
                writer.release()

    @staticmethod
    def _open_writer(output: str | Path, shape, fps: float) -> cv2.VideoWriter:
        h, w = shape[:2]
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(output), fourcc, max(1.0, float(fps)), (w, h))

    @staticmethod
    def _write_image(output: str | Path, image: np.ndarray) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        ext = output.suffix or ".jpg"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            raise RuntimeError(f"Failed to encode {output}")
        buf.tofile(str(output))
