from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .cv_tracker import CvFeatureTracker
from .florence_finder import FlorenceFinder, parse_prompts
from .memory import ObjectMemoryStore
from .types import Detection, FrameResult, FrameStats, RuntimeConfig
from .visualize import draw
from .yoloe_tracker import YoloeVisualTracker
from .settings import default_yoloe

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
        yoloe_weights: str | Path | None = None,
        florence_id: str | None = None,
        device: str | None = None,
        *,
        finder: FlorenceFinder | None = None,
        yoloe_tracker: YoloeVisualTracker | None = None,
        memory_store: ObjectMemoryStore | None = None,
    ):
        self.florence = finder or FlorenceFinder(model_id=florence_id, device=device)
        self.yoloe = yoloe_tracker or YoloeVisualTracker(
            weights=yoloe_weights if yoloe_weights is not None else default_yoloe(), device=device)
        self.cv = CvFeatureTracker()
        self._frame_idx = 0
        self._seen_version = -1
        self._seen_epoch = -1
        self._has_target = False
        self._lost_frames = 0
        self._last_slow = -10_000
        self._fps_ema = 0.0
        self._loaded = False
        self._seen_backend: str | None = None
        self.memory_store = memory_store
        self._memory_key: tuple[str | None, str | None] = (None, None)
        self._memory_ready = False
        self._recovery_yoloe = False

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
            "memory_ready": self._memory_ready,
            "recovery_yoloe": self._recovery_yoloe,
            "device": str(self.florence.device),
        }

    def reset(self) -> None:
        self._seen_backend = None
        self._frame_idx = 0
        self._seen_version = -1
        self._seen_epoch = -1
        self._has_target = False
        self._lost_frames = 0
        self._last_slow = -10_000
        self._fps_ema = 0.0
        self._memory_key = (None, None)
        self._memory_ready = False
        self._recovery_yoloe = False
        self.yoloe.reset()
        self.cv.reset()

    def process_frame(self, frame_bgr: np.ndarray, cfg: RuntimeConfig) -> FrameResult:
        t0 = time.perf_counter()
        prompt_text = cfg.prompt
        prompts = parse_prompts(prompt_text)
        prompt_changed = cfg.prompt_version != self._seen_version
        find_requested = cfg.find_epoch != self._seen_epoch
        backend_changed = cfg.fast_backend != self._seen_backend
        if prompt_changed or find_requested or backend_changed:
            self._seen_backend = cfg.fast_backend
            self._seen_version = cfg.prompt_version
            self._seen_epoch = cfg.find_epoch
            self._has_target = False
            self.yoloe.reset()
            self.cv.reset()
            self._memory_ready = False
            self._recovery_yoloe = False

        due_interval = (
            cfg.slow_interval > 0
            and self._frame_idx > 0
            and (self._frame_idx - self._last_slow) >= cfg.slow_interval
        )
        lost = self._lost_frames >= cfg.lost_patience
        memory = self.memory_store.get(cfg.memory_id) if self.memory_store and cfg.memory_id else None
        memory_key = (cfg.memory_id, cfg.memory_view)
        if memory_key != self._memory_key:
            self._memory_key = memory_key
            self._memory_ready = False
        if memory is not None and not self._memory_ready:
            # A remembered crop can seed YOLOE before Florence is needed. For a
            # CV runtime this is a temporary reacquisition backend; successful
            # detections are handed back to CV on the following frame.
            sample_view = cfg.memory_view or "scene"
            sample = next((view for view in reversed(memory.views) if view.view == sample_view), memory.views[-1])
            image = imread_unicode(sample.image)
            if image is not None:
                self.yoloe.load()
                self.yoloe.set_image_prompt(image, memory.label, conf=cfg.conf)
                self._memory_ready = True
                self._has_target = True
                self._recovery_yoloe = cfg.fast_backend == "cv"
        bootstrap_dets: list[Detection] = []
        bootstrap_success = False
        bootstrap_ms = 0.0
        if bool(prompts) and cfg.fast_backend == "yoloe" and not self._has_target and not memory:
            # YOLOE's label vocabulary is the cheap first pass. Florence is
            # reserved for labels YOLOE cannot localise confidently.
            try:
                t_bootstrap = time.perf_counter()
                self.yoloe.load()
                self.yoloe.set_text_prompt(prompts)
                bootstrap_dets = self.yoloe.detect(frame_bgr, conf=cfg.conf)
                bootstrap_ms = (time.perf_counter() - t_bootstrap) * 1000
                best_score = max((float(det.score) for det in bootstrap_dets), default=0.0)
                bootstrap_success = bool(bootstrap_dets) and best_score >= max(float(cfg.conf), 0.55)
                if bootstrap_success:
                    self._has_target = True
                    self.cv.init(frame_bgr, bootstrap_dets)
                else:
                    self.yoloe.reset()
                    bootstrap_dets = []
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("YOLOE text bootstrap failed; using Florence: %s", exc)
                self.yoloe.reset()
        need_slow = bool(prompts) and not bootstrap_success and (
            (not self._has_target)
            or lost
            or due_interval
            or ((prompt_changed or find_requested) and not self._memory_ready)
        )

        slow_dets: list[Detection] = []
        slow_ms = 0.0
        path = "idle"
        message = ""

        recovery_requested = lost and cfg.fast_backend == "cv"
        if (cfg.fast_backend == "yoloe" or recovery_requested or self._recovery_yoloe) and not self.yoloe.ready:
            try:
                self.yoloe.load()
            except (FileNotFoundError, ImportError):
                if cfg.fast_backend == "yoloe":
                    raise
                # CV remains usable without YOLO weights. Keep upstream's
                # YOLO-assisted reacquisition when installed, otherwise FIND
                # with Florence and initialize CV on the fresh frame.
                recovery_requested = self._recovery_yoloe = False
                LOGGER.warning("YOLOE recovery unavailable; reacquiring with Florence + CV")

        if need_slow:
            t_slow = time.perf_counter()
            slow_dets = self.florence.find(frame_bgr, prompts)
            slow_ms = (time.perf_counter() - t_slow) * 1000
            self._last_slow = self._frame_idx
            if slow_dets:
                self.cv.init(frame_bgr, slow_dets)
                if cfg.fast_backend == "yoloe" or recovery_requested or self._recovery_yoloe:
                    self.yoloe.imgsz = cfg.imgsz
                    self.yoloe.conf = cfg.conf
                    self.yoloe.set_visual_prompt(frame_bgr, slow_dets, conf=cfg.conf)
                    self._recovery_yoloe = cfg.fast_backend == "cv"
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
            if bootstrap_success:
                fast_dets = bootstrap_dets
                path = "fast"
            elif (cfg.fast_backend == "yoloe" or self._recovery_yoloe) and self.yoloe.has_prompt:
                if path == "slow":
                    fast_dets = self.yoloe.detect(frame_bgr, conf=cfg.conf)
                else:
                    fast_dets = self.yoloe.track(frame_bgr, conf=cfg.conf, persist=True)
                    path = "fast"
                if fast_dets and self._recovery_yoloe:
                    self.cv.init(frame_bgr, fast_dets)
                    self._recovery_yoloe = False
            else:
                fast_dets = self.cv.update(frame_bgr)
                if path != "slow":
                    path = "fast"
        fast_ms = bootstrap_ms + (time.perf_counter() - t_fast) * 1000

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
        cap = None
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
        finally:
            if cap is not None:
                cap.release()
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
