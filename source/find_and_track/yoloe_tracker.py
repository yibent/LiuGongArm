from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .types import Detection

LOGGER = logging.getLogger("find_and_track.yoloe")


class YoloeVisualTracker:
    """Fast-path tracker: bake Florence boxes as YOLOE visual prompt embeddings, then track."""

    def __init__(self, weights: str | Path, device: str | None = None, imgsz: int = 640, conf: float = 0.25):
        self.weights = Path(weights)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.conf = conf
        self.model = None
        self.has_prompt = False
        self.class_names: list[str] = []

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.ready:
            return
        from ultralytics import YOLOE

        if not self.weights.exists():
            raise FileNotFoundError(f"YOLOE weights not found: {self.weights}")
        LOGGER.info("Loading YOLOE from %s on %s", self.weights, self.device)
        self.model = YOLOE(str(self.weights))
        try:
            self.model.to(self.device)
        except Exception:  # noqa: BLE001
            pass
        LOGGER.info("YOLOE ready")

    def reset(self) -> None:
        self.has_prompt = False
        self.class_names = []
        if self.model is not None:
            self.model.predictor = None

    def set_visual_prompt(self, frame_bgr: np.ndarray, detections: list[Detection], conf: float | None = None) -> None:
        if not detections:
            return
        if not self.ready:
            self.load()
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        unique_ids = sorted({int(d.class_id) for d in detections})
        remap = {old: new for new, old in enumerate(unique_ids)}
        names: list[str] = []
        for old in unique_ids:
            label = next((d.label for d in detections if int(d.class_id) == old), f"object{old}")
            names.append(label)

        bboxes = np.stack([d.xyxy.astype(np.float32) for d in detections], axis=0)
        cls = np.array([remap[int(d.class_id)] for d in detections], dtype=np.int32)
        visual_prompts = {"bboxes": bboxes, "cls": cls}
        conf = self.conf if conf is None else conf

        self.model.predictor = None
        self.model.predict(
            frame_bgr,
            visual_prompts=visual_prompts,
            refer_image=frame_bgr,
            predictor=YOLOEVPSegPredictor,
            verbose=False,
            save=False,
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
        )
        pe = getattr(self.model.model, "pe", None)
        if pe is not None:
            try:
                self.model.set_classes(names, pe)
            except Exception:  # noqa: BLE001
                self.model.model.names = names
        else:
            self.model.model.names = names
        self.class_names = names
        self.has_prompt = True
        self.model.predictor = None
        LOGGER.info("YOLOE visual prompt set: %s (%d boxes)", names, len(detections))

    def track(self, frame_bgr: np.ndarray, conf: float | None = None, persist: bool = True) -> list[Detection]:
        if not self.has_prompt or not self.ready:
            return []
        conf = self.conf if conf is None else conf
        results = self.model.track(
            frame_bgr,
            persist=persist,
            verbose=False,
            save=False,
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
        )
        return self._parse_results(results)

    def detect(self, frame_bgr: np.ndarray, conf: float | None = None) -> list[Detection]:
        if not self.has_prompt or not self.ready:
            return []
        conf = self.conf if conf is None else conf
        results = self.model.predict(
            frame_bgr,
            verbose=False,
            save=False,
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
        )
        return self._parse_results(results)

    def _parse_results(self, results) -> list[Detection]:
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        names = result.names if isinstance(result.names, dict) else {i: n for i, n in enumerate(result.names or [])}
        h, w = result.orig_shape[:2] if getattr(result, "orig_shape", None) is not None else (None, None)
        out: list[Detection] = []
        for i in range(len(xyxy)):
            class_id = int(clss[i])
            label = str(names.get(class_id, self.class_names[class_id] if class_id < len(self.class_names) else f"object{class_id}"))
            det = Detection(
                xyxy=xyxy[i].astype(np.float32),
                label=label,
                score=float(confs[i]),
                track_id=None if ids[i] is None else int(ids[i]),
                source="fast",
                class_id=class_id,
            )
            if w and h:
                det.clip(int(w), int(h))
            out.append(det)
        return out
