from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

from .types import Detection

LOGGER = logging.getLogger("find_and_track.florence")

FLORENCE_IDS = (
    "florence-community/Florence-2-large",
    "microsoft/Florence-2-large",
)
OVD = "<OPEN_VOCABULARY_DETECTION>"
GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"


def parse_prompts(text: str) -> list[str]:
    raw = (text or "").replace("|", ";").replace("\n", ";")
    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _polygon_to_xyxy(poly) -> np.ndarray | None:
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if pts.size < 4:
        return None
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def _parse_task_result(parsed: dict, task: str, fallback_label: str) -> list[tuple[np.ndarray, str]]:
    block = parsed.get(task, parsed)
    if not isinstance(block, dict):
        return []
    out: list[tuple[np.ndarray, str]] = []
    bboxes = block.get("bboxes") or []
    labels = block.get("bboxes_labels") or block.get("labels") or []
    for i, box in enumerate(bboxes):
        if box is None or len(box) < 4:
            continue
        label = str(labels[i]) if i < len(labels) and labels[i] else fallback_label
        out.append((np.array(box, dtype=np.float32), label))
    polygons = block.get("polygons") or []
    poly_labels = block.get("polygons_labels") or block.get("labels") or []
    for i, poly in enumerate(polygons):
        xyxy = _polygon_to_xyxy(poly)
        if xyxy is None:
            continue
        label = str(poly_labels[i]) if i < len(poly_labels) and poly_labels[i] else fallback_label
        out.append((xyxy, label))
    return out


class FlorenceFinder:
    """Slow-path open-vocabulary finder (Florence-2-large)."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.loaded_id: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        if self.ready:
            return
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        ids = (self.model_id,) if self.model_id else FLORENCE_IDS
        last_err: Exception | None = None
        for model_id in ids:
            for extra in ({"attn_implementation": "sdpa"}, {}):
                for local_only in (True, False):
                    try:
                        LOGGER.info(
                            "Loading Florence-2 from %s on %s extra=%s local_files_only=%s",
                            model_id,
                            self.device,
                            extra,
                            local_only,
                        )
                        kwargs = {"dtype": self.dtype, "local_files_only": local_only, **extra}
                        if self.device.type == "cuda":
                            kwargs["device_map"] = {"": self.device.index or 0}
                        self.model = Florence2ForConditionalGeneration.from_pretrained(model_id, **kwargs)
                        if "device_map" not in kwargs:
                            self.model = self.model.to(self.device)
                        self.model.eval()
                        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
                        self.loaded_id = model_id
                        LOGGER.info("Florence-2 ready: %s", model_id)
                        return
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        LOGGER.warning("Failed to load %s extra=%s local=%s: %s", model_id, extra, local_only, exc)
                        self.model = None
                        self.processor = None
        raise RuntimeError(f"Unable to load Florence-2-large: {last_err}") from last_err

    @torch.inference_mode()
    def _run_task(self, image: Image.Image, task: str, text_input: str, beams: int) -> dict:
        prompt = f"{task}{text_input}"
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
                continue
            if getattr(value, "dtype", None) in (torch.int32, torch.int64, torch.long):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value.to(self.device, dtype=self.dtype)
        try:
            generated_ids = self.model.generate(
                **moved,
                max_new_tokens=1024,
                num_beams=max(1, int(beams)),
                do_sample=False,
            )
        except torch.cuda.OutOfMemoryError:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            generated_ids = self.model.generate(
                **moved,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image.size,
        )

    def find(
        self,
        frame_bgr: np.ndarray,
        prompts: Iterable[str],
        beams: int = 3,
    ) -> list[Detection]:
        if not self.ready:
            self.load()
        if isinstance(prompts, str):
            names = parse_prompts(prompts)
        else:
            names = [str(p).strip() for p in prompts if str(p).strip()]
        if not names:
            return []

        image = _bgr_to_pil(frame_bgr)
        h, w = frame_bgr.shape[:2]
        found: list[Detection] = []
        for class_id, name in enumerate(names):
            boxes: list[tuple[np.ndarray, str]] = []
            try:
                parsed = self._run_task(image, OVD, name, beams)
                boxes = _parse_task_result(parsed, OVD, name)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("OVD failed for %r: %s", name, exc)
            if not boxes:
                try:
                    parsed = self._run_task(image, GROUNDING, name, max(1, beams - 1))
                    boxes = _parse_task_result(parsed, GROUNDING, name)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Phrase grounding failed for %r: %s", name, exc)
            area_img = float(w * h)
            for xyxy, label in boxes:
                det = Detection(
                    xyxy=xyxy,
                    label=label or name,
                    score=1.0,
                    source="slow",
                    class_id=class_id,
                ).clip(w, h)
                ratio = det.area / max(area_img, 1.0)
                if ratio < 0.0004 or ratio > 0.98:
                    continue
                found.append(det)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return found
