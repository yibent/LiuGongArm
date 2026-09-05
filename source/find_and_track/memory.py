"""Offline object memories for YOLOE visual prompting.

The memory layer deliberately stores small, inspectable JPEG references and JSON
metadata. It does not store model tensors, so memories survive model upgrades and
can be copied between the simulator and a real robot.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock

import cv2
import numpy as np


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass
class MemoryView:
    view: str
    image: str
    bbox: list[float]
    score: float = 1.0
    pose: dict = field(default_factory=dict)
    # New memories retain the source frame so YOLOE can receive the original
    # box coordinates. Older JSON files omit this field and use the crop.
    reference_image: str | None = None


@dataclass
class ObjectMemory:
    memory_id: str
    label: str
    created_at: float
    views: list[MemoryView] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ObjectMemoryStore:
    """Filesystem-backed memory store with atomic metadata writes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @staticmethod
    def _normalise(value: str) -> str:
        return " ".join(str(value).casefold().split())

    def _read(self, path: Path) -> ObjectMemory | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return ObjectMemory(
                memory_id=str(raw["memory_id"]),
                label=str(raw["label"]),
                created_at=float(raw["created_at"]),
                views=[MemoryView(**item) for item in raw.get("views", [])],
                aliases=[str(item) for item in raw.get("aliases", [])],
                metadata=dict(raw.get("metadata", {})),
            )
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def get(self, memory_id: str) -> ObjectMemory | None:
        with self._lock:
            return self._read(self.root / f"{memory_id}.json")

    def list(self) -> list[dict]:
        with self._lock:
            result = []
            for path in sorted(self.root.glob("*.json")):
                item = self._read(path)
                if item is not None:
                    result.append(asdict(item))
            return result

    def find(self, label: str) -> list[ObjectMemory]:
        wanted = self._normalise(label)
        with self._lock:
            result = []
            for path in self.root.glob("*.json"):
                item = self._read(path)
                if item and (self._normalise(item.label) == wanted or
                             any(self._normalise(alias) == wanted for alias in item.aliases)):
                    result.append(item)
            return sorted(result, key=lambda item: item.created_at, reverse=True)

    def remember(
        self,
        label: str,
        frames: dict[str, np.ndarray],
        detections: dict[str, object],
        *,
        aliases: list[str] | None = None,
        metadata: dict | None = None,
        memory_id: str | None = None,
        padding: float = 0.15,
    ) -> ObjectMemory:
        """Persist multi-view object crops and source boxes as one memory."""
        label = str(label).strip()
        if not label:
            raise ValueError("memory label must not be empty")
        if not frames or not detections:
            raise ValueError("memory capture requires frames and detections")
        mid = memory_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        item_dir = self.root / mid
        item_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read(self.root / f"{mid}.json") if memory_id else None
        if existing is not None and self._normalise(existing.label) != self._normalise(label):
            raise ValueError(f"memory {mid!r} already belongs to {existing.label!r}")
        views: list[MemoryView] = list(existing.views) if existing is not None else []
        for view, frame in frames.items():
            det = detections.get(view)
            if det is None:
                continue
            bbox = np.asarray(_field(det, "xyxy"), dtype=np.float32).reshape(-1)
            if bbox.size != 4:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
            sx1, sy1 = int(max(0, x1 - bw * padding)), int(max(0, y1 - bh * padding))
            sx2, sy2 = int(min(w, x2 + bw * padding)), int(min(h, y2 + bh * padding))
            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue
            image_path = item_dir / f"{view}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:4]}.jpg"
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                continue
            encoded.tofile(str(image_path))
            reference_path = item_dir / f"{view}-reference-{int(time.time() * 1000)}-{uuid.uuid4().hex[:4]}.jpg"
            reference_ok, reference_encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            reference_image = None
            if reference_ok:
                reference_encoded.tofile(str(reference_path))
                reference_image = str(reference_path)
            views.append(MemoryView(
                view=view,
                image=str(image_path),
                bbox=[x1, y1, x2, y2],
                score=float(_field(det, "score", 1.0)),
                pose=dict(_field(det, "pose", {}) or {}),
                reference_image=reference_image,
            ))
        if not views:
            raise ValueError("no valid detections to remember")
        memory = ObjectMemory(
            mid,
            existing.label if existing is not None else label,
            existing.created_at if existing is not None else time.time(),
            views,
            sorted(set((existing.aliases if existing is not None else []) + (aliases or []))),
            {**(existing.metadata if existing is not None else {}), **(metadata or {})},
        )
        target = self.root / f"{mid}.json"
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(asdict(memory), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, target)
        return memory

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            item = self._read(self.root / f"{memory_id}.json")
            if item is None:
                return False
            for view in item.views:
                try:
                    Path(view.image).unlink(missing_ok=True)
                    if view.reference_image:
                        Path(view.reference_image).unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                (self.root / f"{memory_id}.json").unlink(missing_ok=True)
                (self.root / memory_id).rmdir()
            except OSError:
                pass
            return True
