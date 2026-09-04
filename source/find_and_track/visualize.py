from __future__ import annotations

import cv2
import numpy as np

from .types import Detection, FrameStats

SLOW = (62, 138, 255)  # BGR orange
FAST = (176, 224, 62)  # BGR mint
HUD_BG = (18, 16, 12)
TEXT = (236, 232, 224)
MUTED = (170, 168, 160)
LOST = (90, 90, 255)

PALETTE = [
    (176, 224, 62),
    (255, 196, 72),
    (80, 200, 255),
    (180, 120, 255),
    (70, 230, 180),
    (90, 160, 255),
    (220, 180, 255),
    (120, 255, 210),
]


def _color_for(det: Detection) -> tuple[int, int, int]:
    if det.source == "slow":
        return SLOW
    if det.track_id is not None:
        return PALETTE[int(det.track_id) % len(PALETTE)]
    return FAST


def _draw_box(img: np.ndarray, det: Detection, dashed: bool = False) -> None:
    x1, y1, x2, y2 = det.xyxy.astype(int)
    color = _color_for(det)
    if dashed:
        _dashed_rect(img, (x1, y1), (x2, y2), color, 2)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
    tag = "FIND" if det.source == "slow" else "TRACK"
    ident = f"#{det.track_id}" if det.track_id is not None else ""
    label = f"{tag} {ident} {det.label} {det.score:.2f}".strip()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    tx, ty = x1, max(0, y1 - 6)
    cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 8, ty + 2), color, -1)
    cv2.putText(img, label, (tx + 4, ty - 2), font, scale, (12, 12, 12), 1, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, thickness, gap=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness, cv2.LINE_AA)


def draw(frame_bgr: np.ndarray, slow: list[Detection], fast: list[Detection], stats: FrameStats) -> np.ndarray:
    vis = frame_bgr.copy()
    for det in slow:
        _draw_box(vis, det, dashed=True)
    for det in fast:
        _draw_box(vis, det, dashed=False)
    _hud(vis, stats)
    return vis


def _hud(img: np.ndarray, stats: FrameStats) -> None:
    h, w = img.shape[:2]
    bar_h = 42
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    path = stats.path.upper()
    if stats.lost:
        path_color = LOST
        path = "LOST"
    elif stats.path == "slow":
        path_color = SLOW
    elif stats.path == "fast":
        path_color = FAST
    else:
        path_color = MUTED
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(img, (18, 21), 6, path_color, -1, cv2.LINE_AA)
    line = (
        f"{path}   FPS {stats.fps:.1f}   slow {stats.slow_ms:.0f}ms   "
        f"fast {stats.fast_ms:.0f}ms   {stats.backend}   "
        f"find {stats.n_slow}  track {stats.n_fast}"
    )
    cv2.putText(img, line, (34, 18), font, 0.48, TEXT, 1, cv2.LINE_AA)
    prompt = stats.prompt if stats.prompt else "(no prompt)"
    cv2.putText(img, f"prompt: {prompt[:80]}", (34, 34), font, 0.45, MUTED, 1, cv2.LINE_AA)
