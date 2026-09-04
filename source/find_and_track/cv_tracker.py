from __future__ import annotations

import logging

import cv2
import numpy as np

from .types import Detection

LOGGER = logging.getLogger("find_and_track.cv")


def _crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


class _SingleTrack:
    def __init__(self, frame_bgr: np.ndarray, det: Detection, track_id: int):
        self.track_id = track_id
        self.label = det.label
        self.class_id = det.class_id
        self.score = float(det.score)
        self.xyxy = det.xyxy.astype(np.float32)
        self.alive = True
        self.misses = 0
        self.template = _crop(frame_bgr, self.xyxy)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.orb = cv2.ORB_create(nfeatures=500)
        self.des = None
        if self.template is not None:
            tmpl_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            _kp, self.des = self.orb.detectAndCompute(tmpl_gray, None)
        self.mil = None
        try:
            ww, hh = int(self.xyxy[2] - self.xyxy[0]), int(self.xyxy[3] - self.xyxy[1])
            mil = cv2.TrackerMIL_create()
            if mil.init(frame_bgr, (int(self.xyxy[0]), int(self.xyxy[1]), max(2, ww), max(2, hh))):
                self.mil = mil
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("MIL init failed: %s", exc)
        x1, y1, x2, y2 = self.xyxy.astype(int)
        roi = gray[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        pts = None
        if roi.size > 16:
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=48, qualityLevel=0.01, minDistance=4)
            if pts is not None:
                pts = pts.copy()
                pts[:, 0, 0] += x1
                pts[:, 0, 1] += y1
        self.pts = pts

    def update(self, frame_bgr: np.ndarray) -> Detection | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = frame_bgr.shape[:2]
        box = None
        score = 0.2

        if self.mil is not None:
            ok, xywh = self.mil.update(frame_bgr)
            if ok:
                x, y, bw, bh = xywh
                mil_box = np.array([x, y, x + bw, y + bh], dtype=np.float32)
                mil_score = 0.55
                if hasattr(self.mil, "getTrackingScore"):
                    try:
                        mil_score = float(np.clip(self.mil.getTrackingScore(), 0.05, 0.99))
                    except Exception:  # noqa: BLE001
                        pass
                box, score = mil_box, max(score, mil_score)

        if self.pts is not None and len(self.pts) >= 4:
            nxt, st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.pts, None, winSize=(21, 21), maxLevel=3)
            if nxt is not None and st is not None:
                good_new = nxt[st.flatten() == 1]
                good_old = self.pts[st.flatten() == 1]
                if len(good_new) >= 4:
                    shift = np.median(good_new - good_old, axis=0).reshape(2)
                    flowed = self.xyxy.copy()
                    flowed[0] += shift[0]
                    flowed[2] += shift[0]
                    flowed[1] += shift[1]
                    flowed[3] += shift[1]
                    if box is None:
                        box, score = flowed, 0.45
                    self.pts = good_new.reshape(-1, 1, 2)
                else:
                    self.pts = None
        else:
            self.pts = None

        if box is None and self.template is not None:
            box = self._match_template(frame_bgr)
            if box is not None:
                score = 0.4

        if box is None and self.des is not None:
            box = self._match_orb(gray)
            if box is not None:
                score = 0.5

        self.prev_gray = gray
        if box is None:
            self.misses += 1
            self.alive = self.misses < 8
            return None

        det = Detection(
            xyxy=box.astype(np.float32),
            label=self.label,
            score=float(score),
            track_id=self.track_id,
            source="fast",
            class_id=self.class_id,
        ).clip(w, h)
        if det.area < 16:
            self.misses += 1
            self.alive = self.misses < 8
            return None
        self.xyxy = det.xyxy
        self.score = det.score
        self.misses = 0
        self.alive = True
        x1, y1, x2, y2 = det.xyxy.astype(int)
        roi = gray[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if roi.size > 16:
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=48, qualityLevel=0.01, minDistance=4)
            if pts is not None:
                pts = pts.copy()
                pts[:, 0, 0] += x1
                pts[:, 0, 1] += y1
                self.pts = pts
        crop = _crop(frame_bgr, det.xyxy)
        if crop is not None and crop.size > 64:
            self.template = crop
        return det

    def _search_roi(self, shape, pad: float = 1.8) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        x1, y1, x2, y2 = self.xyxy
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        nw, nh = bw * pad, bh * pad
        sx1 = int(np.clip(cx - nw / 2, 0, w - 1))
        sy1 = int(np.clip(cy - nh / 2, 0, h - 1))
        sx2 = int(np.clip(cx + nw / 2, 1, w))
        sy2 = int(np.clip(cy + nh / 2, 1, h))
        return sx1, sy1, sx2, sy2

    def _match_template(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if self.template is None:
            return None
        sx1, sy1, sx2, sy2 = self._search_roi(frame_bgr.shape)
        roi = frame_bgr[sy1:sy2, sx1:sx2]
        th, tw = self.template.shape[:2]
        if roi.shape[0] <= th or roi.shape[1] <= tw:
            return None
        res = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
        if max_v < 0.45:
            return None
        x, y = max_l
        return np.array([sx1 + x, sy1 + y, sx1 + x + tw, sy1 + y + th], dtype=np.float32)

    def _match_orb(self, gray: np.ndarray) -> np.ndarray | None:
        if self.des is None:
            return None
        sx1, sy1, sx2, sy2 = self._search_roi(gray.shape, pad=2.2)
        roi = gray[sy1:sy2, sx1:sx2]
        kp2, des2 = self.orb.detectAndCompute(roi, None)
        if des2 is None or len(kp2) < 6:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            pairs = matcher.knnMatch(self.des, des2, k=2)
        except cv2.error:
            return None
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 6:
            return None
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        xs, ys = dst[:, 0] + sx1, dst[:, 1] + sy1
        bw = max(8.0, self.xyxy[2] - self.xyxy[0])
        bh = max(8.0, self.xyxy[3] - self.xyxy[1])
        cx, cy = float(np.median(xs)), float(np.median(ys))
        return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32)


class CvFeatureTracker:
    """Fast-path fallback: MIL + optical flow + ORB / template matching."""

    def __init__(self):
        self.tracks: list[_SingleTrack] = []
        self._next_id = 1

    def init(self, frame_bgr: np.ndarray, detections: list[Detection]) -> None:
        self.tracks = []
        self._next_id = 1
        for det in detections:
            self.tracks.append(_SingleTrack(frame_bgr, det, self._next_id))
            self._next_id += 1

    def update(self, frame_bgr: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        alive: list[_SingleTrack] = []
        for track in self.tracks:
            det = track.update(frame_bgr)
            if det is not None:
                out.append(det)
            if track.alive:
                alive.append(track)
        self.tracks = alive
        return out

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    @property
    def has_prompt(self) -> bool:
        return any(t.alive for t in self.tracks)
