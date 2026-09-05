"""Masked pyramidal LK tracking with forward/backward and motion consistency.

No MIL/template fallback: failure means stop/re-localize, never reuse a box.
"""
from __future__ import annotations

import cv2
import numpy as np


class OpticalFlowTracker:
    def __init__(self, min_points=3, max_fb_error_px=1.0):
        self.min_points = min_points
        self.max_fb_error_px = max_fb_error_px
        self.gray = self.points = self.mask = None
        self.diagnostic = {}

    def initialize(self, rgb, mask):
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.mask = np.asarray(mask, np.uint8).copy()
        feature_mask = cv2.dilate(self.mask, np.ones((3, 3), np.uint8))
        self.points = cv2.goodFeaturesToTrack(self.gray, 80, 0.003, 2, mask=feature_mask, blockSize=3)
        count = 0 if self.points is None else len(self.points)
        self.diagnostic = {"flow_points": count, "flow_reason": "initialized"}
        return count >= self.min_points

    def update(self, rgb):
        if self.points is None or len(self.points) < self.min_points:
            self.diagnostic = {"flow_reason": "insufficient_features"}
            return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        new, status, err = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None,
                                                 winSize=(21, 21), maxLevel=3)
        if new is None:
            self.diagnostic = {"flow_reason": "no_forward_flow"}
            return None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, new, None,
                                                      winSize=(21, 21), maxLevel=3)
        if back is None:
            self.diagnostic = {"flow_reason": "no_backward_flow"}
            return None
        fb = np.linalg.norm(back - self.points, axis=2).reshape(-1)
        valid = ((status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
                 & (fb <= self.max_fb_error_px) & (err.reshape(-1) < 35))
        old, current = self.points.reshape(-1, 2)[valid], new.reshape(-1, 2)[valid]
        if len(old) < self.min_points:
            self.diagnostic = {"flow_reason": "forward_backward_rejected", "flow_points": len(old)}
            return None
        shifts = current - old
        shift = np.median(shifts, axis=0)
        inliers = np.linalg.norm(shifts - shift, axis=1) <= 1.5
        if inliers.sum() < self.min_points or inliers.mean() < 0.6 or np.linalg.norm(shift) > 40:
            self.diagnostic = {"flow_reason": "inconsistent_motion", "flow_points": int(inliers.sum())}
            return None
        # Fixed external camera + short intervals: robust translation is less
        # underconstrained than fitting scale/rotation to three tiny-object corners.
        shift = np.median(shifts[inliers], axis=0)
        affine = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        mask = cv2.warpAffine(self.mask, affine, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST)
        self.gray, self.points, self.mask = gray, current[inliers].reshape(-1, 1, 2), mask
        self.diagnostic = {"flow_reason": "lk_forward_backward", "flow_points": int(inliers.sum()),
                           "flow_shift_px": shift.tolist(), "flow_fb_median_px": float(np.median(fb[valid]))}
        return mask.astype(bool)
