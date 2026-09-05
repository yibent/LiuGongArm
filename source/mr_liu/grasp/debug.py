"""Failure-oriented RGB-D visualizations for FineGrasp development."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mr_liu.grasp.contracts import RGBDObservation, TargetSpec, TargetSegmenter


class FineGraspDebugRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trace.jsonl").unlink(missing_ok=True)
        self.events: list[dict[str, object]] = []

    def trace(self, event: dict[str, object]) -> None:
        self.events.append(dict(event))
        with (self.output_dir / "trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=_json_default) + "\n")

    def record_segmentation(
        self,
        observation: RGBDObservation,
        mask: np.ndarray | None,
        target: TargetSpec,
    ) -> None:
        stem = f"obs_{observation.sequence:04d}"
        rgb = np.asarray(observation.rgb, dtype=np.uint8)
        cv2.imwrite(str(self.output_dir / f"{stem}_rgb.png"), rgb[:, :, ::-1])
        depth = np.asarray(observation.depth_m, dtype=np.float32)
        finite = np.isfinite(depth) & (depth > 0)
        depth_vis = np.zeros(depth.shape, dtype=np.uint8)
        if finite.any():
            low, high = np.percentile(depth[finite], [2, 98])
            depth_vis[finite] = np.clip((depth[finite] - low) * 255.0 / max(high - low, 1e-6), 0, 255)
        cv2.imwrite(
            str(self.output_dir / f"{stem}_depth.png"),
            cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_TURBO),
        )
        mask_bool = np.zeros(depth.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        np.savez_compressed(
            self.output_dir / f"{stem}_rgbd.npz", rgb=rgb, depth_m=depth,
            mask=mask_bool, intrinsics=observation.intrinsics.matrix,
            T_base_camera=observation.T_base_camera,
            timestamp_s=observation.timestamp_s,
            T_base_ee=observation.T_base_ee,
            T_ee_camera=observation.T_ee_camera,
            robot_self_mask=(np.zeros(depth.shape, bool) if observation.metadata.get("robot_self_mask") is None
                             else np.asarray(observation.metadata["robot_self_mask"], bool)),
            has_robot_self_mask=observation.metadata.get("robot_self_mask") is not None,
        )
        overlay = rgb.copy()
        overlay[mask_bool] = (0.35 * overlay[mask_bool] + 0.65 * np.asarray([40, 255, 40])).astype(np.uint8)
        cv2.imwrite(str(self.output_dir / f"{stem}_overlay.png"), overlay[:, :, ::-1])
        metadata = {
            "sequence": observation.sequence,
            "timestamp_s": observation.timestamp_s,
            "target": target.object_id,
            "mask_pixels": int(mask_bool.sum()),
            "valid_mask_depth_pixels": int((mask_bool & finite).sum()),
            "T_base_camera": observation.T_base_camera.tolist(),
        }
        (self.output_dir / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


class DebugSegmenter:
    def __init__(self, segmenter: TargetSegmenter, recorder: FineGraspDebugRecorder) -> None:
        self.segmenter = segmenter
        self.recorder = recorder

    def segment(self, observation: RGBDObservation, target: TargetSpec) -> np.ndarray | None:
        mask = self.segmenter.segment(observation, target)
        self.recorder.record_segmentation(observation, mask, target)
        diagnostic = getattr(self.segmenter, "last_diagnostic", None)
        if diagnostic:
            self.recorder.trace({"event": "identity", "sequence": observation.sequence, **diagnostic})
        return mask


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
