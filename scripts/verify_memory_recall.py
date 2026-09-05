"""Verify multi-view object memory capture and YOLOE image-prompt recall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from find_and_track import Detection, FindTrackPipeline, ObjectMemoryStore, RuntimeConfig  # noqa: E402
from find_and_track.pipeline import imread_unicode  # noqa: E402


class _FlorenceGuard:
    """A ready facade that fails if memory recall accidentally invokes Florence."""

    ready = True
    loaded_id = "guard"
    device = "cpu"

    def load(self) -> None:
        return

    def find(self, *_args, **_kwargs):
        raise AssertionError("memory recall unexpectedly invoked Florence")


def _bbox(value: str) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be x1,y1,x2,y2") from exc
    if len(values) != 4 or not all(np.isfinite(values)) or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError("bbox must contain an increasing finite xyxy box")
    return np.asarray(values, dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-image", type=Path, required=True)
    parser.add_argument("--scene-bbox", type=_bbox, required=True)
    parser.add_argument("--wrist-image", type=Path, required=True)
    parser.add_argument("--wrist-bbox", type=_bbox, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-view", default="scene")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scene = imread_unicode(args.scene_image)
    wrist = imread_unicode(args.wrist_image)
    if scene is None or wrist is None:
        raise FileNotFoundError("Both memory source images must be readable")

    store = ObjectMemoryStore(output / "memory")
    memory = store.remember(
        args.label,
        {"scene": scene, "wrist": wrist},
        {
            "scene": Detection(args.scene_bbox, args.label, score=1.0),
            "wrist": Detection(args.wrist_bbox, args.label, score=1.0),
        },
        aliases=[args.label.replace(" ", "_")],
        metadata={"source": "isaac_generalization_capture", "views": ["scene", "wrist"]},
    )

    pipeline = FindTrackPipeline(finder=_FlorenceGuard(), memory_store=store)
    config = RuntimeConfig(
        prompt=args.label,
        fast_backend="yoloe",
        memory_id=memory.memory_id,
        memory_view=args.memory_view,
        conf=0.25,
    )
    result = pipeline.process_frame(scene, config)
    detections = [
        {"label": det.label, "score": float(det.score), "xyxy": det.xyxy.tolist(), "track_id": det.track_id}
        for det in result.fast
    ]
    report = {
        "protocol": 1,
        "label": args.label,
        "memory_id": memory.memory_id,
        "memory_views": [view.view for view in memory.views],
        "memory_files": [view.image for view in memory.views],
        "reference_files": [view.reference_image for view in memory.views],
        "memory_ready": pipeline.status()["memory_ready"],
        "florence_guard": "not_called",
        "yoloe_ready": pipeline.status()["yoloe"],
        "path": result.stats.path,
        "n_fast": len(result.fast),
        "detections": detections,
        "passed": bool(pipeline.status()["memory_ready"] and result.stats.path == "fast" and result.fast),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
