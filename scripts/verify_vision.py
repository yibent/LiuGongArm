"""Real-model, offline smoke test. This is NOT an accuracy or grasp benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("HF_HOME", str(ROOT / "_models" / "hf_cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from find_and_track.pipeline import FindTrackPipeline  # noqa: E402
from find_and_track.types import RuntimeConfig  # noqa: E402


def detection_dict(det) -> dict:
    return {**asdict(det), "xyxy": det.xyxy.tolist()}


def run(args, out: Path) -> dict:
    frames = []
    cap = cv2.VideoCapture(str(args.source))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open sample: {args.source}")
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) < 3:
        raise RuntimeError("At least three video frames required to test TRACK beyond the FIND frame")
    pipe = FindTrackPipeline(device=args.device)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "kind": "offline integration smoke; no ground-truth accuracy or grasp claims",
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "prompt": args.prompt, "frames": len(frames),
        "python": sys.executable, "torch": torch.__version__,
        "device": str(pipe.florence.device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "versions": {p: importlib.metadata.version(p) for p in (
            "transformers", "ultralytics", "torchvision", "opencv-python", "huggingface-hub", "lap")},
        "backends": {},
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for backend in ("yoloe", "cv") if args.backend == "all" else (args.backend,):
        cfg = RuntimeConfig(prompt=args.prompt, fast_backend=backend, slow_interval=0)
        t0 = time.perf_counter()
        pipe.load(backend)
        load_ms = (time.perf_counter() - t0) * 1000
        pipe.reset()
        writer = pipe._open_writer(out / f"{backend}.mp4", frames[0].shape, 15)
        if not writer.isOpened():
            raise RuntimeError("Cannot create annotated MP4")
        rows = []
        try:
            for frame in frames:
                t0 = time.perf_counter()
                result = pipe.process_frame(frame, cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                rows.append({"stats": asdict(result.stats), "wall_ms": elapsed_ms,
                             "slow": [detection_dict(d) for d in result.slow],
                             "fast": [detection_dict(d) for d in result.fast]})
                writer.write(result.image)
                if len(rows) == 1:
                    pipe._write_image(out / f"{backend}_first.jpg", result.image)
                print(f"{backend} frame={result.stats.frame_idx} path={result.stats.path} "
                      f"find={len(result.slow)} track={len(result.fast)} wall={elapsed_ms:.1f}ms", flush=True)
        finally:
            writer.release()
        fast = [r for r in rows[1:] if r["stats"]["path"] == "fast"]
        latencies = [r["wall_ms"] for r in fast]
        tracks_present = sum(bool(r["fast"]) for r in fast)
        data = {
            "load_ms": load_ms, "first_frame_ms": rows[0]["wall_ms"],
            "fast_frames": len(fast), "fast_frames_with_detections": tracks_present,
            "fast_wall_median_ms": float(np.median(latencies)) if latencies else None,
            "fast_wall_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "cv_mil_active": sum(t.mil is not None for t in pipe.cv.tracks) if backend == "cv" else None,
            "passed": bool(rows[0]["slow"]) and len(fast) == len(frames) - 1 and tracks_present == len(fast),
            "rows": rows,
        }
        report["backends"][backend] = data
    report["models"] = pipe.status()
    report["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    report["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else 0
    report["passed"] = all(b["passed"] for b in report["backends"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "samples" / "bus_long.mp4")
    parser.add_argument("--prompt", default="bus")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--backend", choices=["all", "yoloe", "cv"], default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "vision_validation" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    # Also block raw TCP connections, not just HF downloads, while loading/inferencing.
    def deny_network(*_args, **_kwargs):
        raise OSError("Network disabled by offline vision verification")
    try:
        with patch.object(socket.socket, "connect", deny_network), patch.object(socket.socket, "connect_ex", deny_network):
            report = run(args, args.output)
    except Exception as exc:
        report = {"passed": False, "error": repr(exc)}
        logging.exception("Vision smoke failed")
    dest = args.output / "report.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {dest.resolve()} passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
