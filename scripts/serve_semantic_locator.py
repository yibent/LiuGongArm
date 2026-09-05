"""Local, lazy Florence -> optional YOLOE localization; never controls a robot."""
from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
from find_and_track.pipeline import FindTrackPipeline


def locate(body):
    label = str(body["label"]).strip()
    if not label or len(label) > 256:
        raise ValueError("A single nonempty label of at most 256 characters is required")
    mode = body.get("mode", "florence_yoloe")
    if mode not in ("florence", "florence_yoloe"):
        raise ValueError("Unknown localization mode")
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(body["jpeg"], validate=True), np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size > 16_000_000:
        raise ValueError("Invalid/oversized JPEG")
    started = time.monotonic()
    pipe = FindTrackPipeline()
    try:
        pipe.load("yoloe" if mode == "florence_yoloe" else "cv")
        found = pipe.florence.find(frame, [label])
        slow_boxes = [d.xyxy.tolist() for d in found]
        if found and mode == "florence_yoloe":
            pipe.yoloe.set_visual_prompt(frame, found)
            found = pipe.yoloe.detect(frame)
        boxes = [{"xyxy": d.xyxy.tolist(), "score": float(d.score)} for d in found]
        if body.get("segment", False):
            if mode != "florence" or len(boxes) > 8:
                raise ValueError("Segmentation requires Florence mode and <=8 candidates")
            for item in boxes:
                item["segmentation"] = pipe.florence.segment_box(frame, item["xyxy"])
        description = pipe.florence.describe(frame) if body.get("describe", False) else None
        return {"protocol": 1, "sequence": int(body["sequence"]), "label": label,
                "mode": mode, "florence_boxes": slow_boxes,
                "boxes": boxes, "description": description,
                "elapsed_s": time.monotonic() - started}
    finally:
        # Coarse TRACK is CPU optical flow; release GPU weights for Isaac and
        # the existing GraspGenX process, including after failed localization.
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, value):
        payload = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"service": "busagent-semantic-locator", "protocol": 1,
                               "load_policy": "lazy; evict weights after every FIND"})
        else:
            self.respond(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/locate":
            return self.respond(404, {"error": "not_found"})
        try:
            count = int(self.headers.get("Content-Length", 0))
            if not 0 < count <= 8_000_000:
                raise ValueError("Invalid request length")
            self.respond(200, locate(json.loads(self.rfile.read(count))))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.respond(500, {"error": type(exc).__name__, "message": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5570)
    args = parser.parse_args()
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
