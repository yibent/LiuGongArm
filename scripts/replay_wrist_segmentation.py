"""Replay recorded RGB-D through SAM + geometric identity without Isaac."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", str(ROOT / "_models/graspgenx/checkpoints"))
os.environ.setdefault("GRASPGENX_GRIPPER_CFG_DIR", str(ROOT / "_vendor/GraspGenX/ext/gripper_descriptions"))
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.grasp.contracts import RGBDObservation, CameraIntrinsics, TargetSpec
from mr_liu.grasp.identity import GeometricIdentitySegmenter, RemotePromptSegmenter
from mr_liu.grasp.segmentation import SeededDepthSegmenter
from serve_graspgenx import ReproducibleServer


class LocalTransport:
    def __init__(self):
        self.server = object.__new__(ReproducibleServer)

    def _request(self, request):
        return self.server._segment(request)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    target = TargetSpec("replay", "object", tuple(report["initial_target_position_m"]))
    inner = RemotePromptSegmenter(LocalTransport(), SeededDepthSegmenter(depth_tolerance_m=0.010, max_radius_px=180))
    segmenter = GeometricIdentitySegmenter(inner)
    trace = report["trace"]
    hint = next((e["expected_center_base_m"] for e in trace if e.get("event") == "lift_tracking_seed"), None)
    files = sorted((args.run / "debug").glob("obs_*_rgbd.npz"))
    rows = []
    for index, path in enumerate(files):
        data = np.load(path, allow_pickle=False)
        depth, rgb, k = data["depth_m"], data["rgb"], data["intrinsics"]
        metadata = {}
        if index == len(files)-1 and hint is not None:
            metadata["tracking_hint_base_m"] = hint
        observation = RGBDObservation(index+1, float(data["timestamp_s"]), rgb, depth,
                                     CameraIntrinsics(depth.shape[1], depth.shape[0], k[0,0], k[1,1], k[0,2], k[1,2]),
                                     data["T_base_camera"], data["T_base_ee"], data["T_ee_camera"], metadata=metadata)
        mask = segmenter.segment(observation, target)
        row = {"frame": path.name, "pixels": int(mask.sum()) if mask is not None else 0,
               "model": inner.last_diagnostic, "identity": segmenter.last_diagnostic}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.run / "segmentation_replay.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
