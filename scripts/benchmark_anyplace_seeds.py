"""Run reproducible AnyPlace seed sweeps on one recorded RGB-D cloud pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.place.anyplace_runtime import AnyPlaceRuntime


def rotation_angle_deg(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True,
                        help="Recorded serve_anyplace.py request/response JSON")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "_models/anyplace/anyplace_ckpts/anyplace_multitask/model.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--init-current-orientation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite previous AnyPlace evidence")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be unique nonnegative integers")

    recorded = json.loads(args.request.read_text(encoding="utf-8"))
    request = recorded.get("request", recorded)
    parent = np.asarray(request["parent"], dtype=np.float32)
    child = np.asarray(request["child"], dtype=np.float32)
    support_z = float(np.percentile(parent[:, 2], 50.0))
    runtime = AnyPlaceRuntime(args.checkpoint, root=ROOT)
    rows = []
    for seed in args.seeds:
        result = runtime.infer(parent, child, seed=seed, candidates=args.candidates,
                               iterations=args.iterations,
                               init_current_orientation=args.init_current_orientation)
        diagnostics = []
        for index, transform in enumerate(result["transforms"]):
            relative = np.asarray(transform, dtype=np.float64)
            final = child @ relative[:3, :3].T + relative[:3, 3]
            angle = rotation_angle_deg(relative[:3, :3])
            bottom_gap = float(np.percentile(final[:, 2], 0.5) - support_z)
            diagnostics.append({
                "index": index,
                "rotation_change_deg": angle,
                "observed_child_bottom_gap_m": bottom_gap,
                "passes_rotation_gate": angle <= 1.0,
                "passes_height_gate": -0.003 <= bottom_gap <= 0.012,
            })
        executable = [row for row in diagnostics
                      if row["passes_rotation_gate"] and row["passes_height_gate"]]
        rows.append({
            "seed": seed,
            "inference_s": result["inference_s"],
            "cuda_peak_allocated_bytes": result["cuda_peak_allocated_bytes"],
            "raw_candidates": len(diagnostics),
            "rotation_gate_candidates": sum(row["passes_rotation_gate"] for row in diagnostics),
            "height_gate_candidates": sum(row["passes_height_gate"] for row in diagnostics),
            "joint_gate_candidates": len(executable),
            "best_rotation_change_deg": min(row["rotation_change_deg"] for row in diagnostics),
            "diagnostics": diagnostics,
        })
        print(json.dumps({key: value for key, value in rows[-1].items()
                          if key != "diagnostics"}), flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "anyplace_recorded_cloud_seed_sweep",
        "robot_commands": False,
        "physical_success": None,
        "request_source": str(args.request.resolve()),
        "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": runtime.weight_sha256,
        "parent_points": len(parent),
        "child_points": len(child),
        "support_z_from_parent_m": support_z,
        "init_current_orientation": args.init_current_orientation,
        "seeds": rows,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
