"""Run isolated official Franka PickPlace processes and aggregate outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path,
                        default=Path(r"D:\isaac\env_isaacsim60\python.exe"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=1,
                        help="One-based manifest case index")
    return parser.parse_args()


def values(case: dict, key: str, default: tuple[float, ...]) -> list[str]:
    return [str(value) for value in case.get(key, default)]


args = parse_args()
manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
cases = list(manifest["cases"])
cases = cases[args.start_index - 1:]
if args.limit is not None:
    cases = cases[: args.limit]
if args.output.exists() and any(args.output.iterdir()):
    raise SystemExit("Use a new batch output directory")
args.output.mkdir(parents=True, exist_ok=True)

runner = ROOT / "scripts" / "run_official_franka_pick_place_legacy.py"
batch_started = time.perf_counter()
outcomes: list[dict] = []

for index, case in enumerate(cases, start=1):
    case_dir = args.output / case["name"]
    command = [
        str(args.isaac_python), str(runner), "--output", str(case_dir),
        "--cube-position", *values(case, "cube_position", (0.3, 0.3, 0.02575)),
        "--cube-orientation", *values(case, "cube_orientation", (1.0, 0.0, 0.0, 0.0)),
        "--target-position", *values(case, "target_position", (-0.3, -0.3, 0.02575)),
        "--cube-size", *values(case, "cube_size", (0.0515, 0.0515, 0.0515)),
        "--mass-kg", str(case.get("mass_kg", 0.02)),
        "--target-platform-height", str(case.get("target_platform_height", 0.0)),
        "--target-platform-size", *values(case, "target_platform_size", (0.18, 0.18)),
        "--gripper-yaw-deg", str(case.get("gripper_yaw_deg", 0.0)),
    ]
    print(f"[{index}/{len(cases)}] {case['name']}: {case.get('object', 'object')}", flush=True)
    log_path = args.output / f"{case['name']}.console.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    report_path = case_dir / "official_franka_pick_place_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    outcome = {
        "name": case["name"],
        "object": case.get("object"),
        "exit_code": process.returncode,
        "elapsed_s": time.perf_counter() - started,
        "success": bool(process.returncode == 0 and report.get("physical_pick_place_observed")),
        "sequence_completed": report.get("sequence_completed"),
        "target_error_m": report.get("final_cube_to_requested_target_m"),
        "observed_lift_m": report.get("observed_lift_m"),
        "settle_displacement_m": report.get("settle_displacement_m"),
        "error": report.get("error"),
    }
    outcomes.append(outcome)
    print(json.dumps(outcome, ensure_ascii=False), flush=True)
    summary = {
        "protocol": 1,
        "controller": manifest["controller"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "completed": len(outcomes),
        "total": len(cases),
        "successes": sum(bool(item["success"]) for item in outcomes),
        "success_rate": sum(bool(item["success"]) for item in outcomes) / len(outcomes),
        "elapsed_s": time.perf_counter() - batch_started,
        "cases": outcomes,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

sys.exit(0 if all(item["success"] for item in outcomes) else 1)
