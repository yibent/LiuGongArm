from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"D:\isaac\env_isaacsim60\python.exe")
RUNNER = ROOT / "scripts" / "run_official_franka_shape_pick_place.py"
OUT = ROOT / "output" / "franka" / "official_shapes_20260906_a"
cases = [
    ("cube", (0.30, 0.25, 0.03), (-0.25, -0.25, 0.03), (0.06, 0.06, 0.06), .03),
    ("cylinder", (0.30, -0.22, 0.05), (-0.25, 0.20, 0.05), (0.035, 0.035, 0.10), .04),
    ("sphere", (0.28, 0.22, 0.04), (-0.22, -0.20, 0.04), (0.04, 0.04, 0.04), .03),
]
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit("Use a new output directory")
OUT.mkdir(parents=True)
outcomes = []
for shape, start, target, size, mass in cases:
    case_dir = OUT / shape
    log = OUT / f"{shape}.console.log"
    cmd = [str(PYTHON), str(RUNNER), "--output", str(case_dir), "--shape", shape,
           "--start", *map(str, start), "--target", *map(str, target),
           "--size", *map(str, size), "--mass-kg", str(mass)]
    t = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        code = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    report_path = case_dir / "official_franka_shape_pick_place_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    outcomes.append({"shape": shape, "exit_code": code, "success": bool(report.get("physical_pick_place_observed")),
                     "target_error_m": report.get("target_error_m"), "observed_lift_m": report.get("observed_lift_m"),
                     "settle_displacement_m": report.get("settle_displacement_m"), "elapsed_s": time.perf_counter() - t})
    print(json.dumps(outcomes[-1]), flush=True)
summary = {"controller": "official PickPlaceController", "cases": outcomes,
           "successes": sum(x["success"] for x in outcomes), "total": len(outcomes)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
