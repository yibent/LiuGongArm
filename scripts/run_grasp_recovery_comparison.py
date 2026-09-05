"""Freeze paired development cases and compare baseline / assisted / recovery.

Requires the locally configured Isaac and GraspGenX environments. Each profile
runs isolated Isaac trials with one model server; no acceptance cases are tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.grasp.benchmark import default_unseen_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", default="cube,apple,hammer,wrench,coffee_mug")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--profiles", default="off,assisted,active")
    parser.add_argument("--backend", choices=("graspgenx", "geometric"), default="graspgenx")
    parser.add_argument("--segmenter", choices=("depth", "sam2"), default="depth")
    parser.add_argument("--drop-initial-wrist-frames", type=int, default=0)
    parser.add_argument("--test-target-shift-m", type=float, default=0.)
    args = parser.parse_args()
    output = args.output or ROOT / "output/recovery_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases_dir = output / "cases"
    cases_dir.mkdir()
    profiles = args.profiles.split(",")
    if len(set(profiles)) != len(profiles) or set(profiles) - {"off", "assisted", "active"}:
        raise ValueError("Profiles must be unique off/assisted/active values")
    names = set(args.cases.split(","))
    seeds = [int(seed) for seed in args.seeds.split(",")]
    entries = []
    for seed in seeds:
        available = {case.name: case for case in default_unseen_cases(seed)}
        if names - available.keys():
            raise ValueError(f"Unknown cases: {names - available.keys()}")
        for name in sorted(names):
            path = cases_dir / f"{name}_{seed}.json"
            data = (json.dumps(available[name].to_dict(), sort_keys=True, indent=2) + "\n").encode()
            path.write_bytes(data)
            entries.append({"path": path.relative_to(output).as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"version": 2, "purpose": "paired_development_not_acceptance",
                                   "splits": {"development": entries}}, indent=2), encoding="utf-8")
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                  "profiles": profiles, "backend": args.backend, "segmenter": args.segmenter,
                  "fault_initial_wrist_frames": args.drop_initial_wrist_frames,
                  "test_target_shift_m": args.test_target_shift_m,
                  "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    results = {}
    for profile in profiles:
        command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(ROOT / "scripts/run_fine_grasp_demo.ps1"), "-Benchmark", "-Backend", args.backend,
                   "-Recovery", profile, "-Perception", "single", "-Segmenter", args.segmenter,
                   "-DropInitialWristFrames", str(args.drop_initial_wrist_frames),
                   "-TestTargetShiftM", str(args.test_target_shift_m),
                   "-Manifest", str(manifest), "-Split", "development", "-Output", str(output / profile)]
        print(f"PROFILE_START {profile} {len(entries)} cases", flush=True)
        with (output / f"{profile}.log").open("w", encoding="utf-8") as log:
            run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        summary_path = output / profile / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"{profile}: no summary; exit={run.returncode}; see {profile}.log")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results[profile] = {key: summary.get(key) for key in (
            "trials", "successes", "first_attempt_successes", "recovered_successes", "successful_regrasps",
            "false_successes", "attempted_trials", "active_view_moves", "metrics", "failure_counts")}
        (output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"PROFILE_DONE {profile}: {summary['successes']}/{summary['trials']}, exit={run.returncode}", flush=True)
    lines = ["# Paired development comparison", "", "Not held-out generalization or real-hardware validation.", "",
             "| Profile | First success | Final success | Recovered | True regrasp | False success | Views | Mean cycle s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for profile, value in results.items():
        cycle = value["metrics"]["cycle_time_s"]["mean"]
        lines.append(f"| {profile} | {value['first_attempt_successes']}/{value['trials']} | "
                     f"{value['successes']}/{value['trials']} | {value['recovered_successes']} | "
                     f"{value['successful_regrasps']} | {value['false_successes']} | "
                     f"{value['active_view_moves']} | {cycle if cycle is not None else 'unknown'} |")
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"COMPARISON_WRITTEN {output}", flush=True)


if __name__ == "__main__":
    main()
