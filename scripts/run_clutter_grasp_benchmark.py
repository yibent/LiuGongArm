"""Run reproducible clutter and collision trials in isolated Isaac processes.

The script intentionally reports safety refusals as outcomes.  A blocked path,
ambiguous label, or motion-time guard stop is useful evidence when it is
classified against the scene's expected outcome; it is not a process crash.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.benchmark import default_demo_case, load_case, write_case  # noqa: E402
from mr_liu.sim.clutter import ClutterScene, default_clutter_scenes, write_scene  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim clutter/collision benchmark")
    parser.add_argument("--isaac-python", type=Path,
                        default=Path(os.environ.get("ISAAC_PYTHON", sys.executable)))
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "clutter_benchmark")
    parser.add_argument("--scenes", default="all",
                        help="Comma-separated scene names, or 'all'")
    parser.add_argument("--target-case-json", type=Path,
                        help="Optional BenchmarkCase JSON for the target")
    parser.add_argument("--label", default="red cube",
                        help="Semantic label sent to the Florence/YOLOE locator; pass an empty string to use measured target pose")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coarse-only", action="store_true",
                        help="Stop after scene-to-wrist handoff; useful for perception/path checks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run FineGrasp planning without scoring a physical lift")
    parser.add_argument("--backend", choices=("geometric", "graspgenx"), default="geometric")
    parser.add_argument("--graspgenx-port", type=int, default=5556)
    parser.add_argument("--locator-port", type=int, default=5570)
    parser.add_argument("--localization-mode", choices=("florence", "florence_yoloe"), default="florence_yoloe")
    parser.add_argument("--perception", choices=("single", "multiview"), default="multiview")
    parser.add_argument("--recovery", choices=("off", "assisted", "active"), default="active")
    parser.add_argument("--segmenter", choices=("depth", "sam2"), default="depth")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    return parser.parse_args()


def _slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    return slug.strip("-") or "scene"


def _failure_from_files(run_dir: Path) -> str | None:
    for name in ("report.json", "coarse_report.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        coarse = payload.get("coarse_approach") if isinstance(payload.get("coarse_approach"), dict) else payload
        for value in (result.get("failure"), coarse.get("failure"), coarse.get("reason")):
            if value:
                return str(value)
    error = run_dir / "error.txt"
    if error.is_file():
        text = error.read_text(encoding="utf-8", errors="replace")
        for marker in ("ambiguous_label_instances", "obstacle_collision", "target_moved_during_transit", "coarse_path_infeasible"):
            if marker in text:
                return marker
        if "Failed to place gripper in observation/open state" in text:
            return "gripper_open_blocked"
    return None


def _run_one(args: argparse.Namespace, scene: ClutterScene, target_case, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    scene_path = run_dir / "clutter_scene.json"
    case_path = run_dir / "target_case.json"
    write_scene(scene_path, scene)
    write_case(case_path, target_case)
    command = [
        str(args.isaac_python), str(ROOT / "scripts" / "run_fine_grasp_demo.py"),
        "--case-json", str(case_path), "--clutter-case-json", str(scene_path),
        "--locator-port", str(args.locator_port),
        "--localization-mode", args.localization_mode, "--backend", args.backend,
        "--graspgenx-port", str(args.graspgenx_port), "--perception", args.perception,
        "--recovery", args.recovery, "--segmenter", args.segmenter,
        "--headless" if args.headless else "--no-headless",
        "--output", str(run_dir),
    ]
    if args.coarse_only:
        command.append("--coarse-only")
    if args.dry_run:
        command.append("--dry-run")
    if args.label:
        command.extend(["--label", args.label])
        if scene.metadata.get("inject_motion_fault"):
            # One-shot 25 mm displacement is above the coarse guard threshold
            # and is applied only to the simulated body, never to perception.
            command.extend(["--test-coarse-shift-m", "0.025"])
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    started = time.monotonic()
    process_error = None
    returncode = None
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=args.timeout_s, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, check=False,
        )
        returncode = completed.returncode
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        process_error = f"timeout_after_{args.timeout_s:g}s"
        (run_dir / "stdout.log").write_text(exc.stdout or "", encoding="utf-8")
        (run_dir / "stderr.log").write_text(exc.stderr or "", encoding="utf-8")
    failure = _failure_from_files(run_dir)
    report = {}
    report_path = run_dir / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
    coarse = report.get("coarse_approach") if isinstance(report.get("coarse_approach"), dict) else {}
    node_success = bool((report.get("result") or {}).get("success"))
    handoff_success = bool(coarse.get("passed"))
    expected_failure_match = bool(failure and (scene.expected_failure is None or scene.expected_failure in failure))
    if scene.expected_outcome == "safe_reject":
        passed = expected_failure_match
        status = "SAFE_REJECT" if passed else "UNEXPECTED"
    elif scene.expected_outcome == "recover_or_reject":
        passed = bool(node_success or handoff_success or expected_failure_match)
        status = "RECOVERED" if (node_success or handoff_success) else ("SAFE_REJECT" if expected_failure_match else "FAIL")
    elif args.coarse_only:
        passed = handoff_success
        status = "HANDOFF_PASS" if passed else "FAIL"
    else:
        passed = node_success
        status = "PASS" if passed else "FAIL"
    return {
        "scene": scene.name,
        "description": scene.description,
        "expected_outcome": scene.expected_outcome,
        "expected_failure": scene.expected_failure,
        "status": status,
        "passed": passed,
        "failure": failure,
        "returncode": returncode,
        "process_error": process_error,
        "elapsed_s": round(time.monotonic() - started, 3),
        "clutter_body_count": len(scene.bodies),
        "clutter_prim_paths": report.get("clutter_prim_paths", []),
        "report_path": str(report_path.relative_to(ROOT)) if report_path.is_file() else None,
        "coarse_report_path": str((run_dir / "coarse_report.json").relative_to(ROOT)) if (run_dir / "coarse_report.json").is_file() else None,
    }


def _render(summary: dict[str, Any]) -> str:
    lines = ["# Clutter and collision benchmark", "", f"Passed: **{summary['passed']}/{summary['trials']}**", "",
             "| Scene | Bodies | Expected | Status | Failure | Return code |", "|---|---:|---|---|---|---:|"]
    for row in summary["runs"]:
        lines.append(f"| {row['scene']} | {row['clutter_body_count']} | {row['expected_outcome']} | {row['status']} | {row.get('failure') or '-'} | {row.get('returncode') if row.get('returncode') is not None else '-'} |")
    lines.extend(["", "## Interpretation", "", "`SAFE_REJECT` is a passing safety result when the reported reason matches the scene expectation. `PASS` and `HANDOFF_PASS` require the requested operation to complete. `UNEXPECTED` means the process ran but did not produce the expected refusal.", ""])
    return "\n".join(lines)


def main() -> int:
    args = _args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite clutter benchmark evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scenes = default_clutter_scenes()
    requested = {item.strip() for item in args.scenes.split(",") if item.strip()}
    if requested != {"all"}:
        available = {scene.name: scene for scene in scenes}
        unknown = requested - available.keys()
        if unknown:
            raise ValueError(f"Unknown clutter scenes: {', '.join(sorted(unknown))}")
        scenes = [scene for scene in scenes if scene.name in requested]
    target_case = load_case(args.target_case_json) if args.target_case_json else default_demo_case()
    runs = []
    for index, scene in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {scene.name}: {scene.description}", flush=True)
        runs.append(_run_one(args, scene, target_case, output / f"{index:02d}_{_slug(scene.name)}"))
    summary = {"trials": len(runs), "passed": sum(bool(row["passed"]) for row in runs), "runs": runs,
               "args": {key: str(value) for key, value in vars(args).items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "SUMMARY.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] == summary["trials"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
