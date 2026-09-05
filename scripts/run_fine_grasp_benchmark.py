"""Run reproducible unseen-object FineGrasp trials in isolated Isaac processes."""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.benchmark import (  # noqa: E402
    BenchmarkCase,
    aggregate_runs,
    classify_report,
    default_unseen_cases,
    render_markdown,
    write_case,
    validate_frozen_case_bytes,
)


def _parse_seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physical eye-in-hand FineGrasp unseen-object regression"
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path(os.environ.get("ISAAC_PYTHON", sys.executable)),
        help="Isaac Sim Python executable (or set ISAAC_PYTHON)",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "unseen_benchmark")
    parser.add_argument("--seeds", type=_parse_seed_list, default=[0])
    parser.add_argument(
        "--cases",
        default=(
            "cube,sphere,cylinder,thin,metallic_part,apple,bottle,hammer,"
            "wrench,screwdriver,key,coffee_mug"
        ),
        help="Comma-separated default case names",
    )
    parser.add_argument(
        "--case-json",
        type=Path,
        action="append",
        default=[],
        help="Additional custom case JSON; may be repeated",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--backend", choices=("geometric", "graspgenx"), default="geometric"
    )
    parser.add_argument("--graspgenx-port", type=int, default=5556)
    parser.add_argument("--perception", choices=("single", "multiview"), default="single")
    parser.add_argument("--segmenter", choices=("depth", "sam2"), default="depth")
    parser.add_argument("--recovery", choices=("off", "assisted", "active"), default="off")
    parser.add_argument("--drop-initial-wrist-frames", type=int, default=0)
    parser.add_argument("--test-target-shift-m", type=float, default=0.)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("development", "acceptance"), default="development")
    parser.add_argument("--dry-run", action="store_true", help="Exercise planning only (not scored as physical success)")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--list", action="store_true", help="Print expanded cases without running Isaac")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _slug(value: str) -> str:
    slug = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return slug.strip("-") or "case"


def _expanded_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        cases = []
        for entry in manifest["splits"][args.split]:
            path = args.manifest.parent / entry["path"]
            data = path.read_bytes()
            if not validate_frozen_case_bytes(data, entry["sha256"]):
                raise ValueError(f"Frozen case was modified: {path}")
            cases.append(BenchmarkCase.from_mapping(json.loads(data)))
        return cases
    requested = {item.strip() for item in args.cases.split(",") if item.strip()}
    cases: list[BenchmarkCase] = []
    for seed in args.seeds:
        available = {case.name: case for case in default_unseen_cases(seed)}
        unknown = requested - available.keys()
        if unknown:
            raise ValueError(f"Unknown default cases: {', '.join(sorted(unknown))}")
        cases.extend(available[name] for name in available if name in requested)
    from mr_liu.grasp.benchmark import load_case

    cases.extend(load_case(path) for path in args.case_json)
    return cases


def _load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _persist_summary(output: Path, runs: list[dict]) -> dict:
    summary = aggregate_runs(runs)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "SUMMARY.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _run_case(args: argparse.Namespace, case: BenchmarkCase, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    case_path = run_dir / "case.json"
    write_case(case_path, case)
    command = [
        str(args.isaac_python),
        str(ROOT / "scripts" / "run_fine_grasp_demo.py"),
        "--case-json",
        str(case_path),
        "--output",
        str(run_dir),
        "--backend",
        args.backend,
        "--graspgenx-port",
        str(args.graspgenx_port),
        "--perception", args.perception,
        "--segmenter", args.segmenter,
        "--recovery", args.recovery,
        "--drop-initial-wrist-frames", str(args.drop_initial_wrist_frames),
        "--test-target-shift-m", str(args.test_target_shift_m),
        "--headless" if args.headless else "--no-headless",
    ]
    if args.dry_run:
        command.append("--dry-run")
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    started = time.monotonic()
    returncode: int | None = None
    error: str | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout_s,
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
            check=False,
        )
        returncode = completed.returncode
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        error = f"timeout_after_{args.timeout_s:g}s"
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    except OSError as exc:
        error = f"launcher_error:{type(exc).__name__}:{exc}"

    report = _load_json(run_dir / "report.json")
    if report is None and error is None:
        error_file = run_dir / "error.txt"
        error = error_file.read_text(encoding="utf-8", errors="replace")[-2000:] if error_file.exists() else None
    row = classify_report(
        report,
        process_returncode=returncode,
        process_error=error,
        expected_feasible=case.expected_feasible,
    )
    row.update(
        {
            "case": case.name,
            "seed": case.seed,
            "shape": case.shape,
            "material": case.material,
            "dimensions_m": list(case.dimensions_m),
            "run_dir": str(run_dir.resolve()),
            "wall_time_s": time.monotonic() - started,
            "dry_run": bool(args.dry_run),
        }
    )
    if args.dry_run:
        # Planning-only runs are useful diagnostics but can never satisfy the
        # benchmark's physical grasp definition.
        row["success"] = False
        row["failure_category"] = "dry_run_not_physical"
    (run_dir / "benchmark_result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> int:
    args = _parse_args()
    cases = _expanded_cases(args)
    if args.list:
        print(json.dumps([case.to_dict() for case in cases], indent=2))
        return 0
    if not args.isaac_python.is_file():
        raise FileNotFoundError(f"Isaac Python executable not found: {args.isaac_python}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite benchmark evidence: {output}")
    runs: list[dict] = []
    for index, case in enumerate(cases, start=1):
        run_dir = output / f"{index:03d}_{_slug(case.name)}_seed{case.seed}"
        print(f"[{index}/{len(cases)}] {case.name} seed={case.seed} -> {run_dir}", flush=True)
        row = _run_case(args, case, run_dir)
        runs.append(row)
        summary = _persist_summary(output, runs)
        print(
            f"  {'PASS' if row.get('task_success', row['success']) else 'FAIL'} "
            f"failure={row.get('failure_category')} lift={row.get('actual_lift_m')} "
            f"aggregate={summary['successes']}/{summary['trials']}",
            flush=True,
        )
        if args.fail_fast and not row.get("task_success", row["success"]):
            break
    return 0 if runs and all(run.get("task_success", run["success"]) for run in runs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
