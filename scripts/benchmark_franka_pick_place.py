"""Frozen, resumable Isaac pick/place development trials with strict outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.grasp.benchmark import BenchmarkCase, default_demo_case, default_unseen_cases


def default_matrix():
    cases = []
    def add(name, obj, *, backend="m2t2", fixture="region", x=.25, y=-.198, init=False, clutter=0):
        label = {"cube": "red cube", "sphere": "green ball", "cylinder": "red cylinder"}[obj.shape]
        cases.append(dict(name=name, object=obj.to_dict(), label=label, place_backend=backend,
            grasp_backend="m2t2" if backend == "m2t2" else "graspgenx",
            fixture=fixture, relation="inside" if fixture == "tray" else "on",
            destination="blue tray" if fixture == "tray" else "blue square",
            x=x, y=y, init_current_orientation=init, clutter=clutter))
    cube = default_demo_case()
    varied = {c.name: c for c in default_unseen_cases(1)}
    add("01_m2t2_cube_seed1", varied["cube"])
    add("02_anyplace_cube_current", cube, backend="anyplace", init=True)
    add("03_m2t2_tray_seed1", varied["cube"], fixture="tray", x=.23)
    add("04_m2t2_cylinder_seed1", replace(varied["cylinder"], color_rgb=(.85,.12,.08)), y=-.18)
    add("05_m2t2_sphere_seed1", varied["sphere"], x=.24)
    add("06_anyplace_cube_random", cube, backend="anyplace")
    add("07_m2t2_large_cube", replace(cube, name="large_cube", dimensions_m=(.044,)*3,
        position_m=(.329,-.19,1.072), seed=2, yaw_deg=12), x=.24, y=-.18)
    add("08_anyplace_tray_current", cube, backend="anyplace", fixture="tray", x=.23, init=True)
    add("09_m2t2_cube_clutter1", cube, x=.24, y=-.18, clutter=1)
    add("10_anyplace_cube_clutter2", cube, backend="anyplace", x=.24, y=-.18, init=True, clutter=2)
    return dict(protocol=1, purpose="development_comparison_not_acceptance", robot="franka",
                contact_material="profile", cases=cases)


def validate_matrix(matrix):
    if matrix.get("protocol") != 1 or matrix.get("robot") != "franka" or not matrix.get("cases"):
        raise ValueError("Expected a nonempty Franka protocol-1 matrix")
    if matrix.get("contact_material") not in {"asset", "profile"}:
        raise ValueError("Unknown contact material")
    names = set()
    for case in matrix["cases"]:
        name = case["name"]
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in names:
            raise ValueError("Invalid or duplicate run name")
        names.add(name)
        BenchmarkCase.from_mapping(case["object"])
        if (case["place_backend"], case["grasp_backend"]) not in {("m2t2","m2t2"),("anyplace","graspgenx")}:
            raise ValueError("Unsupported explicit backend combination")
        if (case["fixture"], case["relation"]) not in {("region","on"),("tray","inside"),
                                                          ("socket","insert"),("rack","hang")}:
            raise ValueError("Unsupported fixture/relation")
        if not (.20 <= case["x"] <= .30 and -.24 <= case["y"] <= -.15):
            raise ValueError("Fixture outside development workspace")
        if case.get("clutter", 0) not in (0, 1, 2):
            raise ValueError("Invalid clutter level")
        for key in ("label", "destination"):
            if not isinstance(case[key], str) or not case[key].strip() or len(case[key]) > 256:
                raise ValueError("A bounded semantic label is required")


def fingerprint():
    files = {}
    for folder, patterns in (("source",("*.py",)), ("scripts",("*.py","*.ps1")),
                             ("configs",("*.yaml",)), ("assets/robots/franka",("*.urdf","*.xrdf","*.yaml"))):
        for pattern in patterns:
            for path in sorted((ROOT / folder).rglob(pattern)):
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_report(run, name):
    path = run / name
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {"read_error":"report_is_not_object"}
    except (OSError, ValueError) as exc:
        return {"read_error": str(exc)}


def outcome(run, exit_code, *, timed_out=False):
    grasp, place = read_report(run,"report.json"), read_report(run,"place_report.json")
    pick, put = grasp.get("result",{}), place.get("result",{})
    grasp_ok = pick.get("success") is True
    place_ok = all(put.get(key) is True for key in ("success","released","verified"))
    complete = grasp_ok and place_ok and exit_code == 0 and not timed_out
    failure = ("process_timeout" if timed_out else
        put.get("failure") or pick.get("failure") or
        ("process_exit" if exit_code != 0 else "incomplete_evidence"))
    return dict(complete_success=complete, grasp_success=grasp_ok, place_success=place_ok,
                released=put.get("released") is True, verified=put.get("verified") is True,
                failure=None if complete else failure, phase=put.get("phase",pick.get("phase")),
                exit_code=exit_code, timed_out=timed_out, place_metrics=put.get("metrics",{}),
                grasp_metrics=pick.get("metrics",{}), physical_lift_m=grasp.get("actual_target_lift_m"),
                report_errors=[r["read_error"] for r in (grasp,place) if "read_error" in r])


def command(case, matrix, object_path, run, record_video):
    script = "run_m2t2_pick_place.ps1" if case["place_backend"] == "m2t2" else "run_anyplace_pick_place.ps1"
    args = ["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"scripts"/script),
            "-Robot","franka","-ContactMaterial",matrix["contact_material"],"-Label",case["label"],
            "-Destination",case["destination"],"-Fixture",case["fixture"],"-Relation",case["relation"],
            "-FixtureX",str(case["x"]),"-FixtureY",str(case["y"]),"-Clutter",str(case.get("clutter",0)),
            "-CaseJson",str(object_path),"-Output",str(run)]
    if case.get("init_current_orientation"):
        args.append("-InitCurrentOrientation")
    if record_video:
        args.append("-RecordVideo")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=900.)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        if args.write_manifest.exists():
            raise FileExistsError(args.write_manifest)
        matrix = default_matrix()
        validate_matrix(matrix)
        args.write_manifest.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.write_manifest,matrix)
        return 0
    if not args.manifest or not args.output or args.limit < 0 or args.timeout_s < 60:
        parser.error("--manifest and --output required; timeout >=60 and limit >=0")
    matrix = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    validate_matrix(matrix)
    output = args.output.resolve()
    snapshot = fingerprint()
    if args.resume:
        if json.loads((output/"manifest.json").read_text()) != matrix:
            raise ValueError("Cannot resume a changed manifest")
        if json.loads((output/"source_hashes.json").read_text()) != snapshot:
            raise ValueError("Code/config changed: start a new comparison directory")
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"manifest.json",matrix)
        write_json(output/"source_hashes.json",snapshot)
    (output/"objects").mkdir(exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    rows, executed = [], 0
    for case in matrix["cases"]:
        status_path = output/(case["name"]+".result.json")
        if status_path.exists():
            rows.append(json.loads(status_path.read_text()))
            continue
        if args.limit and executed >= args.limit:
            break
        if fingerprint() != snapshot:
            raise RuntimeError("Code/config changed during the batch; stop before the next trial")
        run = output/case["name"]
        if run.exists():
            raise FileExistsError(f"Unfinished evidence must be inspected, not overwritten: {run}")
        object_path = output/"objects"/(case["name"]+".json")
        write_json(object_path,case["object"])
        cmd = command(case,matrix,object_path,run,args.record_video)
        started = time.monotonic()
        print("START "+case["name"],flush=True)
        timed_out = False
        with (output/"logs"/(case["name"]+".log")).open("wb") as log:
            process = subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:
                exit_code = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Only the process tree owned by this single simulation trial.
                subprocess.run(["taskkill.exe","/PID",str(process.pid),"/T","/F"],stdout=log,stderr=log,check=True)
                exit_code = process.wait(timeout=30)
        row = dict(name=case["name"],configuration=case,command=cmd,wall_s=time.monotonic()-started,
                   completed_at=datetime.now(timezone.utc).isoformat(),**outcome(run,exit_code,timed_out=timed_out))
        write_json(status_path,row)
        rows.append(row)
        executed += 1
        write_json(output/"summary.json",dict(kind="Isaac_development_comparison",real_robot_test=False,
            planned=len(matrix["cases"]),completed=len(rows),complete_successes=sum(r["complete_success"] for r in rows),
            failures=dict(Counter(r["failure"] for r in rows if r["failure"])),results=rows))
        print("DONE "+case["name"]+" "+str(row["failure"] or "complete_success")+f" {row['wall_s']:.1f}s",flush=True)
    return 0 if rows and all(row["complete_success"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
