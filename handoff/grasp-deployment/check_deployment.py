"""Read-only, stdlib-only deployment inventory. Never starts Isaac or inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def text_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_readonly(argv: list[str]) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=15)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "error": type(exc).__name__}


def git(repo: Path, *args: str) -> dict:
    return run_readonly(["git", "--no-optional-locks", "-C", str(repo), *args])


def collect(repo: Path, manifest: dict, hash_weights: bool = False) -> dict:
    report = {
        "schema_version": 1, "repo": str(repo),
        "scope": "Static inventory only; neither health nor grasp success is certified.",
        "reference_commit": manifest["project_config_source"]["commit"],
        "head": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "branch", "--show-current"),
        "worktree": git(repo, "status", "--short"),
        "files": [], "weights": [], "dependencies": [],
    }
    for item in manifest["files"]:
        path = repo / item["target"]
        snapshot = HERE / item["snapshot"]
        row = {"path": item["target"], "exists": path.is_file(),
               "reference_sha256_lf_utf8": item["sha256_lf_utf8"]}
        try:
            row["snapshot_integrity"] = (snapshot.is_file() and
                text_hash(snapshot) == item["sha256_lf_utf8"])
            if path.is_file():
                row["sha256_lf_utf8"] = text_hash(path)
                row["matches_reference"] = row["sha256_lf_utf8"] == item["sha256_lf_utf8"]
        except (OSError, UnicodeError) as exc:
            row["read_error"] = type(exc).__name__
        row["tracked"] = git(repo, "ls-files", "--error-unmatch", "--", item["target"])["exit_code"] == 0
        row["ignore_rule"] = git(repo, "check-ignore", "-v", "--", item["target"]).get("stdout", "")
        report["files"].append(row)
    for item in manifest["weights"]:
        path = repo / item["target"]
        row = {"path": item["target"], "exists": path.is_file(),
               "hash_status": "not_checked", "reference_sha256": item["sha256"]}
        try:
            if path.is_file():
                row["bytes"] = path.stat().st_size
                row["size_matches_reference"] = row["bytes"] == item["bytes"]
                if hash_weights:
                    row["sha256"] = binary_hash(path)
                    row["hash_status"] = "match" if row["sha256"] == item["sha256"] else "mismatch"
        except OSError as exc:
            row["read_error"] = type(exc).__name__
        report["weights"].append(row)
    for relative in (
        "_vendor/GraspGenX/graspgenx/serving/zmq_server.py",
        "_vendor/GraspGenX/ext/gripper_descriptions",
        "assets/robots/so101/robot.urdf", "assets/robots/so101/robot.xrdf",
        "assets/robots/so101/rmp_flow.yaml",
        "_models/florence2/large/config.json", "_models/florence2/large/model.safetensors",
        "_models/yoloe/yoloe-26x-seg.pt",
    ):
        report["dependencies"].append({"path": relative, "exists": (repo / relative).exists()})
    # Only query vendor Git when it has its own .git; otherwise Git walks up to the project.
    vendor = repo / "_vendor/GraspGenX"
    report["vendor_head"] = (git(vendor, "rev-parse", "HEAD") if (vendor / ".git").exists()
                             else {"note": "Vendor clone .git missing; revision not verified"})
    report["expected_vendor_commit"] = manifest["graspgenx"]["commit"]
    report["entrypoint_paths"] = [
        {"entrypoint": "scripts/run_fine_grasp_demo.ps1", "reference_path": r"D:\isaac\env_isaacsim60\python.exe",
         "reference_path_exists": Path(r"D:\isaac\env_isaacsim60\python.exe").is_file()},
        {"entrypoint": "isaac_env.bat", "reference_path": r"D:\isaacsim\python.bat",
         "reference_path_exists": Path(r"D:\isaacsim\python.bat").is_file()},
    ]
    # Flag override presence only; never dump values, tokens, or the whole environment.
    report["vision_override_present"] = {name: bool(os.environ.get(name)) for name in
        ("BUSAGENT_FLORENCE_MODEL", "BUSAGENT_YOLOE_WEIGHTS")}
    report["manual_checks"] = [
        "Same startup command, entrypoint, case/seed, camera profile and Git commit?",
        "Actual runtime joint gains, Isaac version and external USD assets agree?",
        "Port 5556 belongs to the intended server/weights? Port open is not health.",
        "Vision default paths are hints only: overrides/legacy caches may be in use.",
        "Depth units, synchronized hand-eye chain, gripper/TCP and collision calibration agree?",
        "Failure logs must distinguish localization, candidates, IK, servo, closure and lift.",
    ]
    return report


def probe_model_python(executable: str) -> dict:
    # Metadata only: do not import torch, GraspGenX, Isaac, YOLO or Florence.
    code = """import importlib.metadata as m, importlib.util as u, json, sys
versions = {}
for name in ('torch','torchvision','numpy','graspgenx','pyzmq','msgpack','msgpack-numpy'):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
spec = u.find_spec('graspgenx')
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':versions,
                  'graspgenx_origin':None if spec is None else spec.origin}))
"""
    return run_readonly([executable, "-I", "-B", "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Existing target repository; never modified")
    parser.add_argument("--hash-weights", action="store_true", help="Also stream-read ~1.7 GB of checkpoint files")
    parser.add_argument("--model-python", help="Optional trusted local Python executable for metadata-only probe")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir() or git(repo, "rev-parse", "--show-toplevel").get("stdout", "").replace("\\", "/").casefold() != repo.as_posix().casefold():
        parser.error("--repo must name a Git repository root")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    report = collect(repo, manifest, args.hash_weights)
    if args.model_python:
        report["model_python_probe"] = probe_model_python(args.model_python)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    # Zero means inventory finished, not that installation or grasp passed.
    return 0


if __name__ == "__main__":
    sys.exit(main())
