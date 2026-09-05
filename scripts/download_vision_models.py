"""Download pinned public weights; never load unverified YOLO checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
os.environ.setdefault("HF_HOME", str(ROOT / "_models" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from find_and_track.settings import (  # noqa: E402
    FLORENCE_DIR, FLORENCE_REPO, FLORENCE_REVISION, FLORENCE_SHA256,
    YOLOE_PATH, YOLOE_SHA256, YOLOE_URL,
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise RuntimeError(f"Missing or invalid checkpoint: {path}. Keep it for inspection and re-download explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--only", choices=["florence", "yoloe", "all"], default="all")
    args = parser.parse_args()
    manifest = {}
    if args.only in ("florence", "all"):
        if not args.verify_only:
            from huggingface_hub import snapshot_download

            print("Downloading Florence native Transformers checkpoint (no remote Python code)...", flush=True)
            snapshot_download(
                repo_id=FLORENCE_REPO, revision=FLORENCE_REVISION,
                local_dir=FLORENCE_DIR,
                allow_patterns=["*.json", "*.safetensors", "merges.txt", "README.md"],
                max_workers=4,
            )
        verify(FLORENCE_DIR / "model.safetensors", FLORENCE_SHA256)
        manifest["florence"] = {"repo": FLORENCE_REPO, "revision": FLORENCE_REVISION, "sha256": FLORENCE_SHA256}
    if args.only in ("yoloe", "all"):
        if not YOLOE_PATH.exists() and not args.verify_only:
            YOLOE_PATH.parent.mkdir(parents=True, exist_ok=True)
            partial = YOLOE_PATH.with_suffix(".pt.partial")
            print(f"Downloading {YOLOE_URL}", flush=True)
            urllib.request.urlretrieve(YOLOE_URL, partial)
            verify(partial, YOLOE_SHA256)
            partial.replace(YOLOE_PATH)
        verify(YOLOE_PATH, YOLOE_SHA256)
        manifest["yoloe"] = {"url": YOLOE_URL, "sha256": YOLOE_SHA256}
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
