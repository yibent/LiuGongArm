"""Shared model locations for CLI, WebUI and in-process consumers."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLORENCE_REPO = "florence-community/Florence-2-large"
FLORENCE_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
FLORENCE_SHA256 = "7715423d6549bf1e71188bdd84f4ac960cc0597886af24a5ef7b66f128660685"
FLORENCE_DIR = ROOT / "_models" / "florence2" / "large"
YOLOE_PATH = ROOT / "_models" / "yoloe" / "yoloe-26x-seg.pt"
YOLOE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt"
YOLOE_SHA256 = "d08d390a08f98195f7c87807839fe4ff93a5491645fef1bc3bf0700efafdd639"


def default_florence() -> str:
    # A local default fails clearly if setup has not run; no implicit download.
    return os.environ.get("BUSAGENT_FLORENCE_MODEL") or str(FLORENCE_DIR)


def default_yoloe() -> Path:
    if override := os.environ.get("BUSAGENT_YOLOE_WEIGHTS"):
        return Path(override)
    legacy = ROOT / "yoloe-26x-seg.pt"
    return legacy if legacy.is_file() and not YOLOE_PATH.is_file() else YOLOE_PATH
