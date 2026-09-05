"""Launch SO-101 dual-camera vision follow-target with Isaac Sim 6.0."""

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Exit after 100 simulation frames")
parser.add_argument("--test-frames", type=int, help="Exit after this many simulation frames")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--prompt", default="red cube")
parser.add_argument("--fast", choices=["yoloe", "cv"], default="yoloe")
parser.add_argument("--slow-interval", type=int, default=45)
parser.add_argument("--control-host", default="127.0.0.1")
parser.add_argument("--control-port", type=int, default=7861)
parser.add_argument("--grasp-backend", choices=["graspgenx", "geometric", "m2t2", "disabled"], default="graspgenx")
parser.add_argument(
    "--no-follow",
    action="store_true",
    help="Run dual-view FIND/TRACK without moving the arm toward top-view detections",
)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

ROOT = Path(__file__).resolve().parents[1]
# Keep Isaac Sim's bundled packages ahead of the project venv. Prepending the
# venv with PYTHONPATH can replace Isaac's torch/rendering stack and stall Kit,
# while appending it still makes Florence/YOLOE-only packages discoverable.
venv_site = ROOT / ".venv" / "Lib" / "site-packages"
if not venv_site.is_dir():
    venv_site = ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if venv_site.is_dir() and str(venv_site) not in sys.path:
    sys.path.append(str(venv_site))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
venv_root = ROOT / ".venv"
if os.name == "nt" and venv_root.is_dir():
    os.environ["PATH"] = str(venv_root) + ";" + str(venv_root / "Library" / "bin") + ";" + os.environ.get("PATH", "")
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.app.run_vision_follow import main

try:
    main(
        simulation_app,
        test_frames=(
            args.test_frames if args.test_frames is not None else (100 if args.test else None)
        ),
        device=args.device,
        prompt=args.prompt,
        fast=args.fast,
        slow_interval=args.slow_interval,
        control_host=args.control_host,
        control_port=args.control_port,
        follow_target=not args.no_follow,
        grasp_backend=args.grasp_backend,
    )
except KeyboardInterrupt:
    print("\nExiting...")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
