"""Launch SO-101 cuMotion follow-target with Isaac Sim."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.app.run_follow_target import main

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
args, _ = parser.parse_known_args()

try:
    main(simulation_app, test_frames=100 if args.test else None, device=args.device)
except KeyboardInterrupt:
    print("\nExiting...")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
