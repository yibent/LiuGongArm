"""Run the Arena Panda service or a bounded physical smoke test."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import faulthandler
import signal

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path(os.environ.get('ARENA_PANDA_CONFIG', ROOT / 'configs/arena_panda.json')))
parser.add_argument("--output", type=Path, default=ROOT / "output/arena")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--smoke-fault", choices=["miss_grasp", "after_lift"],
                    help="Inject one failure in a smoke test to exercise model escalation")
parser.add_argument("--mode", choices=["basic", "enhanced", "auto"], default="basic")
parser.add_argument("--target", default="red block")
parser.add_argument("--destination", default="blue pad")
parser.add_argument("--port", type=int, default=7861)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode="performance")
args = parser.parse_args()
if args.smoke_fault and not args.smoke:
    parser.error("--smoke-fault requires --smoke")
args.enable_cameras = True
launcher = AppLauncher(args)
print("[Arena Panda] AppLauncher initialized", flush=True)
exit_code = 0

try:
    # Sim 6.0.1-rc.7 starts its own timeline callbacks before Lab 3 patches
    # SimulationManager. Retire those subscriptions before Lab owns physics;
    # otherwise PLAY initializes the same PhysX scene twice and deadlocks.
    from isaacsim.core.simulation_manager.impl.simulation_manager import SimulationManager
    SimulationManager.enable_all_default_callbacks(False)
    import isaacsim.core.experimental.utils.app as app_utils
    app_utils.enable_extension("isaacsim.robot.manipulators")
    from mr_liu.arena.environment import build_environment
    from mr_liu.arena.runtime import ArenaRuntime
    from mr_liu.arena.service import serve
    print("[Arena Panda] imports complete", flush=True)
    config = json.loads(args.config.read_text())
    env, task = build_environment(config, device=args.device)
    runtime = ArenaRuntime(env, task, config, args.output)
    if args.smoke:
        if args.smoke_fault:
            from arena_smoke_faults import inject_once
            inject_once(runtime, args.smoke_fault)
        result = runtime.execute({"command_id": "smoke", "skill": "pick_place", "params": {
            "target": {"category": args.target}, "destination": {"label": args.destination}, "mode": args.mode}})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not result["ok"]:
            raise RuntimeError(result["message"])
    else:
        serve(runtime, launcher.app, args.port)
except BaseException:
    exit_code = 1
    traceback.print_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "startup-error.txt").write_text(traceback.format_exc())
    raise
finally:
    launcher.app.close(exit_code=exit_code)
