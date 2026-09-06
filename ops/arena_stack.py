"""Start/stop only this checkout's services, with separate process groups."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "output/services"
NODE = os.environ.get("BUSAGENT_NODE") or shutil.which("node") or "/root/gpufree-data/tools/node/bin/node"
SERVICES = {
    "vision": (["bash", str(ROOT / "scripts/run_arena_vision.sh")], ROOT),
    "graspgenx": (["bash", str(ROOT / "scripts/run_graspgenx_server.sh")], ROOT),
    "anyplace": (["bash", str(ROOT / "scripts/run_anyplace_server.sh")], ROOT),
    "arena": (["bash", str(ROOT / "scripts/run_arena_panda.sh"), "--viz", "kit"], ROOT),
    "busagent": ([NODE, str(ROOT / "BusAgent/backend/dist/main.js")], ROOT / "BusAgent/backend"),
}


def alive(pid):
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        return state != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("services", nargs="*", metavar="SERVICE")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    names = args.services or list(SERVICES)
    if any(name not in SERVICES for name in names):
        parser.error("Services: " + ", ".join(SERVICES))
    for name in names:
        command, cwd = SERVICES[name]
        file = STATE / f"{name}.json"
        previous = json.loads(file.read_text()) if file.exists() else None
        running = previous and alive(previous["pid"])
        if running:
            actual = Path(f"/proc/{previous['pid']}/cmdline").read_bytes().decode().replace("\0", " ")
            if str(ROOT) not in actual and str(cwd) != os.readlink(f"/proc/{previous['pid']}/cwd"):
                raise RuntimeError(f"PID {previous['pid']} was reused; refusing to signal it")
        if args.action == "start" and not running:
            env = {**os.environ, "BUSAGENT_PORT": "3100", "BUSAGENT_ROBOT": "franka_panda"}
            with (STATE / f"{name}.log").open("ab") as log:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            file.write_text(json.dumps({"pid": process.pid, "command": command, "started_at": time.time()}))
            print(f"{name}: launched {process.pid}; readiness is reported by its HTTP interface/log")
        elif args.action == "stop" and running:
            os.killpg(previous["pid"], signal.SIGTERM)
            for _ in range(30):
                if not alive(previous["pid"]): break
                time.sleep(.1)
            if alive(previous["pid"]): os.killpg(previous["pid"], signal.SIGKILL)
            print(f"{name}: stopped")
        else:
            print(f"{name}: {'process running' if running else 'not managed/running'}")


if __name__ == "__main__":
    main()
