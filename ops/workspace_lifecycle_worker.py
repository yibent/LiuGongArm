"""Execute one validated scene transition without exposing a privileged network API.

BusAgent validates the request and writes the operation journal before spawning
this worker. The worker accepts only a catalog scene id and clears only data
owned by the current simulation session.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "output/system/workspace.json"
SERVICES = ("busagent", "arena")
RUNTIME_PATHS = (
    ROOT / "output/arena",
    ROOT / "output/busagent_grasp",
    ROOT / "BusAgent/backend/.local/mastra",
)


def write_journal(value):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOURNAL.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(JOURNAL)


def update(operation, phase, message, **extra):
    operation.update(phase=phase, message=message, **extra)
    write_journal({"epoch": operation["epoch"], "operation": operation})


def scene_config(scene_id):
    catalog_path = ROOT / "configs/industrial_scenes.json"
    scenes = json.loads(catalog_path.read_text())
    row = next((entry for entry in scenes if entry.get("id") == scene_id), None)
    if row is None:
        raise ValueError("Scene is not in the installed catalog")
    path = (ROOT / row["config"]).resolve()
    if not path.is_relative_to((ROOT / "configs/scenes").resolve()):
        raise ValueError("Scene config is outside the catalog directory")
    config = json.loads(path.read_text())
    if config.get("scene", {}).get("id") != scene_id or not config.get("entities"):
        raise ValueError("Scene config is incomplete")
    for entity in config["entities"]:
        asset = entity.get("usd_path")
        if asset and not (ROOT / asset).is_file():
            raise ValueError("Scene asset is missing")
    return path


def supervisor(action, names):
    command = [
        "/usr/bin/supervisorctl",
        "-c",
        str(ROOT / "ops/arena-supervisord.conf"),
        action,
        *names,
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=200)
    lines = [line for line in result.stdout.splitlines() if line]
    stopped = action == "stop" and lines and not result.stderr and all(
        "stopped" in line or "not running" in line for line in lines
    )
    if result.returncode and not stopped:
        raise RuntimeError(f"Supervisor {action} failed: {result.stdout[-500:]}")


def clear_runtime_tables():
    database = os.environ.get("BUSAGENT_RESET_DATABASE", "busagent_arena")
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError("Invalid database name")
    base = ["/usr/bin/mariadb", "--socket=/run/mysqld/mysqld.sock", "-NB", "--raw"]
    tables = subprocess.check_output(
        [*base, database, "-e", "SHOW TABLES"], text=True, timeout=20
    ).splitlines()
    selected = [
        table
        for table in tables
        if re.fullmatch(r"busagent_[a-z_]+", table)
        and table != "busagent_migrations"
    ]
    statements = ["SET FOREIGN_KEY_CHECKS=0"]
    statements.extend(f"TRUNCATE TABLE `{table}`" for table in selected)
    statements.append("SET FOREIGN_KEY_CHECKS=1")
    subprocess.run(
        [*base, database],
        input=";\n".join(statements) + ";\n",
        text=True,
        check=True,
        capture_output=True,
        timeout=60,
    )
    return len(selected)


def remove_runtime(path):
    resolved_parent = path.parent.resolve()
    if not resolved_parent.is_relative_to(ROOT.resolve()):
        raise ValueError("Runtime path escaped the checkout")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def api(port, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.load(response)


def reset_vision():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        "http://127.0.0.1:5570/reset",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=15) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError("Vision session reset failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--epoch", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9-]{32,40}", args.operation):
        parser.error("invalid operation id")
    if not re.fullmatch(r"[a-f0-9-]{32,40}", args.epoch):
        parser.error("invalid epoch")
    operation = {
        "id": args.operation,
        "scene_id": args.scene,
        "epoch": args.epoch,
        "phase": "stopping",
        "message": "正在停止当前任务和仿真…",
        "started_at": time.time(),
    }
    try:
        config = scene_config(args.scene)
        # Leave enough time for the HTTP response to reach the UI.
        time.sleep(1.5)
        update(operation, "stopping", "正在停止当前任务和仿真…")
        supervisor("stop", SERVICES)
        update(operation, "clearing", "正在清理当前场景的历史和缓存…")
        reset_vision()
        cleared_tables = clear_runtime_tables()
        for path in RUNTIME_PATHS:
            remove_runtime(path)
        profile = ROOT / "output/services/arena-scene.json"
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text(json.dumps({"config": str(config)}))
        update(
            operation,
            "starting",
            "正在加载新场景和模型服务…",
            cleared_tables=cleared_tables,
        )
        supervisor("start", ("arena", "busagent"))
        update(operation, "waiting", "等待场景和任务系统就绪…")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                workspace = api(7861, "/api/workspace")
                queue = api(3100, "/v1/tasks")
                if (
                    workspace.get("scene_id") == args.scene
                    and queue.get("enabled")
                    and not queue.get("goals")
                ):
                    update(
                        operation,
                        "completed",
                        "场景已就绪，当前场景历史已清空。",
                        finished_at=time.time(),
                    )
                    return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError("Scene startup timed out")
    except Exception as error:
        update(operation, "failed", str(error), finished_at=time.time())


if __name__ == "__main__":
    main()
