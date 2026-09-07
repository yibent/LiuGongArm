"""Record an autonomous natural-language Q8 run, without supplying fixture truth.

Requires an idle controller and no runnable goals. Resets the test fixture before
submission, resumes only the newly submitted goal, and restores the queue pause
on exit. Initial/final workspace snapshots are for independent offline evaluation.
"""
import argparse
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from datetime import datetime

INSTRUCTION = (
    "请完成这个工业装箱任务：先识别来料零件最多的区域，优先把该区域的金属圆柱零件"
    "依次装入桌面上的蓝色分格料箱。每件都必须闭口实心端朝上、稳稳放进空格，"
    "遇到倒置的零件要翻正，料箱里已有横倒的零件也要自行发现并扶正。"
    "如果抓取或放置失败，请根据实际反馈调整并继续。料箱没满就继续装到满，"
    "装满后把这只料箱搬到旁边高台上的另一只料箱上叠放整齐。"
    "请自主选择具体零件和空格，按实际执行结果确认完成。"
)


def api(base, path, body=None):
    request = urllib.request.Request(base + path,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bus", default="http://127.0.0.1:3100")
    p.add_argument("--arena", default="http://127.0.0.1:7861")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--queue-db", help="Optional local read-only queue database; includes hidden planning failures")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def frames(prefix):
        for camera in ("scene", "side"):
            with urllib.request.urlopen(args.arena + "/api/frame/" + camera + ".jpg", timeout=12) as response:
                (args.output / (prefix + "-" + camera + ".jpg")).write_bytes(response.read())

    before = api(args.bus, "/v1/tasks")
    save("queue-before", before)
    if any(g["state"] not in {"completed", "cancelled", "blocked", "paused"} for g in before["goals"]):
        raise RuntimeError("Existing runnable goals must be isolated first")
    live = api(args.arena, "/api/status")
    save("status-before", live)
    if live.get("held_object") or live.get("phase") not in {"idle", "completed", "failed", "hold"}:
        raise RuntimeError("Controller is not idle and empty")
    save("workspace-before-reset", api(args.arena, "/api/workspace"))
    frames("before-reset")
    if not args.no_reset:
        cid = "acceptance-reset-" + uuid.uuid4().hex
        api(args.arena, "/api/command", {"command_id": cid, "skill": "workspace", "params": {"action": "reset", "scope": "all"}})
        for _ in range(40):
            reset = api(args.arena, "/api/commands/" + cid)
            if reset["state"] not in {"accepted", "running"}:
                break
            time.sleep(.25)
        save("reset", reset)
        if reset["state"] != "completed":
            raise RuntimeError("Fixture reset failed")
        time.sleep(2)
    save("workspace-initial", api(args.arena, "/api/workspace"))
    frames("initial")
    correlation = "industrial-acceptance-" + uuid.uuid4().hex[:12]
    started = time.monotonic()
    event = api(args.bus, "/v1/events", {"source_agent_id": "robot.stt", "event_type": "intent.created",
        "correlation_id": correlation, "payload": {"text": args.instruction}})
    goal_id = "goal_" + event["event_id"]
    save("submission", {"goal_id": goal_id, "correlation_id": correlation, "instruction": args.instruction,
        "event": event, "timeout_s": args.timeout})
    print(json.dumps({"goal_id": goal_id, "correlation_id": correlation}, ensure_ascii=False), flush=True)
    resumed = False
    current = None
    last = None
    last_capture = -20
    try:
        while time.monotonic() - started < args.timeout:
            elapsed = time.monotonic() - started
            queue = api(args.bus, "/v1/tasks")
            current = next((g for g in queue["goals"] if g["id"] == goal_id), None)
            if current is None and args.queue_db:
                if not args.queue_db.replace('_', '').isalnum():
                    raise ValueError('Invalid database identifier')
                sql = 'SELECT payload FROM ' + args.queue_db + '.busagent_goal_queue WHERE id="arm-01"'
                raw = subprocess.check_output(['mariadb', '--socket=/run/mysqld/mysqld.sock', '-NB', '--raw', '-e', sql], text=True)
                internal = json.loads(raw)
                current = next((g for g in internal['goals'] if g['id'] == goal_id), None)
            if current and not resumed and not current.get('interaction') and current['state'] == 'queued':
                api(args.bus, "/v1/tasks/control", {"action": "resume", "id": goal_id})
                resumed = True
            live = api(args.arena, "/api/status")
            row = {"elapsed_s": round(elapsed, 2), "state": current and current["state"],
                "model_calls": current and current.get("model_calls"),
                "steps": [{k:s.get(k) for k in ("id", "skill", "state", "summary")} for s in (current or {}).get("steps", [])],
                "phase": live.get("phase"), "held_object": live.get("held_object"), "command_id": live.get("command_id")}
            signature = json.dumps({k:v for k,v in row.items() if k != "elapsed_s"}, ensure_ascii=False)
            if signature != last:
                with (args.output / "timeline.jsonl").open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                save("goal-current", current)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                last = signature
            if elapsed - last_capture >= 10:
                frames("frame-%05d" % elapsed)
                save("workspace-%05d" % elapsed, api(args.arena, "/api/workspace"))
                last_capture = elapsed
            if current and current["state"] in {"completed", "blocked", "cancelled", "paused"}:
                break
            time.sleep(1)
        goal_elapsed = ((datetime.fromisoformat(current['updated_at'].replace('Z', '+00:00')) -
            datetime.fromisoformat(current['created_at'].replace('Z', '+00:00'))).total_seconds() if current else None)
        save("result", {"elapsed_s": time.monotonic() - started, "goal_elapsed_s": goal_elapsed, "goal": current,
            "timed_out": time.monotonic() - started >= args.timeout})
    finally:
        if current and current["state"] not in {"completed", "blocked", "cancelled", "paused"}:
            api(args.bus, "/v1/tasks/control", {"action": "pause", "id": goal_id})
        elif before["paused"]:
            api(args.bus, "/v1/tasks/control", {"action": "pause"})
        save("queue-after", api(args.bus, "/v1/tasks"))
        save("workspace-final", api(args.arena, "/api/workspace"))
        save("status-final", api(args.arena, "/api/status"))
        frames("final")


if __name__ == "__main__":
    main()
