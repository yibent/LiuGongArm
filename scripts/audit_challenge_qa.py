"""Run bounded, real BusAgent/Arena tests for the Q8 audit fixture.

Build build_challenge_qa_fixture.py and start its config first. This runner
supplies natural language, never ground truth or object poses, to BusAgent.
Timeouts pause only the submitted test goal and preserve its evidence.
"""
import argparse
import json
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

INSTRUCTIONS = {
    "packing": "帮我把零件最多的区域装箱，每件都要闭口实心端朝上放入料箱格中。如果放倒了，要自己发现并纠正。",
    "slot": "把一个金属圆柱零件放入来料区蓝色料箱的第二排第三个格子，闭口实心端朝上。",
    "carry": "帮我搬运一下来料区的料箱盒。如果没有装满，先继续装到满，再搬到旁边高台上的另一只料箱上叠放整齐。",
    "repair": "请把来料区蓝色料箱里横倒的金属圆柱零件扶正，让闭口实心端朝上放回一个格子。",
}


def api(base, path, body=None):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.load(response)


def save(out, name, value):
    (out / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def baseline(args):
    save(args.output, "fixture-workspace", api(args.arena, "/api/workspace"))
    for camera in ("scene", "side"):
        with urllib.request.urlopen(f"{args.arena}/api/frame/{camera}.jpg", timeout=8) as response:
            (args.output / f"fixture-{camera}.jpg").write_bytes(response.read())
    command_id = "qa-audit-" + uuid.uuid4().hex
    api(args.arena, "/api/command", {
        "command_id": command_id, "skill": "perceive",
        "params": {"scope": "scene", "scene_mode": "inventory"},
    })
    deadline = time.monotonic() + args.timeout
    result = None
    while time.monotonic() < deadline:
        result = api(args.arena, "/api/commands/" + command_id)
        if result["state"] not in ("accepted", "running"):
            save(args.output, "inventory", result)
            print(json.dumps({key: result.get(key) for key in ("ok", "elapsed_s", "message")}, ensure_ascii=False))
            return
        time.sleep(1)
    save(args.output, "inventory-timeout", {"command_id": command_id, "last_result": result})
    raise TimeoutError("Inventory did not finish; see inventory-timeout.json")


def goal(args):
    source = INSTRUCTIONS[args.case]
    if args.goal_id:
        goal_id = args.goal_id
    else:
        save(args.output, args.case + "-before", api(args.arena, "/api/workspace"))
        event = api(args.bus, "/v1/events", {
            "source_agent_id": "robot.stt", "event_type": "intent.created",
            "correlation_id": "qa-audit-" + args.case + "-" + uuid.uuid4().hex[:8],
            "payload": {"text": source},
        })
        goal_id = "goal_" + event["event_id"]
        save(args.output, args.case + "-submission", {"event": event, "goal_id": goal_id})
    deadline = time.monotonic() + args.timeout
    started = time.monotonic()
    last = None
    timeline = []
    current = None
    while time.monotonic() < deadline:
        current = next((item for item in api(args.bus, "/v1/tasks")["goals"] if item["id"] == goal_id), None)
        # Event acknowledgement can precede the asynchronous goal materialization.
        if current is None:
            time.sleep(.25)
            continue
        state = [current["state"], current["model_calls"], [[step["skill"], step["state"]] for step in current["steps"]]]
        if state != last:
            row = {"at": time.time(), "state": state}
            timeline.append(row)
            print(json.dumps({"case": args.case, **row}, ensure_ascii=False), flush=True)
            last = state
        if current["state"] in ("completed", "blocked", "cancelled"):
            break
        time.sleep(1)
    else:
        save(args.output, args.case + "-before-timeout", {"goal_id": goal_id, "goal": current})
        api(args.bus, "/v1/tasks/control", {"action": "pause", "id": goal_id})
        current = next((item for item in api(args.bus, "/v1/tasks")["goals"] if item["id"] == goal_id), current)
    observed_at = time.time()
    created_at = datetime.fromisoformat(current["created_at"].replace("Z", "+00:00")).timestamp() if current else None
    report = {
        "instruction": source, "elapsed_s": time.monotonic() - started,
        "goal_age_s_at_observation": observed_at - created_at if created_at else None,
        "attached_to_existing_goal": bool(args.goal_id), "goal": current,
        "timeline": timeline, "workspace": api(args.arena, "/api/workspace"),
    }
    save(args.output, args.case, report)
    print(json.dumps({"case": args.case, "goal_id": goal_id, "state": current and current["state"], "elapsed_s": report["elapsed_s"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["baseline", *INSTRUCTIONS])
    parser.add_argument("--arena", default="http://127.0.0.1:7861")
    parser.add_argument("--bus", default="http://127.0.0.1:3100")
    parser.add_argument("--output", type=Path, default=Path("output/challenge-qa"))
    parser.add_argument("--timeout", type=float, default=210)
    parser.add_argument("--goal-id", help="Observe an already submitted goal without submitting it again")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not api(args.arena, "/health")["ready"]:
        parser.error("Arena is not ready")
    if args.case == "baseline":
        baseline(args)
    else:
        if not api(args.bus, "/v1/tasks")["enabled"]:
            parser.error("BusAgent intelligence is disabled")
        goal(args)


if __name__ == "__main__":
    main()
