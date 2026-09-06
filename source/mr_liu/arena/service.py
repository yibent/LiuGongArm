"""BusAgent-compatible HTTP bridge. HTTP threads never access Isaac objects."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from urllib.parse import unquote


class CommandQueue:
    def __init__(self, ledger=None):
        self.queue = Queue()
        self.ledger = Path(ledger) if ledger else None
        self.commands = json.loads(self.ledger.read_text()) if self.ledger and self.ledger.exists() else {}
        for key, row in self.commands.items():
            if row["result"]["state"] in {"accepted", "running"}:
                row["result"] = {"ok": False, "state": "failed", "command_id": key,
                    "message": "Controller restarted; prior physical outcome unknown, command will not replay"}
        self.lock = threading.Lock()

    def _save(self):
        if self.ledger:
            temporary = self.ledger.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.commands, ensure_ascii=False))
            temporary.replace(self.ledger)

    def submit(self, command):
        key = command.get("command_id")
        if not isinstance(key, str) or not key or len(key) > 160:
            raise ValueError("A bounded command_id is required")
        with self.lock:
            if key in self.commands:
                existing = self.commands[key]
                if existing["request"] != command:
                    raise ValueError("command_id already used with a different payload")
                return existing["result"]
            if any(row["result"]["state"] in {"accepted", "running"} for row in self.commands.values()):
                raise ValueError("Panda is busy")
            result = {"ok": True, "state": "accepted", "command_id": key, "message": "Queued for Arena execution"}
            self.commands[key] = {"request": command, "result": result}
            self._save()
            self.queue.put_nowait(command)
            return result

    def result(self, key):
        with self.lock:
            return self.commands.get(key, {}).get("result")

    def update(self, key, result):
        with self.lock:
            self.commands[key]["result"] = result
            self._save()

    def claim(self, key):
        """Serialize the accepted-to-running transition with stop requests."""
        with self.lock:
            if self.commands[key]["result"]["state"] != "accepted":
                return False
            self.commands[key]["result"] = {"ok": True, "state": "running", "command_id": key,
                                              "message": "Arena executing"}
            self._save()
            return True

    def interrupt(self, stop_event):
        with self.lock:
            for key, row in self.commands.items():
                if row["result"]["state"] == "accepted":
                    row["result"] = {"ok": False, "state": "cancelled", "command_id": key,
                                     "message": "Cancelled before execution"}
                elif row["result"]["state"] == "running":
                    stop_event.set()
            self._save()


def serve(runtime, app, port):
    commands = CommandQueue(runtime.output / "commands.json")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def respond(self, code, data, content_type="application/json"):
            blob = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if content_type == "application/json" else data
            self.send_response(code); self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            try: self.wfile.write(blob)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in {"/health", "/healthz"}: return self.respond(200, {"ready": True, "service": "arena-panda"})
            if path == "/api/status": return self.respond(200, runtime.snapshot)
            if path == "/api/capabilities": return self.respond(200, runtime.capabilities())
            if path.startswith("/api/commands/"):
                result = commands.result(unquote(path.rsplit("/", 1)[1]))
                if result and result["state"] == "running":
                    result = {**result, "phase": runtime.snapshot.get("phase"),
                              "progress_seq": runtime.snapshot.get("sequence"),
                              "held_object": runtime.snapshot.get("held_object"),
                              "vision": runtime.snapshot.get("vision")}
                return self.respond(200 if result else 404, result or {"ok": False, "error": "unknown command"})
            if path.startswith("/api/frame/"):
                key = path.rsplit("/", 1)[1].removesuffix(".jpg")
                frame = runtime.frames.get(key)
                return self.respond(200, frame, "image/jpeg") if frame else self.respond(503, {"error": "frame unavailable"})
            return self.respond(404, {"error": "not found"})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 65536: raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if self.path != "/api/command": return self.respond(404, {"error": "not found"})
                skill = body.get("skill")
                if skill in {"hold", "stop"}:
                    commands.interrupt(runtime.stop_requested)
                    return self.respond(200, {"ok": True, "state": "completed", "command_id": body.get("command_id"),
                                              "message": "Stop requested; active command reports measured termination separately"})
                if skill in {"status", "capabilities"}:
                    return self.respond(200, {"ok": True, "state": "completed", "command_id": body.get("command_id"),
                                              "message": "Arena Panda state", "status": runtime.snapshot,
                                              **(runtime.capabilities() if skill == "capabilities" else {})})
                if skill not in runtime.capabilities()["skills"]: raise ValueError("Unsupported skill")
                return self.respond(202, commands.submit(body))
            except (ValueError, TypeError, KeyError) as exc:
                return self.respond(422, {"ok": False, "state": "failed", "message": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Arena Panda] ready at 127.0.0.1:{port}", flush=True)
    try:
        while app.is_running():
            try:
                command = commands.queue.get_nowait()
            except Empty:
                if runtime.stop_requested.is_set(): runtime.goal = runtime.tcp_pose()
                runtime.tick(check_stop=False)
                time.sleep(.002)
                continue
            if not commands.claim(command["command_id"]):
                continue
            commands.update(command["command_id"], runtime.execute(command))
            runtime.stop_requested.clear()
    finally:
        server.shutdown()
        runtime.vision_worker.close()
        runtime.pool.shutdown(wait=False, cancel_futures=True)
