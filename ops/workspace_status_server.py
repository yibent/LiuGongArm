"""Serve read-only scene lifecycle progress while BusAgent restarts.

The confirmed transition POST remains owned by BusAgent. This process only
exposes the atomically written journal and installed scene catalog so the
workbench can keep showing progress during a full scene reload.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "output/system/workspace.json"
CATALOG = ROOT / "configs/industrial_scenes.json"
PROFILE = ROOT / "output/services/arena-scene.json"


def read_json(path, fallback):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback


def active_scene():
    profile = read_json(PROFILE, {})
    configured = profile.get("config")
    if not isinstance(configured, str) or not configured:
        return "sorting"
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    config = read_json(path, {})
    scene = config.get("scene", {})
    return scene.get("id") if isinstance(scene.get("id"), str) else "sorting"


def status():
    journal = read_json(JOURNAL, {"epoch": "initial", "operation": None})
    return {
        "epoch": journal.get("epoch", "initial"),
        "operation": journal.get("operation"),
        "scene_id": active_scene(),
        "scenes": read_json(CATALOG, []),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route not in {"/status", "/health"}:
            self.send_error(404)
            return
        payload = {"ready": True} if route == "/health" else status()
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.send_error(405)

    def log_message(self, _format, *_args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 5599), Handler).serve_forever()
