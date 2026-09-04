"""Thread-safe runtime prompt control and dual-camera monitoring HTTP API."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import cv2

from find_and_track.types import Detection, FrameResult, RuntimeConfig


def _detection_dict(det: Detection) -> dict[str, Any]:
    return {
        "xyxy": [round(float(value), 2) for value in det.xyxy.tolist()],
        "label": det.label,
        "score": round(float(det.score), 4),
        "track_id": det.track_id,
        "source": det.source,
        "class_id": int(det.class_id),
    }


class VisionRuntimeControl:
    """Own mutable vision settings without exposing them across threads."""

    def __init__(self, config: RuntimeConfig, *, follow_enabled: bool = True) -> None:
        self._config = replace(config)
        self._follow_enabled = bool(follow_enabled)
        self._lock = threading.Lock()
        self._views: dict[str, dict[str, Any]] = {}
        self._frames: dict[str, bytes] = {}
        self._models: dict[str, Any] = {}
        self._error: str | None = None

    def snapshot(self) -> RuntimeConfig:
        with self._lock:
            return replace(self._config)

    def set_prompt(self, prompt: str) -> RuntimeConfig:
        value = str(prompt).strip()
        if not value:
            raise ValueError("prompt must not be empty")
        with self._lock:
            self._config.prompt = value
            self._config.prompt_version += 1
            self._config.find_epoch += 1
            return replace(self._config)

    def force_find(self) -> RuntimeConfig:
        with self._lock:
            self._config.find_epoch += 1
            return replace(self._config)

    def set_follow_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._follow_enabled = bool(enabled)
            return self._follow_enabled

    def follow_enabled(self) -> bool:
        with self._lock:
            return self._follow_enabled

    def set_model_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self._models = status
            self._error = None

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error = str(message)

    def publish(self, view: str, result: FrameResult) -> None:
        detections = [_detection_dict(det) for det in (result.fast or result.slow)]
        ok, encoded = cv2.imencode(
            ".jpg", result.image, [int(cv2.IMWRITE_JPEG_QUALITY), 84]
        )
        payload = {
            "stats": asdict(result.stats),
            "detections": detections,
        }
        with self._lock:
            self._views[str(view)] = payload
            if ok:
                self._frames[str(view)] = encoded.tobytes()

    def frame(self, view: str) -> bytes | None:
        with self._lock:
            return self._frames.get(view)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "prompt": self._config.prompt,
                "prompt_version": self._config.prompt_version,
                "find_epoch": self._config.find_epoch,
                "fast_backend": self._config.fast_backend,
                "slow_interval": self._config.slow_interval,
                "follow_enabled": self._follow_enabled,
                "models": self._models,
                "views": dict(self._views),
                "error": self._error,
            }


def execute_control_command(
    control: VisionRuntimeControl,
    skill: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one BusAgent semantic skill against the live vision loop."""
    values = params or {}
    if skill == "select_target":
        category = str(values.get("category") or values.get("prompt") or "").strip()
        if not category:
            raise ValueError("select_target requires category")
        attributes = values.get("attributes")
        if isinstance(attributes, dict):
            color = str(attributes.get("color") or "").strip()
            prompt = f"{color} {category}".strip()
        else:
            prompt = category
        cfg = control.set_prompt(prompt)
        return {"ok": True, "message": f"target set to {prompt}", "prompt": cfg.prompt}
    if skill == "perceive":
        cfg = control.force_find()
        return {"ok": True, "message": "new perception requested", "find_epoch": cfg.find_epoch}
    if skill == "follow":
        enabled = control.set_follow_enabled(bool(values.get("enabled", True)))
        return {"ok": True, "message": "following enabled" if enabled else "following disabled"}
    if skill in {"hold", "stop"}:
        control.set_follow_enabled(False)
        message = "robot following held" if skill == "hold" else "robot following stopped"
        return {"ok": True, "message": message}
    if skill == "status":
        return {"ok": True, "message": "status returned", "status": control.status()}
    raise NotImplementedError(
        f"skill {skill!r} is not implemented by the current follow-target controller"
    )


_CONTROL_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>SO-101 双视角追踪</title>
<style>body{font:15px system-ui;background:#111;color:#eee;margin:24px}input{width:420px;padding:9px}
button{padding:9px 15px;margin-left:6px}.views{display:flex;gap:18px;margin-top:20px}.view{width:48%}
img{width:100%;background:#222}pre{white-space:pre-wrap;color:#9fd}</style></head>
<body><h2>Florence FIND + YOLOE TRACK</h2><input id="p" placeholder="red cube">
<button onclick="setPrompt()">切换目标</button><button onclick="forceFind()">立即 FIND</button>
<div class="views"><div class="view"><h3>顶部</h3><img id="scene"></div>
<div class="view"><h3>腕部</h3><img id="wrist"></div></div><pre id="s"></pre>
<script>
async function post(path, body={}){await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})}
async function setPrompt(){await post('/api/prompt',{prompt:document.querySelector('#p').value})}
async function forceFind(){await post('/api/find')}
async function tick(){let s=await (await fetch('/api/status')).json();document.querySelector('#s').textContent=JSON.stringify(s,null,2);
let t=Date.now();document.querySelector('#scene').src='/api/frame/scene.jpg?t='+t;document.querySelector('#wrist').src='/api/frame/wrist.jpg?t='+t}
setInterval(tick,700);tick();</script></body></html>"""


class VisionControlServer:
    """Small embedded HTTP server used while Isaac Sim owns the main thread."""

    def __init__(self, control: VisionRuntimeControl, host: str = "127.0.0.1", port: int = 7861) -> None:
        self.control = control
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            return self.host, self.port
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            return self.address
        control = self.control

        class Handler(BaseHTTPRequestHandler):
            def _send(self, data: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
                self._send(json.dumps(payload, ensure_ascii=False).encode(), "application/json", status)

            def do_GET(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                if path == "/":
                    self._send(_CONTROL_HTML.encode(), "text/html; charset=utf-8")
                    return
                if path == "/api/status":
                    self._json(control.status())
                    return
                if path.startswith("/api/frame/") and path.endswith(".jpg"):
                    view = path.removeprefix("/api/frame/").removesuffix(".jpg")
                    frame = control.frame(view)
                    if frame is None:
                        self._json({"error": f"no frame for {view}"}, HTTPStatus.NOT_FOUND)
                    else:
                        self._send(frame, "image/jpeg")
                    return
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                try:
                    size = min(int(self.headers.get("Content-Length", "0")), 65536)
                    body = json.loads(self.rfile.read(size) or b"{}")
                    if path == "/api/prompt":
                        cfg = control.set_prompt(body.get("prompt", ""))
                        self._json(
                            {
                                "message": "prompt updated; both views will FIND again",
                                "prompt": cfg.prompt,
                                "prompt_version": cfg.prompt_version,
                            }
                        )
                        return
                    if path == "/api/find":
                        cfg = control.force_find()
                        self._json({"message": "FIND requested for both views", "find_epoch": cfg.find_epoch})
                        return
                    if path == "/api/follow":
                        enabled = control.set_follow_enabled(bool(body.get("enabled", True)))
                        self._json({"message": "follow state updated", "follow_enabled": enabled})
                        return
                    if path == "/api/command":
                        skill = str(body.get("skill", "")).strip()
                        params = body.get("params", {})
                        if not isinstance(params, dict):
                            raise ValueError("command params must be an object")
                        result = execute_control_command(control, skill, params)
                        result["command_id"] = body.get("command_id")
                        self._json(result)
                        return
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                except NotImplementedError as exc:
                    self._json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_IMPLEMENTED)
                except (ValueError, json.JSONDecodeError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="vision-control-http",
            daemon=True,
        )
        self._thread.start()
        return self.address

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
