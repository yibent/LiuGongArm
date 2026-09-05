"""Thread-safe runtime prompt control and dual-camera monitoring HTTP API."""

from __future__ import annotations

import json
import threading
import time
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
        self._raw_frames: dict[str, Any] = {}
        self._detections: dict[str, list[dict[str, Any]]] = {}
        self._capture_meta = {}
        self._scene_captures = []
        self._inventory = None
        self._models: dict[str, Any] = {}
        self._error: str | None = None
        self.motion = None
        self.grounding = None
        self.grasp = None
        self.memory_store = None

    def snapshot(self) -> RuntimeConfig:
        with self._lock:
            return replace(self._config)

    def set_prompt(self, prompt: str) -> RuntimeConfig:
        value = str(prompt).strip()
        if not value:
            raise ValueError("prompt must not be empty")
        with self._lock:
            self._config.prompt = value
            # A new target must not retain the previous object's visual prompt.
            self._config.memory_id = None
            self._config.memory_view = None
            self._config.prompt_version += 1
            self._config.find_epoch += 1
            return replace(self._config)

    def force_find(self) -> RuntimeConfig:
        with self._lock:
            self._config.find_epoch += 1
            return replace(self._config)

    def set_follow_enabled(self, enabled: bool) -> bool:
        if enabled and self.motion is not None:
            self.motion.release()
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

    def publish(self, view: str, result: FrameResult, frame_bgr: Any = None,
                *, prompt_version=None, observed_at=None) -> None:
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
            if frame_bgr is not None:
                self._raw_frames[str(view)] = frame_bgr.copy()
            self._detections[str(view)] = detections
            self._capture_meta[str(view)] = (prompt_version, observed_at)
            if view == "scene" and frame_bgr is not None:
                self._scene_captures.append((prompt_version, observed_at, self._raw_frames[view]))
                self._scene_captures = self._scene_captures[-4:]
            if ok:
                self._frames[str(view)] = encoded.tobytes()

    def remember(self, label: str, *, memory_id: str | None = None,
                 aliases: list[str] | None = None, metadata: dict | None = None,
                 fresh=False, selected_box=None, selected_at=None) -> dict[str, Any]:
        if self.memory_store is None:
            raise RuntimeError("object memory store is not configured")
        with self._lock:
            frames = dict(self._raw_frames)
            if fresh:
                frames = {v: f for v, f in frames.items()
                          if self._capture_meta.get(v, (None, None))[0] == self._config.prompt_version
                          and self._capture_meta[v][1] is not None
                          and 0 <= time.monotonic() - self._capture_meta[v][1] < 3}
            detections = {
                view: items[0]
                for view, items in self._detections.items() if len(items) == 1 and view in frames
            }
            if selected_box is not None:
                # Pair the grounded box with its exact RGB frame, not the next
                # tracker update. Other-camera identity is not assumed.
                sample = next((f for version, at, f in reversed(self._scene_captures)
                               if version == self._config.prompt_version and at == selected_at
                               and at is not None and 0 <= time.monotonic() - at < 3), None)
                if sample is None:
                    raise ValueError("定位画面已更新，未写入可能错配的物体记忆，请重试记忆指令。")
                frames = {"scene": sample}
                detections = {"scene": dict(xyxy=selected_box, score=1.)}
            if fresh and "scene" not in detections:
                raise ValueError("没有唯一且新鲜的目标画面，未写入记忆。")
        memory = self.memory_store.remember(
            label, frames, detections, memory_id=memory_id,
            aliases=aliases, metadata=metadata,
        )
        return asdict(memory)

    def recall(self, memory_id: str, view: str | None = None) -> RuntimeConfig:
        memory = self.memory_store.get(memory_id) if self.memory_store else None
        if memory is None:
            raise ValueError(f"unknown object memory: {memory_id}")
        with self._lock:
            self._config.prompt = memory.metadata.get("prompt") or memory.label
            self._config.memory_id = memory_id
            self._config.memory_view = view
            self._config.prompt_version += 1
            self._config.find_epoch += 1
            return replace(self._config)

    def frame(self, view: str) -> bytes | None:
        with self._lock:
            return self._frames.get(view)

    def status(self) -> dict[str, Any]:
        motion = self.motion.status() if self.motion is not None else None
        grounding = self.grounding.snapshot() if self.grounding else None
        grounding_diagnostics = self.grounding.diagnostics() if self.grounding else None
        grasp_status = self.grasp.status() if self.grasp else None
        with self._lock:
            return {
                "prompt": self._config.prompt,
                "prompt_version": self._config.prompt_version,
                "find_epoch": self._config.find_epoch,
                "memory_id": self._config.memory_id,
                "memory_view": self._config.memory_view,
                "fast_backend": self._config.fast_backend,
                "slow_interval": self._config.slow_interval,
                "follow_enabled": self._follow_enabled,
                "models": self._models,
                "views": dict(self._views),
                "error": self._error,
                "motion": motion,
                "grounding": grounding,
                "grounding_diagnostics": grounding_diagnostics,
                "grasp": grasp_status,
                "memories": self.memory_store.list() if self.memory_store else [],
                "capabilities": self.capabilities(),
            }

    def capabilities(self) -> dict[str, Any]:
        from mr_liu.motion.commands import MOTION_SKILLS
        return {
            "skills": ["select_target", "perceive", "follow", "hold", "stop", "status", "capabilities"]
                      + (sorted(MOTION_SKILLS - {"hold", "stop"}) if self.motion is not None else [])
                      + (["remember", "recall", "forget", "list_memories"] if self.memory_store is not None else [])
                      + ["scene_inventory"]
                      + (["grasp"] if self.grasp is not None else []),
            "unsupported": ["plan_grasp", "transport", "place", "verify_placement"] + ([] if self.grasp else ["grasp"]),
            "grasp": self.grasp.capabilities() if self.grasp else None,
            "message": "当前支持识别、跟随、暂停和停止" + (
                "，以及关节转动、末端移动、归位、夹爪开合、调速和物体相对定位。"
                if self.motion is not None else "。基础运动控制尚未接入。") + (
                "支持物体多视角记忆、回忆定位和删除记忆。" if self.memory_store is not None else ""
            ) + (
                "支持单物体抓取，默认双RGB-D辅助验证与机器人自遮罩；失败后通过对话确认，符合条件才重新观察并重试一次，持物不确定时停止。未通过泛化验收。放置未实现，夹爪闭合不代表抓取成功。"
                if self.grasp else "抓取和放置尚未实现，夹爪闭合不代表抓取成功。"),
        }


def execute_control_command(
    control: VisionRuntimeControl,
    skill: str,
    params: dict[str, Any] | None = None,
    command_id: str | None = None,
) -> dict[str, Any]:
    """Execute one BusAgent semantic skill against the live vision loop."""
    values = params or {}
    from mr_liu.motion.commands import MOTION_SKILLS
    if skill in {"hold", "stop"} and control.grounding:
        control.grounding.cancel()
    if skill in {"hold", "stop"} and control.grasp:
        control.grasp.cancel()
    if skill == "grasp" and control.grasp:
        return control.grasp.submit(values, command_id)
    if skill == "capabilities":
        return {"ok": True, **control.capabilities()}
    if skill in MOTION_SKILLS and control.motion is not None:
        if values.get('grounding') is not None and (not control.grounding or not control.grounding.valid(values['grounding'])):
            return {"ok": False, "message": "物体定位已过期或被中断，请重新下达定位移动指令。"}
        result = control.motion.submit(skill, values, command_id)
        if skill != "set_speed" and result.get("state") in {"accepted", "started"}:
            control.set_follow_enabled(False)
        return result
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
    if skill == "remember":
        label = str(values.get("label") or values.get("category") or control.snapshot().prompt).strip()
        target = values.get("target")
        grounded = None
        metadata = values.get("metadata") if isinstance(values.get("metadata"), dict) else {}
        if isinstance(target, dict):
            grounded = control.grounding.resolve(control, dict(category=target.get("category"),
                color=(target.get("attributes") or {}).get("color"),
                selector=target.get("spatial_ref"), offset_m=[0, 0, 0]))
            if not grounded.get("ok"):
                return grounded
            metadata = {**metadata, "prompt": grounded["prompt"], "target": target}
        result = control.remember(
            label,
            memory_id=values.get("memory_id"),
            aliases=values.get("aliases") if isinstance(values.get("aliases"), list) else None,
            metadata=metadata, fresh=True,
            selected_box=grounded.get("xyxy") if grounded else None,
            selected_at=grounded.get("observed_at") if grounded else None,
        )
        return {"ok": True, "message": f"已记住{label}的视觉外观；再次使用时会重新定位，不复用旧坐标。", "memory": result}
    if skill == "list_memories":
        items = control.memory_store.list() if control.memory_store else []
        return {"ok": True, "memories": items, "message": "已保存的物体：" + "、".join(i["label"] for i in items) if items else "尚未保存物体记忆。"}
    if skill == "scene_inventory":
        with control._lock:
            inventory = control._inventory
        if inventory is None or time.monotonic() - inventory["observed_at"] > 3:
            return dict(ok=False, message="场景全览尚无新鲜画面，不能确认桌面物品。")
        return dict(ok=True, **inventory)
    if skill == "recall":
        memory_id = str(values.get("memory_id") or "").strip()
        if not memory_id:
            raise ValueError("recall requires memory_id")
        memory = control.memory_store.get(memory_id) if control.memory_store else None
        if memory is None:
            return dict(ok=False, message="这条物体记忆已不存在。")
        if control.grounding:
            result = control.grounding.resolve(control, dict(category=memory.metadata.get("prompt") or memory.label, memory_id=memory_id))
            return {**result, "memory_id": memory_id}
        cfg = control.recall(memory_id, values.get("view"))
        return {"ok": True, "message": "已启用记忆外观，尚未确认找到物体。", "memory_id": cfg.memory_id, "prompt": cfg.prompt}
    if skill == "forget":
        memory_id = str(values.get("memory_id") or "").strip()
        if not memory_id or control.memory_store is None:
            raise ValueError("forget requires memory_id")
        deleted = control.memory_store.delete(memory_id)
        if deleted and control.snapshot().memory_id == memory_id:
            control.set_prompt(control.snapshot().prompt)
        return {"ok": deleted, "memory_id": memory_id, "message": "已删除这条物体记忆。" if deleted else "这条物体记忆不存在。"}
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
button,select{padding:9px 15px;margin-left:6px}.views{display:flex;gap:18px;margin-top:20px}.view{width:48%}
img{width:100%;background:#222}pre{white-space:pre-wrap;color:#9fd}</style></head>
<body><h2>Florence FIND + YOLOE TRACK</h2><input id="p" placeholder="red cube">
<button onclick="setPrompt()">切换目标</button><button onclick="forceFind()">立即 FIND</button>
<button onclick="remember()">记住当前目标</button><select id="m"><option value="">选择记忆</option></select><button onclick="recall()">回放记忆</button>
<div class="views"><div class="view"><h3>顶部</h3><img id="scene"></div>
<div class="view"><h3>腕部</h3><img id="wrist"></div></div><pre id="s"></pre>
<script>
async function post(path, body={}){await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})}
async function setPrompt(){await post('/api/prompt',{prompt:document.querySelector('#p').value})}
async function forceFind(){await post('/api/find')}
async function remember(){await post('/api/command',{skill:'remember',params:{label:document.querySelector('#p').value}});await tick()}
async function recall(){let id=document.querySelector('#m').value;if(id)await post('/api/command',{skill:'recall',params:{memory_id:id}});await tick()}
async function tick(){let s=await (await fetch('/api/status')).json();document.querySelector('#s').textContent=JSON.stringify(s,null,2);
let select=document.querySelector('#m'), current=select.value;select.innerHTML='<option value="">选择记忆</option>';
for(let item of (s.memories||[])){let option=document.createElement('option');option.value=item.memory_id;option.textContent=item.label+' ('+item.memory_id+')';select.appendChild(option)}select.value=current;
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
                if path == "/api/capabilities":
                    self._json(control.capabilities())
                    return
                if path == "/api/memories":
                    self._json({"memories": control.memory_store.list() if control.memory_store else []})
                    return
                if path.startswith("/api/commands/"):
                    from urllib.parse import unquote
                    result = control.motion.result(unquote(path.removeprefix("/api/commands/"))) if control.motion else None
                    self._json(result or {"ok": False, "error": "command not found"}, HTTPStatus.OK if result else HTTPStatus.NOT_FOUND)
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
                        if not body.get("enabled", True) and control.motion is not None:
                            self._json(execute_control_command(control, "hold"))
                            return
                        enabled = control.set_follow_enabled(bool(body.get("enabled", True)))
                        self._json({"message": "follow state updated", "follow_enabled": enabled})
                        return
                    if path == "/api/command":
                        skill = str(body.get("skill", "")).strip()
                        params = body.get("params", {})
                        if not isinstance(params, dict):
                            raise ValueError("command params must be an object")
                        result = execute_control_command(control, skill, params, body.get("command_id"))
                        self._json(result, HTTPStatus.ACCEPTED if result.get("state") == "accepted" else HTTPStatus.OK)
                        return
                    if path == "/api/ground":
                        if control.grounding is None:
                            self._json({"ok": False, "message": "当前未接入RGB-D物体定位。"})
                        else:
                            self._json(control.grounding.resolve(control, body))
                        return
                    if path == "/api/memory":
                        result = execute_control_command(control, "remember", body)
                        self._json(result)
                        return
                    if path == "/api/recall":
                        result = execute_control_command(control, "recall", body)
                        self._json(result)
                        return
                    if path == "/api/forget":
                        result = execute_control_command(control, "forget", body)
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
