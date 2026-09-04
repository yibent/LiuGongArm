from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import IMAGE_EXTS, VIDEO_EXTS, FindTrackPipeline
from .types import Detection, RuntimeConfig

LOGGER = logging.getLogger("find_and_track.webui")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = HERE / "static" / "index.html"
RUNS = ROOT / "runs"
UPLOADS = RUNS / "uploads"
OUTPUTS = RUNS / "outputs"


class ConfigIn(BaseModel):
    prompt: str = "person"
    fast_backend: str = "yoloe"
    slow_interval: int = 45
    conf: float = Field(0.25, ge=0.01, le=1.0)
    imgsz: int = Field(640, ge=320, le=1280)


class AppState:
    def __init__(self, weights: Path):
        self.pipeline = FindTrackPipeline(yoloe_weights=weights)
        self.cfg = RuntimeConfig()
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.running = False
        self.done = False
        self.source_path: str | None = None
        self.source_kind: str = ""
        self.latest_jpeg: bytes | None = None
        self.stats: dict = {}
        self.detections: list[dict] = []
        self.load_error: str | None = None
        self.output_path: str | None = None
        self._placeholder = _placeholder_jpeg("waiting for source")


STATE: AppState | None = None


def _det_dict(det: Detection) -> dict:
    return {
        "xyxy": [float(x) for x in det.xyxy.tolist()],
        "label": det.label,
        "score": float(det.score),
        "track_id": det.track_id,
        "source": det.source,
    }


def _placeholder_jpeg(text: str) -> bytes:
    img = np.zeros((480, 854, 3), dtype=np.uint8)
    img[:] = (16, 18, 22)
    cv2.putText(img, text, (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


def _apply_config(body: ConfigIn, bump_prompt: bool = False, force_find: bool = False) -> None:
    assert STATE is not None
    with STATE.lock:
        prev = STATE.cfg.prompt
        STATE.cfg.prompt = body.prompt.strip() or STATE.cfg.prompt
        STATE.cfg.fast_backend = "cv" if body.fast_backend == "cv" else "yoloe"  # type: ignore[assignment]
        STATE.cfg.slow_interval = int(body.slow_interval)
        STATE.cfg.conf = float(body.conf)
        STATE.cfg.imgsz = int(body.imgsz)
        if bump_prompt:
            STATE.cfg.prompt_version += 1
        if force_find or bump_prompt or STATE.cfg.prompt != prev:
            STATE.cfg.find_epoch += 1


def _load_models() -> None:
    assert STATE is not None
    try:
        STATE.pipeline.load(STATE.cfg.fast_backend)
        STATE.load_error = None
        LOGGER.info("Models loaded")
    except Exception as exc:  # noqa: BLE001
        STATE.load_error = str(exc)
        LOGGER.exception("Model load failed")


def _worker() -> None:
    assert STATE is not None
    STATE.running = True
    STATE.done = False
    STATE.cfg.stop = False
    source = STATE.source_path
    if not source:
        STATE.running = False
        STATE.load_error = "No source uploaded"
        return
    out = OUTPUTS / f"annotated_{Path(source).stem}.mp4"
    STATE.output_path = str(out)
    try:
        if not STATE.pipeline.loaded:
            STATE.pipeline.load(STATE.cfg.fast_backend)
        for result in STATE.pipeline.process_source(source, STATE.cfg, output=out if Path(source).suffix.lower() in VIDEO_EXTS else None):
            ok, buf = cv2.imencode(".jpg", result.image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            dets = [_det_dict(d) for d in result.slow] + [_det_dict(d) for d in result.fast]
            with STATE.lock:
                if ok:
                    STATE.latest_jpeg = buf.tobytes()
                STATE.stats = asdict(result.stats)
                STATE.detections = dets
            if STATE.cfg.stop:
                break
        STATE.done = True
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Worker failed")
        STATE.load_error = str(exc)
        STATE.latest_jpeg = _placeholder_jpeg(str(exc)[:80])
    finally:
        STATE.running = False


def create_app(weights: str | Path) -> FastAPI:
    global STATE
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    STATE = AppState(Path(weights))
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
    except Exception:  # noqa: BLE001
        LOGGER.warning("CUDA warmup failed", exc_info=True)
    app = FastAPI(title="Find and Track", version="1.0")
    threading.Thread(target=_load_models, name="model-loader", daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        assert STATE is not None
        return {
            "loaded": STATE.pipeline.loaded,
            "florence": STATE.pipeline.florence.ready,
            "florence_id": STATE.pipeline.florence.loaded_id,
            "yoloe": STATE.pipeline.yoloe.ready,
            "load_error": STATE.load_error,
            "running": STATE.running,
            "done": STATE.done,
            "source": STATE.source_path,
            "kind": STATE.source_kind,
            "stats": STATE.stats,
            "detections": STATE.detections,
            "prompt": STATE.cfg.prompt,
            "output": STATE.output_path,
            "has_frame": STATE.latest_jpeg is not None,
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        assert STATE is not None
        suffix = Path(file.filename or "upload.bin").suffix.lower()
        if suffix not in IMAGE_EXTS | VIDEO_EXTS:
            raise HTTPException(400, f"Unsupported file type: {suffix}")
        name = f"{uuid.uuid4().hex}{suffix}"
        dest = UPLOADS / name
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        STATE.source_path = str(dest)
        STATE.source_kind = "image" if suffix in IMAGE_EXTS else "video"
        STATE.latest_jpeg = None
        return {"path": str(dest), "name": file.filename, "kind": STATE.source_kind}

    @app.post("/api/sample")
    def sample():
        assert STATE is not None
        import ultralytics

        src = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if not src.exists():
            raise HTTPException(404, "Sample image not found")
        dest = UPLOADS / "sample_bus.jpg"
        shutil.copy(src, dest)
        STATE.source_path = str(dest)
        STATE.source_kind = "image"
        return {"path": str(dest), "name": "bus.jpg"}

    @app.post("/api/prompt")
    def set_prompt(body: ConfigIn):
        _apply_config(body, bump_prompt=True, force_find=True)
        return {"message": "提示已更新，下一帧将重新 FIND。", "version": STATE.cfg.prompt_version}

    @app.post("/api/find")
    def force_find(body: ConfigIn):
        _apply_config(body, bump_prompt=False, force_find=True)
        return {"message": "已请求立即 FIND。"}

    @app.post("/api/start")
    def start(body: ConfigIn):
        assert STATE is not None
        if not STATE.pipeline.loaded:
            raise HTTPException(503, STATE.load_error or "模型仍在加载，请稍候")
        if not STATE.source_path:
            raise HTTPException(400, "请先上传视频或图片")
        if STATE.running:
            raise HTTPException(409, "已在运行，先停止再开始")
        _apply_config(body, bump_prompt=True, force_find=True)
        STATE.cfg.stop = False
        STATE.pipeline.reset()
        STATE.worker = threading.Thread(target=_worker, name="find-track", daemon=True)
        STATE.worker.start()
        return {"message": "started", "source": STATE.source_path}

    @app.post("/api/stop")
    def stop():
        assert STATE is not None
        STATE.cfg.stop = True
        return {"message": "stopping"}

    @app.get("/api/snapshot.jpg")
    def snapshot():
        assert STATE is not None
        data = STATE.latest_jpeg or STATE._placeholder
        return StreamingResponse(iter([data]), media_type="image/jpeg")

    @app.get("/api/stream")
    async def stream():
        boundary = "frame"

        async def gen():
            import asyncio

            last = None
            idle = 0
            while True:
                data = STATE.latest_jpeg if STATE is not None else None
                if data is None:
                    data = STATE._placeholder if STATE is not None else b""
                if data is not last or idle > 15:
                    yield (
                        b"--" + boundary.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data + b"\r\n"
                    )
                    last = data
                    idle = 0
                else:
                    idle += 1
                if STATE is not None and STATE.done and not STATE.running:
                    await asyncio.sleep(0.4)
                    continue
                await asyncio.sleep(0.03)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    return app


def serve(host: str, port: int, weights: str | Path) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(weights)
    LOGGER.info("WebUI http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
