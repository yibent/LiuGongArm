"""Opt-in, read-only demo video composition; never a grasp perception source."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHASES = {
    "find": "标签 → Florence / YOLOE 定位", "track": "CV 光流跟踪 + RGB-D 更新目标",
    "coarse": "快速粗接近：IK / 碰撞 / 移动中视觉检查", "handoff": "停止粗环 → 交接腕部精细抓取",
    "observe": "腕部观察 / 更新局部几何", "generate": "生成抓取候选",
    "select": "碰撞 / IK / 夹爪约束筛选", "pregrasp": "移动到预抓取位姿",
    "servo": "最新腕部观测 → 小步闭环对齐", "close": "夹爪闭合",
    "verify_close": "检查是否夹住", "lift": "抬升", "verify_lift": "视觉 + 夹爪验证",
    "recovery": "失败恢复：检查空手 / 退让 / 重定位",
    "succeeded": "抓取成功", "failed": "抓取失败 / 安全停止",
    "place_bootstrap": "抓取 → 核对物体与夹爪的相对位置",
    "place_handoff": "持物核验 / 交接精细放置", "place_find": "Florence 定位放置区域 + RGB-D",
    "place_observe": "新 RGB-D 核对目的地与支撑面", "place_align": "持物小步对齐目标区域",
    "place_descend": "闭环下降 / 检查物体底面高度", "place_release_check": "检查开爪与退出空间（尚未释放）",
    "place_open": "缓慢松爪", "place_retreat": "退出 / 检查物体是否留在原位",
    "place_verify": "多帧验证放置位置与稳定性", "place_succeeded": "放置视觉验证通过",
    "place_failed": "放置未完成 / 安全停止", "place_path_rejected": "持物路径检查未通过",
    "place_segment_payload": "同帧持物轮廓核验（SAM2 辅助）",
    "place_path_checked": "新点云 / 持物 / 实际关节路径检查通过",
}


class DemoVideoRecorder:
    """Encode wall-clock-timed rendered frames, holding gaps instead of hiding them.

    Sensor panels show each camera's latest RGB buffer, not a synchronized
    multi-camera acquisition. Recording adds rendering cost; these timings
    must not be used as the non-recording performance benchmark.
    """

    def __init__(self, output: Path, title: str, fault_m: float = 0., fps: int = 15,
                 *, coarse_fault_m: float = 0.):
        import imageio_ffmpeg

        self.output, self.fps = Path(output), fps
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.title, self.fault_m = title, fault_m
        self.coarse_fault_m = coarse_fault_m
        self.phase, self.attempt, self.detail = "observe", 1, ""
        self.font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 21)
        self.small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
        self.writer_log = self.output.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.output),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.writer_log)
        self.started = None
        self.frames = 0
        self.last_frame = None
        self.events = []
        self.closed = False

    def trace(self, event):
        self.phase = event.get("phase", self.phase)
        self.attempt = event.get("attempt", self.attempt)
        if event.get("event") == "active_view_move":
            self.detail = "左顾右盼：移动到安全侧向视点"
        elif event.get("event") == "surface_fusion":
            self.detail = f"局部融合：{event.get('fusion_views', 0)} 次观测"
        elif self.phase in ("close", "lift", "recovery"):
            self.detail = ""
        self.events.append({"wall_s": None if self.started is None else time.monotonic()-self.started,
                            "phase": self.phase, "event": event.get("event"), "attempt": self.attempt})

    @staticmethod
    def _panel(canvas, rgb, xy, size):
        if rgb is not None and getattr(rgb, "size", 0):
            canvas.paste(Image.fromarray(cv2.resize(np.asarray(rgb)[:, :, :3], size)), xy)

    def _fault_note(self):
        notes = []
        if self.coarse_fault_m:
            notes.append(f"粗接近目标移位 {self.coarse_fault_m*100:g} cm")
        if self.fault_m:
            notes.append(f"闭爪前目标移位 {self.fault_m*100:g} cm")
        return " / ".join(notes) or "无故障注入"

    def compose(self, overview, wrist, scene, *, outcome=None):
        canvas = Image.new("RGB", (1280, 720), (18, 24, 33))
        self._panel(canvas, overview, (16, 68), (832, 624))
        self._panel(canvas, wrist, (864, 82), (384, 288))
        self._panel(canvas, scene, (864, 414), (384, 288))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"BusAgent  |  {self.title}  |  第 {self.attempt} 次尝试", font=self.font, fill="white")
        phase = outcome or PHASES.get(self.phase, self.phase)
        draw.text((16, 39), phase + ("  ·  " + self.detail if self.detail and not outcome else ""),
                  font=self.small, fill=(100, 220, 210))
        draw.text((868, 56), "① 腕部 RGB-D：随机械臂移动", font=self.small, fill="white")
        draw.text((868, 388), "② 固定斜上方 RGB-D：辅助观察", font=self.small, fill="white")
        draw.rectangle((16, 630, 848, 692), fill=(18, 24, 33))
        elapsed = 0. if self.started is None else time.monotonic()-self.started
        note = self._fault_note()
        draw.text((26, 635), f"仿真几何代理 · {note} · 录制运行 {elapsed:.1f} s", font=self.small, fill="white")
        draw.text((26, 663), "大画面仅用于展示；右侧为最新 RGB 缓冲，非严格同步帧", font=self.small, fill=(180, 190, 205))
        return np.asarray(canvas)

    def _write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def capture(self, overview, wrist, scene):
        now = time.monotonic()
        if self.started is None:
            self.started = now
        due = int((now-self.started)*self.fps)
        # Retain model/IK wait time as held frames, not accelerated edits.
        while self.last_frame is not None and self.frames < due:
            self._write(self.last_frame)
        self.last_frame = self.compose(overview, wrist, scene)
        if self.frames <= due:
            self._write(self.last_frame)

    def finish(self, overview, wrist, scene, outcome):
        if self.closed:
            return
        self.capture(overview, wrist, scene)
        self.last_frame = self.compose(overview, wrist, scene, outcome=outcome)
        Image.fromarray(self.last_frame).save(self.output.with_suffix(".jpg"))
        for _ in range(self.fps*3):
            self._write(self.last_frame)
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
            raise RuntimeError("Video encoder did not finish")
        finally:
            self.writer_log.close()
        self.output.with_suffix(".video.json").write_text(json.dumps({
            "frames": self.frames, "fps": self.fps, "duration_s": self.frames/self.fps,
            "timing": "wall-clock gaps held; rendering adds overhead; 3 second outcome hold",
            "sensor_panels": "latest buffers, independently timed", "events": self.events,
            "test_coarse_shift_m": self.coarse_fault_m, "test_preclose_shift_m": self.fault_m,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if code:
            raise RuntimeError(f"Video encoder exited {code}; see {self.output.with_suffix('.ffmpeg.log')}")
