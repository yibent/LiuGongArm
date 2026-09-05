"""HTTP admission + main-thread ownership for a complete grasp transaction.

Only RGB-D grounding waits in a worker. Every Isaac/drive operation, including
cancel/cleanup, runs on the caller's simulation thread through poll().
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

import numpy as np


class GraspInterrupted(RuntimeError):
    pass


class GraspRuntime:
    def __init__(self, control, session_factory, *, clock=time.monotonic, timeout_s=90):
        self.control = control
        self.motion = control.motion
        self.session_factory = session_factory
        self.clock = clock
        self.timeout_s = timeout_s
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grasp-grounding")
        self.job = None
        self.running = False

    def submit(self, params, command_id=None):
        cid = command_id or str(uuid.uuid4())
        target = params.get("target", {})
        if not isinstance(target, dict) or not isinstance(target.get("category"), str) or not target["category"].strip():
            return dict(ok=False, message="请说明要抓取的物体类别和颜色。")
        attributes = target.get("attributes") or {}
        if not isinstance(attributes, dict):
            return dict(ok=False, message="目标属性必须是对象。")
        if target.get("quantity", 1) != 1 or target.get("ordinal") is not None:
            return dict(ok=False, message="当前每次抓取一个物体，请用类别、颜色或最左/最右指定。")
        selector = target.get("spatial_ref")
        if selector not in (None, "leftmost", "rightmost"):
            return dict(ok=False, message="当前抓取只支持最左或最右的空间指代。")
        with self.motion.lock:
            previous = self.motion.result(cid)
            if previous:
                return previous
            if self.motion.external_owner or self.motion.active or self.motion.pending or self.motion.interruption:
                return dict(ok=False, message="BUSY：机械臂正在执行其他动作。")
            self.control.set_follow_enabled(False)
            self.motion.saved = None
            self.motion.external_owner = cid
            record = dict(ok=True, state="accepted", command_id=cid, skill="grasp",
                          message="已接收抓取请求，正在获取新鲜目标位置。", phase="grounding",
                          accepted_at=self.clock())
            self.motion.records[cid] = record
            for old in list(self.motion.records):
                if len(self.motion.records) <= 256:
                    break
                if old != cid:
                    del self.motion.records[old]
            job = dict(cid=cid, target=copy.deepcopy(target), deadline=self.clock()+self.timeout_s,
                       cancelled=threading.Event())
            try:
                job["future"] = self.pool.submit(self.control.grounding.resolve, self.control, {
                    "category": target["category"], "color": attributes.get("color"),
                    "selector": selector, "offset_m": [0, 0, 0],
                    "cancel_epoch": self.control.grounding.cancel_epoch,
                })
            except Exception as exc:
                self.motion.external_owner = None
                self.motion._finish(cid, "failed", f"无法开始目标定位：{exc}")
                return copy.deepcopy(record)
            # Publish only a fully initialized job to the main thread.
            self.job = job
            return copy.deepcopy(record)

    def cancel(self):
        with self.motion.lock:
            if self.job:
                self.job["cancelled"].set()
        self.control.grounding.cancel()

    def guard(self):
        job = self.job
        if not job:
            raise GraspInterrupted("抓取已结束。")
        if job["cancelled"].is_set() or self.motion.interruption:
            raise GraspInterrupted("抓取已中断；未确认抓取成功，请检查当前夹持状态。")
        if self.clock() >= job["deadline"]:
            raise TimeoutError("抓取超时，未确认成功，已请求保持当前位置。")

    def progress(self, phase, message):
        with self.motion.lock:
            if self.job:
                self.motion.records[self.job["cid"]].update(phase=phase, message=message)

    def poll(self):
        """Called outside app.update(), never recursively from physics callbacks."""
        job = self.job
        if not job or self.running:
            return
        session = None
        waiting = False
        state, message, result = "failed", "抓取未完成。", None
        try:
            self.guard()
            if not job["future"].done():
                waiting = True
                return
            grounded = job["future"].result()
            if not grounded.get("ok"):
                raise ValueError(grounded.get("message", "目标定位失败。"))
            if not self.control.grounding.valid(grounded):
                raise ValueError("目标定位已失效，请重新抓取。")
            point = np.asarray(grounded.get("object_position_world_m"), dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("目标没有有效三维位置，未执行抓取。")
            with self.motion.lock:
                self.motion.records[job["cid"]].update(state="started", started_at=self.clock())
            self.running = True
            session = self.session_factory(grounded, job["target"], self.guard, self.progress)
            self.guard()
            result = session.execute(job["cid"])
            self.guard()  # A cancelled request cannot be reported successful.
            state = "completed" if result.get("success") is True else "failed"
            message = ("目标已抓起，夹持与抬升视觉验证通过。" if state == "completed"
                       else f"抓取未完成：{result.get('failure') or 'verification_failed'}；{result.get('message', '')}")
        except GraspInterrupted as exc:
            state, message = "cancelled", str(exc)
        except Exception as exc:
            message = str(exc)
        finally:
            # Waiting for a camera result is not a terminal transition.
            if waiting:
                return
            try:
                if session:
                    session.close()
            except Exception as exc:
                state, message = "failed", f"抓取清理失败，请检查控制器：{exc}"
            with self.motion.lock:
                if state == "completed" and (job["cancelled"].is_set() or self.motion.interruption):
                    state, message = "cancelled", "抓取已中断，请检查当前夹持状态。"
                self.motion.records[job["cid"]]["result"] = result
                # A stop/hold request is consumed by the ordinary motion loop
                # on its next tick; it is not marked done before drive cleanup.
                self.motion.holding = True
                self.motion.hold_target = None  # next tick captures measured pose
                self.motion.external_owner = None
                self.motion.saved = None
                self.motion._finish(job["cid"], state, message)
                self.job = None
                self.running = False

    def close(self):
        self.cancel()
        if not self.running:
            self.poll()
        self.pool.shutdown(wait=False, cancel_futures=True)
