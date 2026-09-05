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
    def __init__(self, control, session_factory, *, clock=time.monotonic, timeout_s=240):
        self.control = control
        self.motion = control.motion
        self.session_factory = session_factory
        self.clock = clock
        self.timeout_s = timeout_s
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grasp-grounding")
        self.job = None
        self.running = False
        self.pending_retry = None
        self.pending_preparation = None

    def capabilities(self):
        return dict(dual_camera_default=True, retry_via_dialogue=True, max_attempts=2,
                    scene_assisted_verification=True, robot_self_mask=True,
                    geometric_fallback=False, preparation_via_dialogue=True,
                    label_coarse_approach=True, optical_flow_tracking=True)

    def status(self):
        with self.motion.lock:
            record = next((r for r in reversed(self.motion.records.values()) if r.get("skill") == "grasp"), None)
            result = copy.deepcopy(record)
            if result is not None:
                result["retry_available"] = self._retry_valid()
                result["preparation_available"] = self._preparation_valid()
            return result

    def _preparation_valid(self):
        p = self.pending_preparation
        if not p or self.clock() >= p["expires"]:
            return False
        latest = next((r for r in reversed(self.motion.records.values())
                       if r.get("skill") not in {"status", "capabilities", "set_speed"}), None)
        return latest is not None and latest.get("command_id") == p["cid"]

    def submit_preparation(self, params, cid):
        with self.motion.lock:
            previous = self.motion.result(cid)
            if previous:
                return previous
            if self.motion.external_owner or self.motion.active or self.motion.pending or self.motion.interruption:
                return dict(ok=False, message="BUSY：机械臂正在执行其他动作。")
            if not self._preparation_valid() or params.get("proposal_id") != self.pending_preparation["id"]:
                return dict(ok=False, message="初始准备建议已失效或未获对应确认；未执行移动，请重新评估抓取。")
            p, self.pending_preparation = self.pending_preparation, None
            self.control.set_follow_enabled(False)
            self.motion.saved = None
            self.motion.external_owner = cid
            self.motion.records[cid] = dict(ok=True, state="accepted", command_id=cid, skill="grasp",
                message="已确认初始准备移动，尚未抓取。", phase="preparation", accepted_at=self.clock(),
                progress_seq=0, grasp=dict(attempt=0, max_attempts=2, dual_camera=True, preparation_only=True))
            self.job = dict(cid=cid, target=p["target"], deadline=self.clock()+self.timeout_s,
                            cancelled=threading.Event(), preparation_session=p["session"])
            return copy.deepcopy(self.motion.records[cid])

    def _retry_valid(self):
        pending = self.pending_retry
        if not pending or self.clock() >= pending["expires"]:
            return False
        latest = next((r for r in reversed(self.motion.records.values())
                       if r.get("skill") not in {"status", "capabilities", "set_speed"}), None)
        return latest is not None and latest.get("command_id") == pending["cid"]

    def submit(self, params, command_id=None):
        cid = command_id or str(uuid.uuid4())
        if params.get("prepare_last") is True:
            return self.submit_preparation(params, cid)
        if "recovery_mode" in params or "perception" in params:
            return dict(ok=False, message="双相机默认启用；不接受模式选择，重试请通过对话下达。")
        if params.get("retry_last") is True:
            return self.submit_retry(cid)
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
            if self._retry_valid():
                return dict(ok=False, message="上次抓取尚待处理；要恢复请说再试一次，取消请说停止。")
            self.control.set_follow_enabled(False)
            self.motion.saved = None
            self.motion.external_owner = cid
            record = dict(ok=True, state="accepted", command_id=cid, skill="grasp",
                          message="已接收抓取请求，正在获取新鲜目标位置。", phase="grounding",
                          accepted_at=self.clock(), progress_seq=0,
                          grasp=dict(attempt=0, max_attempts=2, dual_camera=True))
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
                    "memory_id": target.get("memory_id"),
                    "cancel_epoch": self.control.grounding.cancel_epoch,
                })
            except Exception as exc:
                self.motion.external_owner = None
                self.motion._finish(cid, "failed", f"无法开始目标定位：{exc}")
                return copy.deepcopy(record)
            # Publish only a fully initialized job to the main thread.
            self.job = job
            return copy.deepcopy(record)

    def submit_retry(self, cid):
        with self.motion.lock:
            previous = self.motion.result(cid)
            if previous:
                return previous
            if self.motion.external_owner or self.motion.active or self.motion.pending or self.motion.interruption:
                return dict(ok=False, message="BUSY：机械臂正在执行其他动作。")
            if not self._retry_valid():
                return dict(ok=False, message="没有可恢复的抓取，或上下文已失效；未执行重试。")
            pending = self.pending_retry
            self.pending_retry = None
            self.control.set_follow_enabled(False)
            self.motion.saved = None
            self.motion.external_owner = cid
            record = dict(ok=True, state="accepted", command_id=cid, skill="grasp", phase="recovery",
                          message="已接收重试请求，正在重新确认目标与夹持状态。", accepted_at=self.clock(),
                          progress_seq=0, grasp=dict(attempt=1, max_attempts=2, dual_camera=True,
                                                    retry_of=pending["cid"]))
            self.motion.records[cid] = record
            self.job = dict(cid=cid, target=pending["target"], deadline=self.clock()+self.timeout_s,
                            cancelled=threading.Event(), retry_session=pending["session"])
            return copy.deepcopy(record)

    def cancel(self):
        with self.motion.lock:
            if self.job:
                self.job["cancelled"].set()
            if self.pending_retry:
                self.pending_retry["expires"] = -float("inf")
            if self.pending_preparation:
                self.pending_preparation["expires"] = -float("inf")
        self.control.grounding.cancel()

    def guard(self):
        job = self.job
        if not job:
            raise GraspInterrupted("抓取已结束。")
        if job["cancelled"].is_set() or self.motion.interruption:
            raise GraspInterrupted("抓取已中断；未确认抓取成功，请检查当前夹持状态。")
        if self.clock() >= job["deadline"]:
            raise TimeoutError("抓取超时，未确认成功，已请求保持当前位置。")

    def progress(self, phase, message, **details):
        with self.motion.lock:
            if self.job:
                record = self.motion.records[self.job["cid"]]
                record.update(phase=phase, message=message, progress_seq=record["progress_seq"] + 1)
                record["grasp"].update(details)

    def poll(self):
        """Called outside app.update(), never recursively from physics callbacks."""
        if self.pending_retry and not self._retry_valid():
            pending, self.pending_retry = self.pending_retry, None
            pending["session"].close(stop_motion=False)
        if self.pending_preparation and not self._preparation_valid():
            pending, self.pending_preparation = self.pending_preparation, None
            pending["session"].close(stop_motion=False)
        job = self.job
        if not job or self.running:
            return
        session = job.get("retry_session") or job.get("preparation_session")
        waiting = False
        state, message, result = "failed", "抓取未完成。", None
        try:
            self.guard()
            if session is None and not job["future"].done():
                waiting = True
                return
            if session is None:
                grounded = job["future"].result()
                if not grounded.get("ok"):
                    raise ValueError(grounded.get("message", "目标定位失败。"))
                if not self.control.grounding.valid(grounded):
                    raise ValueError("目标定位已失效，请重新抓取。")
                point = np.asarray(grounded.get("object_position_world_m"), dtype=float)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError("目标没有有效三维位置，未执行抓取。")
                session = self.session_factory(grounded, job["target"], self.guard, self.progress)
            with self.motion.lock:
                self.motion.records[job["cid"]].update(state="started", started_at=self.clock())
            self.running = True
            self.guard()
            result = (session.prepare(job["cid"]) if "preparation_session" in job else
                      session.retry(job["cid"]) if "retry_session" in job else session.execute(job["cid"]))
            self.guard()  # A cancelled request cannot be reported successful.
            state = "completed" if result.get("success") is True else "failed"
            message = ("目标已抓起，夹持与抬升视觉验证通过。" if state == "completed"
                       else f"抓取未完成：{result.get('failure') or 'verification_failed'}；{result.get('message', '')}")
            if result.get("preparation_only") is True:
                message = result["message"]
            if result.get("metrics", {}).get("recovery_stop_reason") == "payload_uncertain_do_not_open":
                message = "夹持状态不确定，已停止恢复并保持夹爪；未确认抓取成功，请检查当前持物状态。"
            elif result.get("retry_available") is True:
                message += "。已保持当前位置，等待对话确认是否重新观察再试一次；尚未重试。"
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
                    if state == "failed" and result and result.get("preparation") and not job["cancelled"].is_set() and not self.motion.interruption:
                        session.hold_for_retry()
                        self.pending_preparation = dict(session=session, cid=job["cid"], target=job["target"],
                                                        id=result["preparation"]["id"], expires=self.clock()+120)
                    elif state == "failed" and result and result.get("retry_available") is True and not job["cancelled"].is_set() and not self.motion.interruption:
                        session.hold_for_retry()
                        self.pending_retry = dict(session=session, cid=job["cid"], target=job["target"], expires=self.clock()+120)
                    else:
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
