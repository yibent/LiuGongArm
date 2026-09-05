"""Fresh RGB-D detections -> object-relative world targets (no stage truth)."""
from __future__ import annotations
import copy
import threading
import time
import numpy as np


def confirmed_mask(bgr, semantic, xyxy, prompt):
    """Simulation provider: confirm class with sensor labels, color with RGB.

    Explicitly uses Isaac metadata for identity, not a claim of visual class
    recognition. No configured spawn positions are used to create goals.
    """
    import cv2
    if not isinstance(semantic, dict) or 'data' not in semantic:
        return None, '相机语义标签尚未就绪，无法核对模型候选类别。'
    ids = np.asarray(semantic['data']).squeeze()
    if ids.shape != bgr.shape[:2]:
        return None, '相机语义与RGB帧不匹配。'
    colors = {'red', 'yellow', 'green', 'blue', 'purple', 'black', 'white', 'orange'}
    words = prompt.lower().split()
    color = words[0] if words and words[0] in colors else None
    category = ' '.join(words[1:] if color else words).replace('_', ' ')
    labels = semantic.get('info', {}).get('idToLabels', {})
    matching = [int(key) for key, value in labels.items()
                if isinstance(value, dict) and str(value.get('class', '')).replace('_', ' ').lower() == category]
    mask = np.isin(ids, matching)
    if color:
        hue, saturation, value = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
        if color == 'white':
            color_pixels = (saturation < 55) & (value > 140)
        elif color == 'black':
            color_pixels = value < 65
        else:
            ranges = {'red': (0, 10), 'orange': (10, 22), 'yellow': (22, 38),
                      'green': (38, 90), 'blue': (90, 130), 'purple': (130, 170)}
            lo, hi = ranges[color]
            color_pixels = ((hue >= lo) & (hue <= hi)) | ((hue >= 170) if color == 'red' else False)
            color_pixels &= (saturation > 65) & (value > 40)
        mask &= color_pixels
    x1,y1,x2,y2 = np.asarray(xyxy,dtype=int)
    roi = np.zeros_like(mask)
    roi[max(0,y1):min(mask.shape[0],y2), max(0,x1):min(mask.shape[1],x2)] = True
    mask &= roi
    if mask.sum() < 4:
        return None, f'模型候选未通过 {prompt} 的仿真相机类别及颜色核对，不能确认目标。'
    return mask, None


def surface_point(xyxy, depth, unproject, mask=None):
    """Median of valid central depth samples, not the old table-plane heuristic."""
    if depth is None:
        return None
    h, w = depth.shape
    x1, y1, x2, y2 = np.asarray(xyxy, dtype=float)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    rx, ry = max(2, (x2-x1)*.12), max(2, (y2-y1)*.12)
    xs = np.arange(max(0, int(np.ceil(cx-rx))), min(w, int(np.floor(cx+rx))+1))
    ys = np.arange(max(0, int(np.ceil(cy-ry))), min(h, int(np.floor(cy+ry))+1))
    u, v = np.meshgrid(xs, ys)
    if mask is not None:
        v, u = np.nonzero(mask)
    z = depth[v, u]
    valid = np.isfinite(z) & (z > .01) & (z < 5)
    if valid.sum() < 4:
        return None
    z_med = np.median(z[valid])
    valid &= abs(z-z_med) < .015
    if valid.sum() < 4:
        return None
    points = np.asarray(unproject(np.c_[u[valid], v[valid]], z[valid]), dtype=float)
    if not np.isfinite(points).all():
        return None
    return np.median(points, axis=0).tolist()


class SceneGrounding:
    def __init__(self):
        self.condition = threading.Condition()
        self.query_lock = threading.Lock()
        self.observation = None
        self.cancel_epoch = 0
        self.last_result = None

    def cancel(self):
        with self.condition:
            self.cancel_epoch += 1
            self.condition.notify_all()

    def publish(self, prompt_version, prompt, detections, timestamp, rejections=None):
        with self.condition:
            self.observation = dict(prompt_version=prompt_version, prompt=prompt,
                                    detections=copy.deepcopy(detections), observed_at=timestamp,
                                    rejections=rejections or [])
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return copy.deepcopy(self.last_result)

    def valid(self, token):
        with self.condition:
            return (isinstance(token, dict) and token.get('cancel_epoch') == self.cancel_epoch
                    and 0 <= time.monotonic()-float(token.get('observed_at', 0)) < 3)

    def resolve(self, control, request, timeout=8):
        # The visual pipeline has a single prompt; don't mix concurrent queries.
        if not self.query_lock.acquire(blocking=False):
            return dict(ok=False, message='感知正在处理另一目标，请稍后重试。')
        try:
            category = str(request.get('category') or '').strip()
            color = str(request.get('color') or '').strip()
            if not category:
                return dict(ok=False, message='请说明目标物体，无法仅凭它或那里确定目标。')
            prompt = f'{color} {category}'.strip().replace('_', ' ')
            with self.condition:
                epoch = self.cancel_epoch
            cfg = control.set_prompt(prompt)
            deadline = time.monotonic()+timeout
            with self.condition:
                while True:
                    if epoch != self.cancel_epoch:
                        return dict(ok=False, message='物体定位请求已被中断，未执行移动。')
                    obs = self.observation
                    if (obs and obs['prompt_version'] == cfg.prompt_version
                            and time.monotonic()-obs['observed_at'] < 3
                            and obs['detections']):
                        break
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        reason = obs['rejections'][0] if obs and obs['prompt_version'] == cfg.prompt_version and obs.get('rejections') else f'当前相机未获得 {prompt} 的新鲜检测，不能确认存在或执行定位移动。'
                        return dict(ok=False, message=reason)
                    self.condition.wait(min(.2, remaining))
                obs = copy.deepcopy(obs)
            candidates = obs['detections']
            selected = None
            if request.get('selector') in {'leftmost', 'rightmost'}:
                # Explicit image-relative selector, never silently choose nearest.
                selected = sorted(candidates, key=lambda x: x['xyxy'][0])[0 if request['selector'] == 'leftmost' else -1]
            elif len(candidates) == 1:
                selected = candidates[0]
            result = dict(ok=True, prompt=prompt, count=len(candidates), source='scene_camera_rgbd+isaac_semantics',
                          observed_at=obs['observed_at'], cancel_epoch=epoch,
                          message=f'顶部相机检测并经Isaac语义标签与颜色核对，当前有 {len(candidates)} 个 {prompt} 候选。')
            if selected:
                result['object_position_world_m'] = selected.get('world_position_m')
                result['confidence'] = selected.get('score')
            if request.get('offset_m') is not None:
                if selected is None:
                    return dict(ok=False, message=f'检测到 {len(candidates)} 个 {prompt} 候选，请说明最左边或最右边的那个。')
                point = selected.get('world_position_m')
                if point is None:
                    return dict(ok=False, message='检测到了目标，但有效深度不足；不能生成三维移动位置。')
                offset = np.asarray(request['offset_m'], dtype=float)
                if offset.shape != (3,) or not np.isfinite(offset).all():
                    return dict(ok=False, message='物体相对偏移需要三个有限米制数值。')
                result['target_world_m'] = (np.asarray(point)+offset).tolist()
                result['offset_m'] = offset.tolist()
                result['message'] += '已通过标定和深度计算目标位置，尚未执行移动。'
            with self.condition:
                self.last_result = copy.deepcopy(result)
            return result
        finally:
            self.query_lock.release()
