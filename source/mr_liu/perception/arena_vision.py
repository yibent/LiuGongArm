"""Reusable Florence / YOLOE / SAM2 / LK image pipeline; no simulator imports."""
from collections import OrderedDict
import time
import cv2
import numpy as np
from find_and_track.types import Detection
from mr_liu.perception.optical_flow import OpticalFlowTracker


def mask_box(mask):
    y, x = np.nonzero(mask)
    return np.array([x.min(), y.min(), x.max()+1, y.max()+1], dtype=np.float32) if len(x) else None


def overlap(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(hi-lo, 0))
    return intersection / max(1., np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)


class ImagePipeline:
    def __init__(self, florence, yoloe, sam):
        self.florence, self.yoloe, self.sam = florence, yoloe, sam
        self.tracks = OrderedDict()

    def observe(self, rgb, *, scene_id, camera, label, sequence, refine=False, reset=False):
        started = time.perf_counter()
        key = (scene_id, camera, label)
        state = self.tracks.pop(key, None)
        if reset: state = None
        stages = []
        mask = None
        if state is not None and sequence > state['sequence']:
            mask = state['flow'].update(rgb)
            stages.append({'model': 'lk', **state['flow'].diagnostic})
        if mask is not None and not refine and state['updates'] < 8:
            state['updates'] += 1
        else:
            box = mask_box(mask) if mask is not None else None
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            if box is None:
                if state is not None:
                    self.yoloe.set_visual_prompt(state['reference'], [Detection(state['box'], label)])
                    found = self.yoloe.detect(bgr)
                    stages.append({'model': 'yoloe', 'operation': 'relocalize', 'candidates': len(found)})
                else:
                    found = []
                if not found:
                    found = self.florence.find(bgr, [label], beams=1)
                    stages.append({'model': 'florence2', 'operation': 'find', 'candidates': len(found)})
                    # Merge duplicate boxes for the same visible instance.
                    unique = []
                    for detection in found:
                        if not any(overlap(detection.xyxy, d.xyxy) > .5 for d in unique): unique.append(detection)
                    found = unique
                    if len(found) == 1:
                        seed = found[0]
                        self.yoloe.set_visual_prompt(bgr, [seed])
                        refined = self.yoloe.detect(bgr)
                        stages.append({'model': 'yoloe', 'operation': 'visual_prompt', 'candidates': len(refined)})
                        matching = [d for d in refined if overlap(d.xyxy, seed.xyxy) > .5]
                        if matching: found = [max(matching, key=lambda d: overlap(d.xyxy, seed.xyxy))]
                if len(found) != 1:
                    return None, {'camera': camera, 'label': label, 'sequence': sequence,
                        'status': 'not_found' if not found else 'ambiguous', 'stages': stages,
                        'elapsed_s': time.perf_counter()-started}
                box = found[0].xyxy
            self.sam.set_image(rgb)
            masks, scores, _ = self.sam.predict(box=box, multimask_output=True)
            index = int(np.argmax(scores))
            mask = np.asarray(masks[index], bool)
            stages.append({'model': 'sam2_tiny', 'operation': 'segment', 'score': float(scores[index])})
            if not mask.any():
                return None, {'camera': camera, 'label': label, 'sequence': sequence,
                    'status': 'empty_mask', 'stages': stages, 'elapsed_s': time.perf_counter()-started}
            flow = OpticalFlowTracker()
            flow.initialize(rgb, mask)
            state = {'flow': flow, 'reference': bgr.copy(), 'box': mask_box(mask), 'updates': 0}
        state['sequence'] = sequence
        self.tracks[key] = state
        while len(self.tracks) > 24: self.tracks.popitem(last=False)
        return mask, {'camera': camera, 'label': label, 'sequence': sequence, 'status': 'observed',
            'box': mask_box(mask).tolist(), 'mask_pixels': int(mask.sum()), 'stages': stages,
            'elapsed_s': time.perf_counter()-started}
