"""Task-routed vision: fast tracking/detection, slow localization, persistent references."""
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


def unique_detections(found):
    unique = []
    for detection in sorted(found, key=lambda d: d.score, reverse=True):
        if not any(detection.label == d.label and overlap(detection.xyxy, d.xyxy) > .5 for d in unique):
            unique.append(detection)
    return unique


def credible_florence(found, shape):
    """Reject the full-frame phrase-grounding fallback used for absent labels."""
    height, width = shape[:2]
    output = []
    for detection in unique_detections(found):
        x0, y0, x1, y1 = np.asarray(detection.xyxy, dtype=float)
        area = max(0., x1-x0)*max(0., y1-y0)
        if area < width*height*.92:
            output.append(detection)
    return output


class ImagePipeline:
    def __init__(self, florence, yoloe, sam, *, memory_store=None, fast_conf=.45,
                 refresh_updates=8, max_frame_gap=90, slow_localizer='florence2', localizers=None):
        self.florence, self.yoloe, self.sam = florence, yoloe, sam
        self.memory = memory_store
        self.fast_conf, self.refresh_updates, self.max_frame_gap = fast_conf, refresh_updates, max_frame_gap
        # Task-selected providers, not a fixed chain. Qwen remains planned only.
        self.slow_localizer = slow_localizer
        self.localizers = localizers or {}
        self.tracks = OrderedDict()

    def _text(self, bgr, labels, stages, accept_conf=None):
        started = time.perf_counter()
        try:
            threshold = self.fast_conf if accept_conf is None else float(accept_conf)
            self.yoloe.set_text_prompt(list(labels))
            found = unique_detections(self.yoloe.detect(bgr, conf=min(.1, self.fast_conf)))
            accepted = [d for d in found if d.score >= threshold]
            stages.append({'model': 'yoloe', 'loop': 'fast', 'operation': 'text_detect',
                           'candidates': len(found), 'accepted': len(accepted),
                           'accept_conf': threshold,
                           'accepted_boxes': [{'box': d.xyxy.tolist(), 'label': d.label,
                                               'score': float(d.score)} for d in accepted],
                           'best_score': max((float(d.score) for d in found), default=0.),
                           'elapsed_s': time.perf_counter()-started})
            return accepted, 'low_confidence' if found else 'no_text_detection'
        except Exception as error:
            stages.append({'model': 'yoloe', 'loop': 'fast', 'operation': 'text_detect', 'error': str(error)})
            return [], 'text_detector_unavailable'

    def _recall(self, bgr, label, camera, stages):
        if self.memory is None:
            return None, None
        for memory in self.memory.find(label)[:2]:
            views = sorted(memory.views, key=lambda v: v.view != camera)
            for view in views[:2]:
                reference = cv2.imread(view.image)
                if reference is None:
                    continue
                box = view.crop_bbox or [0, 0, reference.shape[1], reference.shape[0]]
                self.yoloe.set_visual_prompt(reference, [Detection(np.asarray(box), label)], cache_key=view.image)
                candidates = self.yoloe.detect(bgr, conf=min(.1, self.fast_conf))
                found = unique_detections([d for d in candidates if d.score >= self.fast_conf])
                stages.append({'model': 'yoloe', 'loop': 'fast', 'operation': 'memory_recall',
                               'memory_id': memory.memory_id, 'candidates': len(found),
                               'best_score': max((float(d.score) for d in candidates), default=0.)})
                if len(found) == 1:
                    return found[0], memory
                if len(found) > 1:
                    return None, None
        return None, None

    def _remember(self, bgr, detection, camera, semantic_status, origin, scene_id):
        if self.memory is None:
            return None
        existing = next(iter(self.memory.find(detection.label)), None)
        # A provisional result must not overwrite a grounded template.
        if existing and existing.metadata.get('semantic_status') == 'detected' and semantic_status != 'detected':
            return existing.memory_id
        item = self.memory.remember(detection.label, {camera: bgr}, {camera: detection},
            memory_id=existing.memory_id if existing else None, replace_view=True, full_frame=True,
            metadata={'semantic_status': semantic_status, 'origin': origin, 'scene_id': scene_id,
                      'updated_at': time.time(), 'weight_training': False})
        return item.memory_id

    def forget(self, label):
        self.tracks = OrderedDict((key, value) for key, value in self.tracks.items() if key[2] != label)
        memories = self.memory.find(label) if self.memory else []
        return sum(self.memory.delete(memory.memory_id) for memory in memories)

    def _segment(self, rgb, box, stages):
        self.sam.set_image(rgb)
        masks, scores, _ = self.sam.predict(box=box, multimask_output=True)
        index = int(np.argmax(scores))
        stages.append({'model': 'sam2_tiny', 'loop': 'fast', 'operation': 'segment',
                       'mask_score': float(scores[index])})
        return np.asarray(masks[index], bool)

    def describe(self, rgb, *, camera, sequence, scene_id, queries=(), mode='describe'):
        started = time.perf_counter()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        stages, regions = [], []
        # One batched inference, not a Florence call per configured object.
        threshold = min(self.fast_conf, .3) if mode == 'auto' else self.fast_conf
        found, reason = self._text(
            bgr, tuple(dict.fromkeys(queries)), stages, threshold
        ) if queries else ([], 'no_vocabulary')
        objects = []
        for detection in found:
            memory_id = self._remember(bgr, detection, camera, 'detected', 'yoloe_text', scene_id)
            objects.append({'label': detection.label, 'box': detection.xyxy.tolist(),
                            'score': float(detection.score), 'semantic_status': 'detected', 'memory_id': memory_id})
        caption = ''
        informative = [item for item in objects if item['label'].lower() not in {'table'}]
        used_slow = False
        if mode == 'auto' and not informative and 'sam3' in self.localizers:
            located = self.localizers['sam3'].locate(bgr, tuple(dict.fromkeys(queries)))
            for detection in unique_detections([item for item, _ in located]):
                if any(
                    item['label'] == detection.label and
                    overlap(np.asarray(item['box']), detection.xyxy) > .5
                    for item in objects
                ):
                    continue
                memory_id = self._remember(
                    bgr, detection, camera, 'detected', 'sam3', scene_id)
                objects.append({
                    'label': detection.label, 'box': detection.xyxy.tolist(),
                    'score': float(detection.score), 'semantic_status': 'detected',
                    'memory_id': memory_id, 'origin': 'sam3',
                })
            informative = [item for item in objects if item['label'].lower() not in {'table'}]
            used_slow = True
            stages.append({
                'model': 'sam3', 'loop': 'slow', 'operation': 'scene_concept_detection',
                'reason': 'fast_inventory_empty', 'candidates': len(located),
                'accepted': len(informative),
            })
        if mode == 'caption' or (mode == 'auto' and not informative):
            parsed = self.florence.describe(bgr, detail='detailed', beams=1)
            block = parsed.get('<DETAILED_CAPTION>', parsed)
            caption = block if isinstance(block, str) else str(block.get('caption', '')) if isinstance(block, dict) else ''
            stages.append({'model': 'florence2', 'loop': 'slow', 'operation': 'scene_caption',
                           'reason': 'scene_caption_requested'})
        elif mode == 'describe':
            parsed = self.florence.describe(bgr, detail='regions', beams=1)
            block = parsed.get('<DENSE_REGION_CAPTION>', parsed)
            regions = [{'description': label, 'box': box} for label, box in
                       zip(block.get('labels', []), block.get('bboxes', []))]
            stages.append({'model': 'florence2', 'loop': 'slow', 'operation': 'dense_region_caption',
                           'reason': 'scene_description_requested'})
        missing = [label for label in queries if label not in {d.label for d in found}]
        return {'camera': camera, 'sequence': sequence,
                'status': 'described' if regions or caption else 'observed',
                'objects': objects, 'regions': regions, 'caption': caption,
                'exhaustive_inventory': False,
                'unconfirmed_queries': [{'label': label, 'reason': reason} for label in missing],
                'loop': 'slow' if mode in {'describe', 'caption'} or caption or used_slow else 'fast',
                'stages': stages,
                'elapsed_s': time.perf_counter()-started}

    def from_reference(self, rgb, reference_rgb, reference, *, camera, sequence, scene_id):
        """Reacquire in the current image, never execute an old bounding box."""
        stages = []
        label = reference['label']
        box = np.asarray(reference['box'])
        seed = self._segment(reference_rgb, box, stages)
        flow = OpticalFlowTracker()
        flow.initialize(reference_rgb, seed)
        mask = flow.update(rgb)
        stages.append({'model': 'lk', 'operation': 'reference_reacquire', **flow.diagnostic})
        if mask is not None and mask.any():
            mask = self._segment(rgb, mask_box(mask), stages)
        else:
            self.yoloe.set_visual_prompt(np.ascontiguousarray(reference_rgb[:, :, ::-1]),
                [Detection(box, label)], cache_key=reference['ref'])
            found = unique_detections(self.yoloe.detect(np.ascontiguousarray(rgb[:, :, ::-1]), conf=self.fast_conf))
            stages.append({'model': 'yoloe_visual', 'operation': 'reference_reacquire', 'candidates': len(found)})
            if len(found) != 1:
                return None, {'camera': camera, 'sequence': sequence, 'label': label,
                    'status': 'ambiguous' if found else 'reference_lost', 'loop': 'fast',
                    'candidates': [{'label': label, 'box': d.xyxy.tolist(), 'score': float(d.score)} for d in found],
                    'stages': stages, 'semantic_status': 'unknown'}
            mask = self._segment(rgb, found[0].xyxy, stages)
        detail = {'camera': camera, 'sequence': sequence, 'label': label, 'loop': 'fast',
            'status': 'observed' if mask.any() else 'empty_mask', 'semantic_status': reference.get('semantic_status', 'candidate'),
            'origin': 'selected_visual_reference', 'selected_ref': reference['ref'], 'stages': stages,
            'box': mask_box(mask).tolist() if mask.any() else None}
        return (mask if mask.any() else None), detail

    def collect(self, rgb, *, camera, sequence, scene_id, label, mode='auto', slow_provider=None):
        """Discover a set; multiple instances are useful evidence, not ambiguity.

        Keep masks outside the returned metadata. A collection never initializes
        one class-wide flow track or overwrites its memory with an arbitrary item.
        """
        started = time.perf_counter()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        stages, located, reason = [], [], None
        provider, semantic, loop = 'yoloe_text', 'detected', 'fast'
        if mode != 'slow':
            found, reason = self._text(bgr, [label], stages)
            located = [(item, None) for item in found]
        if not located and mode != 'fast':
            loop = 'slow'
            provider = slow_provider or self.slow_localizer
            reason = reason or 'collection_localization_requested'
            if provider in self.localizers:
                located = self.localizers[provider].locate(bgr, [label])
                if not located and provider == 'sam3':
                    fallback = credible_florence(
                        self.florence.find(bgr, [label], beams=1), bgr.shape)
                    located = [(item, None) for item in fallback]
                    if located:
                        provider, semantic = 'florence2', 'candidate'
                    stages.append({'model': 'florence2', 'loop': loop,
                                   'operation': 'find_all', 'reason': 'sam3_not_found',
                                   'candidates': len(located)})
            elif provider == 'florence2':
                located = [(item, None) for item in credible_florence(
                    self.florence.find(bgr, [label], beams=1), bgr.shape)]
                semantic = 'candidate'
            else:
                return {}, {'camera': camera, 'sequence': sequence, 'label': label,
                    'status': 'provider_unavailable', 'objects': [], 'loop': loop, 'stages': stages}
            stages.append({'model': provider, 'loop': loop, 'operation': 'find_all',
                           'reason': reason, 'candidates': len(located)})
        masks, objects = {}, []
        image_set = False
        for detection, mask in located:
            if mask is None:
                if not image_set:
                    self.sam.set_image(rgb)
                    image_set = True
                proposals, scores, _ = self.sam.predict(box=detection.xyxy, multimask_output=True)
                mask = np.asarray(proposals[int(np.argmax(scores))], bool)
            else:
                mask = np.asarray(mask, bool)
            if mask.shape != rgb.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not mask.any():
                continue
            box = mask_box(mask)
            # Repeated overlapping masks from one image are one proposal.
            if any(overlap(box, np.asarray(item['box'])) > .75 for item in objects):
                continue
            key = f'{camera}__{len(objects)}'
            masks[key] = mask
            objects.append({'label': label, 'box': box.tolist(), 'mask_key': key,
                'score': float(detection.score), 'semantic_status': semantic, 'origin': provider})
        if image_set:
            stages.append({'model': 'sam2_tiny', 'loop': 'fast', 'operation': 'segment_collection', 'instances': len(objects)})
        return masks, {'camera': camera, 'sequence': sequence, 'label': label,
            'status': 'observed' if objects else 'not_found', 'objects': objects,
            'semantic_status': semantic if objects else 'unknown', 'loop': loop,
            'fallback_reason': reason if loop == 'slow' else None, 'stages': stages,
            'exhaustive_inventory': False, 'elapsed_s': time.perf_counter()-started}

    def observe(self, rgb, *, scene_id, camera, label, sequence, refine=False, reset=False,
                mode='auto', slow_provider=None):
        started = time.perf_counter()
        key = (scene_id, camera, label)
        state = self.tracks.pop(key, None)
        if reset:
            state = None
        stages, mask, fallback_reason = [], None, None
        proposals = []
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        semantic_status, origin, memory_id, score, score_model = 'unknown', None, None, None, None
        loop = 'fast'

        def result(status):
            return {'camera': camera, 'label': label, 'sequence': sequence, 'status': status,
                    'semantic_status': semantic_status, 'origin': origin, 'score': score, 'score_model': score_model,
                    'memory_id': memory_id, 'loop': loop, 'fallback_reason': fallback_reason,
                    'stages': stages, 'candidates': proposals, 'elapsed_s': time.perf_counter()-started}

        # Repeated or widely separated frames cannot validate a live track.
        if state and mode != 'slow' and 0 < sequence-state['sequence'] <= self.max_frame_gap:
            mask = state['flow'].update(rgb)
            stages.append({'model': 'lk', 'loop': 'fast', **state['flow'].diagnostic})
            if mask is not None:
                semantic_status, origin, memory_id, score, score_model = (state[k] for k in
                    ('semantic_status', 'origin', 'memory_id', 'score', 'score_model'))
                if refine or state['updates'] >= self.refresh_updates:
                    mask = self._segment(rgb, mask_box(mask), stages)
                    state['updates'] = 0
                state['updates'] += 1
            else:
                fallback_reason = 'tracking_lost'
        elif state and mode != 'slow':
            fallback_reason = 'frame_gap_or_reset'

        if mask is None:
            detection = memory = None
            if mode != 'slow':
                detection, memory = self._recall(bgr, label, camera, stages)
                if detection is not None:
                    semantic_status = memory.metadata.get('semantic_status', 'candidate')
                    origin, memory_id = memory.metadata.get('origin', 'visual_reference'), memory.memory_id
                    score_model = 'yoloe_visual'
                else:
                    found, reason = self._text(bgr, [label], stages)
                    if len(found) > 1:
                        proposals = [{'label': label, 'box': d.xyxy.tolist(), 'score': float(d.score)} for d in found]
                        # Multiple fast proposals are model uncertainty, not
                        # proof of multiple matching objects in the scene.
                        reason = 'multiple_fast_candidates'
                        if mode == 'fast':
                            fallback_reason = reason
                            return None, result('ambiguous')
                    elif found:
                        detection, semantic_status, origin = found[0], 'detected', 'yoloe_text'
                        score_model = 'yoloe_text'
                    if detection is None:
                        fallback_reason = reason if fallback_reason is None else fallback_reason+';'+reason
            if detection is None:
                if mode == 'fast':
                    return None, result('not_found')
                loop = 'slow'
                provider = slow_provider or self.slow_localizer
                fallback_reason = fallback_reason or 'slow_localization_requested'
                if provider in self.localizers:
                    located = self.localizers[provider].locate(bgr, [label])
                    found = [d for d, _ in located]
                    if len(located) == 1:
                        detection, mask = located[0]
                        semantic_status, origin = 'detected', provider
                        score_model = provider
                    elif provider == 'sam3':
                        sam3_found = found
                        found = credible_florence(
                            self.florence.find(bgr, [label], beams=1), bgr.shape)
                        stages.append({'model': 'florence2', 'loop': 'slow',
                                       'operation': 'find', 'reason': 'sam3_not_unique',
                                       'candidates': len(found)})
                        if len(found) == 1:
                            detection, semantic_status, origin = found[0], 'candidate', 'florence2'
                            score_model = 'florence2'
                        elif not found:
                            found = sam3_found
                elif provider == 'florence2':
                    found = credible_florence(
                        self.florence.find(bgr, [label], beams=1), bgr.shape)
                    if len(found) == 1:
                        detection, semantic_status, origin = found[0], 'candidate', provider
                else:
                    stages.append({'model': provider, 'loop': 'slow', 'operation': 'unavailable'})
                    return None, result('provider_unavailable')
                stages.append({'model': provider, 'loop': 'slow', 'operation': 'find',
                               'reason': fallback_reason, 'candidates': len(found)})
                if len(found) != 1:
                    proposals = [{'label': label, 'box': d.xyxy.tolist(), 'score': float(d.score)} for d in found]
                    return None, result('not_found' if not found else 'ambiguous')
            score = float(detection.score) if semantic_status == 'detected' else None
            if mask is None:
                mask = self._segment(rgb, detection.xyxy, stages)
            if not mask.any():
                return None, result('empty_mask')
            if memory_id is None:
                memory_id = self._remember(bgr, detection, camera, semantic_status, origin, scene_id)
            state = {'updates': 0, 'semantic_status': semantic_status, 'origin': origin,
                     'memory_id': memory_id, 'score': score, 'score_model': score_model}
        if not mask.any():
            return None, result('empty_mask')
        flow = OpticalFlowTracker()
        flow.initialize(rgb, mask)
        state.update(flow=flow, sequence=sequence)
        self.tracks[key] = state
        while len(self.tracks) > 24:
            self.tracks.popitem(last=False)
        detail = result('observed' if semantic_status == 'detected' else 'candidate')
        detail.update(box=mask_box(mask).tolist(), mask_pixels=int(mask.sum()))
        return mask, detail
