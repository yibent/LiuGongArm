"""Local image inference; BusAgent receives observation metadata asynchronously."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['YOLO_AUTOINSTALL'] = 'false'
import numpy as np
import torch
from find_and_track.florence_finder import FlorenceFinder
from find_and_track.yoloe_tracker import YoloeVisualTracker
from find_and_track.memory import ObjectMemoryStore
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from mr_liu.perception.arena_vision import ImagePipeline
from mr_liu.perception.sam3_localizer import Sam3Localizer

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5570)
args = parser.parse_args()
store = ROOT/'output/perception'
store.mkdir(parents=True, exist_ok=True)
finder = FlorenceFinder()
yolo = YoloeVisualTracker(os.environ['BUSAGENT_YOLOE_WEIGHTS'])
finder.load(); yolo.load()
config = json.loads((ROOT/'configs/arena_panda.json').read_text())['vision']
# Fail visibly at startup if the independent fast detector cannot load.
yolo.set_text_prompt(config['vocabulary'])
sam = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
    str(ROOT/'_models/sam2/sam2.1_hiera_tiny.pt'), device='cuda'))
memory = ObjectMemoryStore(ROOT/'output/visual-memory')
localizers = {'sam3': Sam3Localizer(ROOT/'_models/sam3.pt', config.get('sam3_conf', .4))}
pipeline = ImagePipeline(finder, yolo, sam, memory_store=memory,
    fast_conf=config.get('fast_conf', .45), refresh_updates=config.get('refresh_updates', 8),
    max_frame_gap=config.get('max_frame_gap', 90), slow_localizer=config.get('slow_localizer', 'sam3'),
    localizers=localizers)


def observe(body):
    request_id = body['request_id']
    if not re.fullmatch('[a-f0-9]{32}', request_id): raise ValueError('Invalid observation reference')
    directory = store/request_id
    request = json.loads((directory/'request.json').read_text())
    frames = np.load(directory/'frames.npz', allow_pickle=False)
    masks, views = {}, []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for view in request['views']:
            camera = view['camera']
            if request.get('scope') == 'scene':
                result = pipeline.describe(frames[camera+'_rgb'], camera=camera, sequence=view['sequence'],
                    scene_id=request['scene_id'], queries=request.get('queries', config['vocabulary']),
                    mode=request.get('scene_mode', 'describe'))
                views.append(result)
                continue
            mask, result = pipeline.observe(frames[camera+'_rgb'], scene_id=request['scene_id'],
                camera=camera, label=request['label'], sequence=view['sequence'],
                refine=request.get('refine', False), reset=request.get('reset', False),
                mode=request.get('vision_mode', 'auto'), slow_provider=request.get('slow_provider'))
            views.append(result)
            if mask is not None: masks[camera] = mask
    np.savez_compressed(directory/'masks.npz', **masks)
    result = {'request_id': request_id, 'command_id': request.get('command_id'),
        'label': request['label'], 'views': views, 'result_ref': request_id,
        'scope': request.get('scope', 'target'), 'observed_at': request.get('observed_at'),
        'elapsed_s': time.perf_counter()-started, 'ok': bool(views) if request.get('scope') == 'scene' else bool(masks),
        'perception_source': 'task_routed_image_models', 'transient': request.get('transient', False)}
    result.update(loop='slow' if any(v.get('loop') == 'slow' for v in views) else 'fast',
        fallback_reasons=sorted({v['fallback_reason'] for v in views if v.get('fallback_reason')}),
        semantic_status='detected' if any(v.get('semantic_status') == 'detected' for v in views)
            else 'candidate' if masks else 'scene' if request.get('scope') == 'scene' else 'unknown')
    (directory/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def respond(self, status, result):
        data = json.dumps(result).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers()
        try: self.wfile.write(data)
        except BrokenPipeError: pass
    def do_GET(self):
        if self.path == '/memories':
            return self.respond(200, {'memories': [{'memory_id': m['memory_id'], 'label': m['label'],
                'views': len(m['views']), 'metadata': m['metadata']} for m in memory.list()]})
        self.respond(200, {'ready': True, 'architecture': 'task_routed_fast_slow',
            'fast': ['yoloe_text', 'yoloe_visual_memory', 'sam2_tiny', 'lk'],
            'slow': {'localization': config.get('slow_localizer', 'sam3'), 'description': 'florence2'},
            'planned_disabled': ['qwen_multimodal'], 'memory_count': len(memory.list())})
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/forget':
                return self.respond(200, {'ok': True, 'deleted': pipeline.forget(body['label'])})
            if self.path != '/observe': return self.respond(404, {'error': 'not_found'})
            self.respond(200, observe(body))
        except Exception as error:
            import traceback; traceback.print_exc()
            self.respond(500, {'ok': False, 'error': str(error)})

print('[Arena vision] models loaded', flush=True)
HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
