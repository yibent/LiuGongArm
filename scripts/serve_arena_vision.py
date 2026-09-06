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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from mr_liu.perception.arena_vision import ImagePipeline

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5570)
args = parser.parse_args()
store = ROOT/'output/perception'
store.mkdir(parents=True, exist_ok=True)
finder = FlorenceFinder()
yolo = YoloeVisualTracker(os.environ['BUSAGENT_YOLOE_WEIGHTS'])
finder.load(); yolo.load()
sam = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
    str(ROOT/'_models/sam2/sam2.1_hiera_tiny.pt'), device='cuda'))
pipeline = ImagePipeline(finder, yolo, sam)


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
            mask, result = pipeline.observe(frames[camera+'_rgb'], scene_id=request['scene_id'],
                camera=camera, label=request['label'], sequence=view['sequence'],
                refine=request.get('refine', False), reset=request.get('reset', False))
            views.append(result)
            if mask is not None: masks[camera] = mask
    np.savez_compressed(directory/'masks.npz', **masks)
    result = {'request_id': request_id, 'command_id': request.get('command_id'),
        'label': request['label'], 'views': views, 'result_ref': request_id,
        'elapsed_s': time.perf_counter()-started, 'ok': bool(masks),
        'perception_source': 'florence2_yoloe_sam2_lk_rgb'}
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
        self.respond(200, {'ready': True, 'models': ['florence2', 'yoloe', 'sam2_tiny', 'lk']})
    def do_POST(self):
        try:
            if self.path != '/observe': return self.respond(404, {'error': 'not_found'})
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.respond(200, observe(body))
        except Exception as error:
            import traceback; traceback.print_exc()
            self.respond(500, {'ok': False, 'error': str(error)})

print('[Arena vision] models loaded', flush=True)
HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
