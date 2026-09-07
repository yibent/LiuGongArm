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
from mr_liu.arena.visual_refs import annotate_references, load_reference, load_snapshot
from mr_liu.arena.observed_scene import collection_geometry, masked_points
from mr_liu.arena.placement_geometry import inspect_grid_views, principal_axis, retained_grid

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5570)
args = parser.parse_args()
store = ROOT/'output/perception'
store.mkdir(parents=True, exist_ok=True)
finder = FlorenceFinder()
yolo = YoloeVisualTracker(os.environ['BUSAGENT_YOLOE_WEIGHTS'])
yolo.load()  # Florence loads lazily only when selected.
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
            if request.get('scope') == 'collection':
                detected, detail = pipeline.collect(frames[camera+'_rgb'], camera=camera,
                    sequence=view['sequence'], scene_id=request['scene_id'], label=request['label'],
                    mode=request.get('vision_mode', 'auto'), slow_provider=request.get('slow_provider'))
                masks.update(detected)
                views.append(detail)
                continue
            if request.get('scope') == 'scene':
                result = pipeline.describe(frames[camera+'_rgb'], camera=camera, sequence=view['sequence'],
                    scene_id=request['scene_id'], queries=request.get('queries', config['vocabulary']),
                    mode=request.get('scene_mode', 'describe'))
                views.append(result)
                continue
            if request.get('grounding'):
                grounding = request['grounding']
                origin, box = load_snapshot(store, grounding['snapshot_ref'], request['scene_id'], camera, grounding['box_normalized'])
                reference = {'label': request['label'], 'box': box, 'ref': 'image:'+grounding['snapshot_ref'], 'semantic_status': 'candidate'}
                with np.load(origin/'frames.npz', allow_pickle=False) as previous:
                    mask, result = pipeline.from_reference(frames[camera+'_rgb'], previous[camera+'_rgb'], reference,
                        camera=camera, sequence=view['sequence'], scene_id=request['scene_id'])
                result['origin'] = 'image_box_sam2'
            elif request.get('visual_ref'):
                reference, origin = load_reference(store, request.get('visual_refs', {}).get(camera, request['visual_ref']), request['scene_id'])
                with np.load(origin/'frames.npz', allow_pickle=False) as previous:
                    mask, result = pipeline.from_reference(frames[camera+'_rgb'], previous[camera+'_rgb'], reference,
                        camera=camera, sequence=view['sequence'], scene_id=request['scene_id'])
            else:
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
        'elapsed_s': time.perf_counter()-started, 'ok': bool(views) if request.get('scope') in {'scene', 'collection'} else bool(masks),
        'perception_source': 'task_routed_image_models', 'transient': request.get('transient', False)}
    result.update(loop='slow' if any(v.get('loop') == 'slow' for v in views) else 'fast',
        fallback_reasons=sorted({v['fallback_reason'] for v in views if v.get('fallback_reason')}),
        semantic_status='detected' if any(v.get('semantic_status') == 'detected' for v in views)
            else 'candidate' if masks else 'scene' if request.get('scope') == 'scene' else 'unknown')
    annotate_references(result, frames)
    if request.get('scope') == 'collection':
        result['ok'] = any(v.get('status') in {'observed', 'not_found'} for v in views)
        collection_geometry(result, frames, masks)
    elif masks and request.get('inspect'):
        points = np.concatenate([masked_points(frames, camera, mask) for camera,mask in masks.items()])
        camera = next(iter(masks))
        height,width = frames[camera+'_rgb'].shape[:2]
        if request['inspect'] == 'axis':
            result['geometry'] = {'kind':'axis','camera':camera, **principal_axis(points,
                {'K':frames[camera+'_K'],'T':frames[camera+'_T'],'size':[width,height]})}
        elif request['inspect'] == 'grid':
            clouds=[]
            low,high=np.quantile(points,[.01,.99],axis=0)
            for view in request['views']:
                key=view['camera']
                cloud=masked_points(frames,key,np.ones(frames[key+'_depth'].shape,bool))
                # Full depth density matters for small visible cell floors;
                # discard distant scene points after calibrated projection.
                cloud=cloud[((cloud[:,:2]>=low[:2]-.02)&(cloud[:,:2]<=high[:2]+.02)).all(1)]
                clouds.append(cloud)
            result['geometry'] = {'kind':'grid', **inspect_grid_views({key:{'points':masked_points(frames,key,mask),
                'T':frames[key+'_T']} for key,mask in masks.items()},np.concatenate(clouds))}
            if request.get('visual_ref'):
                _,origin=load_reference(store,request['visual_ref'],request['scene_id'])
                previous=json.loads((origin/'result.json').read_text()).get('geometry',{})
                if previous.get('status')=='observed' and previous.get('kind')=='grid':
                    with np.load(origin/'frames.npz',allow_pickle=False) as old_frames, np.load(origin/'masks.npz',allow_pickle=False) as old_masks:
                        old_points=np.concatenate([masked_points(old_frames,key,mask) for key,mask in old_masks.items()])
                    for key,mask in masks.items():
                        kept=retained_grid(previous,old_points,masked_points(frames,key,mask),np.concatenate(clouds))
                        if kept is not None:
                            result['geometry']={**kept,'prior_observation_ref':origin.name,'confirmation_camera':key}
                            break
            camera = result['geometry']['camera']
        # Geometry is stored once per observation, not repeated in every ref.
        if request['inspect'] == 'grid' and result.get('geometry', {}).get('status') == 'observed':
            container = next(row for row in result['references'] if row['camera'] == camera)
            index = 1+max(int(row['ref'].rsplit(':',1)[1]) for row in result['references'] if row['camera'] == camera)
            for cell in result['geometry']['cells']:
                ref = f'obs:{request_id}:{camera}:{index}';index+=1
                cell['ref'] = ref
                result['references'].append({'ref':ref,'label':request['label'],'kind':'cell','camera':camera,
                    'container_ref':container['ref'],'row':cell['row'],'column':cell['column'],
                    'observed_at':result['observed_at'],'frame_sequence':container['frame_sequence'],
                    'semantic_status':'observed_grid_cell'})
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
