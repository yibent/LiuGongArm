"""Freeze sensor data on the sim thread; vision runs directly; BusAgent receives only metadata."""
import json
import time
import shutil
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.ndimage import binary_erosion
import urllib.request
import urllib.error
from mr_liu.arena.arrays import numpy_data, pose_matrix
from mr_liu.grasp.transforms import transform_points


class PerceptionBridge:
    def __init__(self, root, url):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.url = url
        self.scene_id = uuid4().hex

    def capture(self, runtime, name, *, refine=False, reset=False, cameras=None, vision_mode='auto',
                slow_provider=None, scene_mode='describe', transient=False):
        request_id = uuid4().hex
        directory = self.root/request_id; directory.mkdir()
        row = next((r for r in runtime.config['objects'] + runtime.config['destinations']
                    if name in [r['name'], r['label'], *r.get('aliases', [])]), None)
        held_target = name and (runtime.held == name or (row and runtime.held == row['name']))
        cameras = cameras or (['wrist_camera'] if held_target else ['scene_camera', 'side_camera'])
        label = row['label'] if row else name
        arrays, views = {}, []
        for camera in cameras:
            data = runtime.env.scene[camera].data
            arrays[camera+'_rgb'] = numpy_data(data.output['rgb'])[0, :, :, :3].copy()
            if not transient:
                arrays[camera+'_depth'] = numpy_data(data.output['distance_to_image_plane'])[0].squeeze().copy()
                arrays[camera+'_K'] = numpy_data(data.intrinsic_matrices)[0].copy()
                arrays[camera+'_T'] = pose_matrix(numpy_data(data.pos_w)[0], numpy_data(data.quat_w_ros)[0])
            views.append({'camera': camera, 'sequence': runtime.sequence})
        request = {'request_id': request_id, 'scene_id': self.scene_id, 'command_id': runtime.current or 'observe',
            'label': label or 'scene', 'scope': 'target' if name else 'scene',
            'observed_at': time.time(),
            'queries': runtime.config.get('vision', {}).get('vocabulary', []) if not name else [],
            'vision_mode': vision_mode, 'slow_provider': slow_provider, 'scene_mode': scene_mode,
            'transient': transient,
            'views': views, 'refine': refine, 'reset': reset, **runtime.bus_context}
        # Same-machine SSD transport: compression cost exceeded model inference on the fast path.
        np.savez(directory/'frames.npz', **arrays)
        (directory/'request.json').write_text(json.dumps(request))
        return request

    def request(self, request):
        # Local vision only. Image bytes never go through BusAgent or an LLM.
        body = json.dumps({'request_id': request['request_id']}).encode()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        query = urllib.request.Request(self.url, data=body, headers={'Content-Type': 'application/json'})
        try:
            with opener.open(query, timeout=85) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            result = {'request_id': request['request_id'], 'command_id': request['command_id'],
                      'scope': request['scope'], 'observed_at': request['observed_at'],
                      'label': request['label'], 'ok': False, 'views': [], 'error': str(error)}
        finally:
            if request.get('transient'):
                shutil.rmtree(self.root/request['request_id'], ignore_errors=True)
        if result.get('request_id') != request['request_id']:
            raise RuntimeError('Vision returned a different observation reference')
        return result

    def cloud(self, result):
        directory = self.root/result['request_id']
        frames = np.load(directory/'frames.npz', allow_pickle=False)
        masks = np.load(directory/'masks.npz', allow_pickle=False)
        clouds, diagnostics = [], {}
        for camera in masks.files:
            depth, K, T = (frames[camera+s] for s in ['_depth','_K','_T'])
            core = binary_erosion(masks[camera])
            mask = core if core.any() else masks[camera]
            valid = mask & np.isfinite(depth) & (depth > .02) & (depth < 3.)
            rows, cols = np.nonzero(valid)
            z = depth[rows, cols]
            points = np.stack([(cols-K[0,2])*z/K[0,0], (rows-K[1,2])*z/K[1,1], z], axis=1)
            points = transform_points(T, points)
            clouds.append(points)
            diagnostics[camera] = {'points': len(points), 'camera_world': T.tolist(), 'intrinsics': K.tolist(),
                'bounds_world': [points.min(0).tolist(), points.max(0).tolist()] if len(points) else None}
        points = np.concatenate(clouds) if clouds else np.empty((0,3))
        if not len(points): raise RuntimeError('No observed target depth')
        _, indices = np.unique(np.round(points/.0005).astype(int), axis=0, return_index=True)
        points = points[np.sort(indices)]
        return points[np.linspace(0,len(points)-1,min(len(points),16000),dtype=int)], diagnostics
