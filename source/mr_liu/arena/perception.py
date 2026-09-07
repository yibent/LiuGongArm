"""Freeze sensor data on the sim thread; vision runs directly; BusAgent receives only metadata."""
import json
import time
import shutil
from pathlib import Path
from uuid import uuid4
from collections import Counter
import numpy as np
from scipy.ndimage import binary_erosion
import urllib.request
import urllib.error
from mr_liu.arena.arrays import numpy_data, pose_matrix
from mr_liu.grasp.transforms import transform_points
from mr_liu.arena.instances import instance_votes, consistent_witness
from mr_liu.arena.visual_refs import load_reference
from mr_liu.arena.observed_scene import ObservedScene


class PerceptionBridge:
    def __init__(self, root, url):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.url = url
        self.scene_id = uuid4().hex
        self.references = {}
        self.world = ObservedScene()

    def capture(self, runtime, name, *, refine=False, reset=False, cameras=None, vision_mode='auto',
                slow_provider=None, scene_mode='describe', transient=False, visual_ref=None, collection=False, grounding=None, inspect=None):
        request_id = uuid4().hex
        directory = self.root/request_id; directory.mkdir()
        row = getattr(runtime, 'observed_entities', {}).get(name)
        held_target = name and (runtime.held == name or (row and runtime.held == row['name']))
        visual_refs = {}
        if visual_ref:
            selected, origin = load_reference(self.root, visual_ref, self.scene_id)
            requested_cameras = cameras
            visual_refs = {selected['camera']: visual_ref, **self.world.view_references(visual_ref)}
            cameras = [camera for camera in (requested_cameras or [selected['camera']]) if camera in visual_refs]
            if not cameras:
                raise RuntimeError('所选视角尚无该物体的对应观察，请先从此视角定位并关联物体。')
            if inspect:
                previous = json.loads((origin/'result.json').read_text())
                instance = next((item for item in previous.get('collection', {}).get('instances', [])
                                 if visual_ref in item['references']), None)
                refs = list(self.world.view_references(visual_ref).values()) or (instance['references'] if instance else [r['ref'] for r in previous.get('references', [])
                    if r.get('kind') == 'object'] if previous.get('scope') == 'target' else []
                )
                if refs:
                    for ref in refs:
                        ref_row, _ = load_reference(self.root, ref, self.scene_id)
                        visual_refs[ref_row['camera']] = ref
                    if not requested_cameras: cameras = list(visual_refs)
        if grounding:
            cameras = [grounding['camera']]
        cameras = cameras or (['wrist_camera'] if held_target else ['scene_camera', 'side_camera'])
        label = row['label'] if row else name
        arrays, views, instances, instance_labels = {}, [], {}, {}
        for camera in cameras:
            data = runtime.env.scene[camera].data
            arrays[camera+'_rgb'] = numpy_data(data.output['rgb'])[0, :, :, :3].copy()
            if not transient:
                arrays[camera+'_depth'] = numpy_data(data.output['distance_to_image_plane'])[0].squeeze().copy()
                arrays[camera+'_K'] = numpy_data(data.intrinsic_matrices)[0].copy()
                arrays[camera+'_T'] = pose_matrix(numpy_data(data.pos_w)[0], numpy_data(data.quat_w_ros)[0])
                kind = 'instance_id_segmentation_fast'
                if kind in data.output:
                    instances[camera] = numpy_data(data.output[kind])[0].squeeze().copy()
                    instance_labels[camera] = data.info[0][kind]['idToLabels']
            views.append({'camera': camera, 'sequence': runtime.sequence})
        request = {'request_id': request_id, 'scene_id': self.scene_id, 'command_id': runtime.current or 'observe',
            'label': label or 'scene', 'scope': 'collection' if collection else 'target' if name else 'scene',
            'observed_at': time.time(),
            'queries': runtime.config.get('vision', {}).get('vocabulary', []) if not name else [],
            'vision_mode': vision_mode, 'slow_provider': slow_provider, 'scene_mode': scene_mode,
            'transient': transient, 'visual_ref': visual_ref,
            'visual_refs': visual_refs,
            'grounding': grounding,
            'inspect': inspect,
            'views': views, 'refine': refine, 'reset': reset, **runtime.bus_context}
        # Same-machine SSD transport: compression cost exceeded model inference on the fast path.
        np.savez(directory/'frames.npz', **arrays)
        if instances:
            # Evaluation-only sidecar. Never sent to the image model or used
            # to manufacture a target mask/point cloud.
            np.savez(directory/'physical_instances.npz', **instances)
            (directory/'physical_instances.json').write_text(json.dumps(instance_labels))
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
        self.references.update({r['ref']: r for r in result.get('references', [])})
        self.world.update(result)
        while len(self.references) > 256: self.references.pop(next(iter(self.references)))
        return result

    def remember_target(self, result, points, source_ref=None):
        scene = self.scene_cloud(result) if result.get('geometry', {}).get('kind') == 'grid' else None
        identity = self.world.observe_target(result, points, source_ref, scene)
        if identity:
            self.references.update({r['ref']: r for r in result.get('references', [])})
            # Publish references and geometry together, never leave readers half a JSON file.
            directory = self.root / result['request_id']
            temporary = directory / 'result.state.json'
            temporary.write_text(json.dumps({k:v for k,v in result.items() if k != 'physical_witness'}, ensure_ascii=False))
            temporary.replace(directory / 'result.json')
        return identity

    def resolve_reference(self, ref):
        return load_reference(self.root, ref, self.scene_id)[0]

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

    def witness(self, result, bodies):
        directory = self.root/result['request_id']
        views = {}
        labels = json.loads((directory/'physical_instances.json').read_text())
        with np.load(directory/'physical_instances.npz') as instances, np.load(directory/'masks.npz') as masks:
            for camera in masks.files:
                mask = masks[camera].astype(bool)
                core = binary_erosion(mask)
                views[camera] = instance_votes(core if core.any() else mask, instances[camera], labels[camera], bodies)
        return consistent_witness(views), dict(sum(views.values(), Counter()))

    def scene_cloud(self, result):
        """Unmasked depth from the same observation, for placement obstacles."""
        clouds = []
        with np.load(self.root/result['request_id']/'frames.npz', allow_pickle=False) as frames:
            for key in frames.files:
                if not key.endswith('_depth'): continue
                camera = key[:-6]
                depth, K, T = (frames[camera+s] for s in ['_depth', '_K', '_T'])
                valid = np.isfinite(depth) & (depth > .02) & (depth < 3.)
                valid[1::2] = False
                valid[:, 1::2] = False
                rows, cols = np.nonzero(valid)
                z = depth[rows, cols]
                points = np.stack([(cols-K[0,2])*z/K[0,0], (rows-K[1,2])*z/K[1,1], z], axis=1)
                clouds.append(transform_points(T, points))
        return np.concatenate(clouds) if clouds else np.empty((0, 3))
