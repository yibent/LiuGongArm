"""Collection geometry from model masks and calibrated depth, with no asset catalog."""
from uuid import uuid4
from copy import deepcopy
from threading import RLock
import numpy as np
from scipy.ndimage import binary_erosion
from mr_liu.grasp.transforms import transform_points


def masked_points(frames, camera, mask):
    depth, K, T = (frames[camera + suffix] for suffix in ('_depth', '_K', '_T'))
    core = binary_erosion(mask)
    valid = (core if core.any() else mask) & np.isfinite(depth) & (depth > .02) & (depth < 3.)
    y, x = np.nonzero(valid)
    z = depth[y, x]
    if not len(z):
        return np.empty((0, 3))
    points = np.stack([(x-K[0, 2])*z/K[0, 0], (y-K[1, 2])*z/K[1, 1], z], axis=1)
    return transform_points(T, points)


def collection_geometry(result, frames, masks):
    """Associate different camera views once each; retain uncertain detections."""
    references = {row['ref']: row for row in result['references']}
    instances = []
    for view in result['views']:
        used = set()
        for row in view.get('objects', []):
            points = masked_points(frames, view['camera'], masks[row['mask_key']])
            if len(points) < 3:
                references[row['ref']]['geometry_status'] = 'insufficient_depth'
                continue
            bounds = np.quantile(points, [.02, .98], axis=0)
            centre = bounds.mean(0)
            ranked = sorted(((np.linalg.norm(centre-item['_centre']), i) for i, item in enumerate(instances)
                             if i not in used and view['camera'] not in item['cameras']), key=lambda x: x[0])
            # Require nearby centres and overlapping observed extents. One camera
            # cannot merge two adjacent parts through a third-camera proposal.
            match = None
            for distance, index in ranked:
                other = instances[index]
                extent = min(np.linalg.norm(bounds[1]-bounds[0]), np.linalg.norm(other['_bounds'][1]-other['_bounds'][0]))
                if distance <= min(.025, max(.008, extent*.45)) and np.all(bounds[0] <= other['_bounds'][1]+.008) and np.all(other['_bounds'][0] <= bounds[1]+.008):
                    match = index
                    break
            if match is None:
                match = len(instances)
                instances.append({'ref': row['ref'], 'label': row['label'], 'score': row['score'],
                    'semantic_status': row['semantic_status'], 'references': [], 'cameras': [], '_clouds': []})
            used.add(match)
            item = instances[match]
            item['references'].append(row['ref'])
            item['cameras'].append(view['camera'])
            item['_clouds'].append(points)
            joined = np.concatenate(item['_clouds'])
            item['_bounds'] = np.quantile(joined, [.02, .98], axis=0)
            item['_centre'] = item['_bounds'].mean(0)
            if row['score'] > item['score']:
                item.update(ref=row['ref'], score=row['score'])
    for item in instances:
        item['position_m'] = item['_centre'].tolist()
        item['extent_m'] = (item['_bounds'][1]-item['_bounds'][0]).tolist()
        primary = references[item['ref']]
        item.update(camera=primary['camera'], box_normalized=primary['box_normalized'])
        for ref in item['references']:
            references[ref]['position_m'] = item['position_m']
    groups = spatial_groups(instances)
    for i, members in enumerate(groups):
        first = references[instances[members[0]]['ref']]
        camera = first['camera']
        boxes = [references[ref]['box'] for m in members for ref in instances[m]['references'] if references[ref]['camera'] == camera]
        low, high = np.asarray(boxes)[:, :2].min(0), np.asarray(boxes)[:, 2:].max(0)
        index = 1 + max((int(row['ref'].rsplit(':', 1)[1]) for row in result['references'] if row['camera'] == camera), default=-1)
        ref = f"obs:{result['request_id']}:{camera}:{index}"
        height, width = frames[camera+'_rgb'].shape[:2]
        region = {'ref': ref, 'kind': 'region', 'label': result['label']+' region', 'camera': camera,
            'box': np.r_[low, high].tolist(), 'box_normalized': (np.r_[low, high]/[width,height,width,height]).tolist(),
            'frame_sequence': first['frame_sequence'], 'observed_at': result.get('observed_at'),
            'semantic_status': 'spatial_group', 'score': None}
        result['references'].append(region)
        groups[i] = {'ref': ref, 'count': len(members), 'members': [instances[m]['ref'] for m in members],
                     'position_m': np.mean([instances[m]['position_m'] for m in members], axis=0).tolist(),
                     'box_normalized': region['box_normalized'], 'camera': camera}
    clean = [{key:value for key,value in item.items() if not key.startswith('_')} for item in instances]
    result['collection'] = {'label': result['label'], 'count': len(clean), 'instances': clean,
        'groups': groups, 'complete': False, 'count_basis': 'model_masks_multiview_depth_association',
        'grouping': 'spatial_proximity_proposal', 'unlocalized_count': sum(r.get('geometry_status') == 'insufficient_depth' for r in references.values())}
    return result


def spatial_groups(instances):
    """Propose spatial groups; a caller can select a different observed region."""
    if not instances:
        return []
    typical_width = np.median([max(item['extent_m'][:2]) for item in instances])
    reach = max(.02, typical_width*3.)
    remaining, groups = set(range(len(instances))), []
    while remaining:
        members, pending = [], [min(remaining)]
        remaining.remove(pending[0])
        while pending:
            current = pending.pop()
            members.append(current)
            position = np.asarray(instances[current]['position_m'])
            neighbours = [i for i in remaining if np.linalg.norm(position[:2]-np.asarray(instances[i]['position_m'])[:2]) < reach
                          and abs(position[2]-instances[i]['position_m'][2]) < reach]
            remaining.difference_update(neighbours)
            pending.extend(neighbours)
        groups.append(sorted(members))
    return sorted(groups, key=lambda members: (-len(members), min(members)))


class ObservedScene:
    """Observation identities and action effects; never an asset-name resolver."""
    def __init__(self):
        self.objects = {}
        self.collections = {}
        self.references = {}
        self.geometry_memory = {}
        self.revision = 0
        self.lock = RLock()

    def _associate(self, item, used=(), source_ref=None):
        refs = list(dict.fromkeys(item.get('references', []) + [item['ref']]))
        explicit = {self.references[r] for r in refs + [source_ref] if r in self.references}
        candidates = []
        if len(explicit) == 1:
            key = next(iter(explicit))
        else:
            point = np.asarray(item['position_m'])
            for identity, row in self.objects.items():
                if identity in used or row.get('state') in {'held', 'unknown'}:
                    continue
                distance = np.linalg.norm(point - row['position_m'])
                extent = np.maximum(np.asarray(row['extent_m']), .003)
                ratio = np.asarray(item['extent_m']) / extent
                if distance <= .025 and np.all((ratio > .35) & (ratio < 2.8)):
                    candidates.append(identity)
            # Dense neighbours must not be collapsed by a nearest-name heuristic.
            key = candidates[0] if len(candidates) == 1 else 'object:' + uuid4().hex
        previous = self.objects.get(key, {})
        all_refs = list(dict.fromkeys(previous.get('references', []) + refs))
        # Preserve the latest reference from every camera; old refs remain aliases.
        latest = {}
        for ref in all_refs:
            if len(ref.split(':')) == 4: latest[ref.split(':')[2]] = ref
        self.objects[key] = {**previous, **item, 'track_id': key, 'current': True,
            'state': previous.get('state', 'observed'), 'references': list(latest.values()),
            'labels': sorted(set(previous.get('labels', [])) | {item.get('label', 'object')}),
            'association': 'reference' if explicit else 'geometry' if len(candidates) == 1 else 'new',
            'association_candidates': candidates if len(candidates) > 1 else []}
        for ref in refs: self.references[ref] = key
        item['track_id'] = key
        return key

    def update(self, result):
        collection = result.get('collection')
        if collection is None: return
        with self.lock:
            self.revision += 1
            for row in self.objects.values():
                if collection['label'] in row['labels']: row['current'] = False
            used = set()
            for item in collection['instances']:
                item.update(observed_at=result.get('observed_at'), observation_ref=result['request_id'])
                used.add(self._associate(item, used))
            for ref in result.get('references', []):
                if ref['ref'] in self.references: ref['track_id'] = self.references[ref['ref']]
            self.collections[collection['label']] = {**deepcopy(collection),
                'observed_at': result.get('observed_at'), 'observation_ref': result['request_id']}

    def observe_target(self, result, points, source_ref=None, scene=None):
        refs = [r for r in result.get('references', []) if r.get('kind') == 'object']
        if not refs or len(points) < 3: return None
        with self.lock:
            bounds = np.quantile(points, [.02, .98], axis=0)
            item = {**refs[0], 'position_m': bounds.mean(0).tolist(),
                'extent_m': (bounds[1]-bounds[0]).tolist(), 'references': [r['ref'] for r in refs],
                'observed_at': result.get('observed_at'), 'observation_ref': result['request_id']}
            key = self._associate(item, source_ref=source_ref)
            for ref in refs: ref['track_id'] = key
            result['object_id'] = key
            geometry = result.get('geometry', {})
            if geometry.get('kind') == 'grid':
                from mr_liu.arena.placement_geometry import retained_grid
                prior = self.geometry_memory.get(key)
                kept = retained_grid(prior['geometry'], prior['points'], points, scene) if prior and scene is not None else None
                if kept:
                    geometry = {**kept, 'prior_observation_ref': prior['observation_ref']}
                elif geometry.get('status') == 'observed':
                    geometry = {**geometry, 'grid_frame_id': 'grid:' + uuid4().hex}
                if geometry.get('status') == 'observed':
                    primary = refs[0]
                    result['references'] = [r for r in result['references'] if r.get('kind') != 'cell']
                    index = 1 + max(int(r['ref'].rsplit(':', 1)[1]) for r in result['references'] if r['camera'] == primary['camera'])
                    cells = []
                    for cell in geometry['cells']:
                        ref = f"obs:{result['request_id']}:{primary['camera']}:{index}"; index += 1
                        cell_id = f"{geometry['grid_frame_id']}:{cell['row']}:{cell['column']}"
                        cells.append({**cell, 'ref': ref, 'cell_id': cell_id})
                        result['references'].append({'ref': ref, 'cell_id': cell_id, 'kind': 'cell',
                            'label': primary.get('label'), 'camera': primary['camera'],
                            'container_ref': primary['ref'], 'container_id': key,
                            'row': cell['row'], 'column': cell['column'],
                            'grid_frame_id': geometry['grid_frame_id'], 'observed_at': result.get('observed_at')})
                    geometry = {**geometry, 'cells': cells, 'container_id': key,
                        'reference_camera': geometry.get('reference_camera', geometry.get('camera')),
                        'camera': primary['camera']}
                    self.geometry_memory[key] = {'geometry': deepcopy(geometry), 'points': np.asarray(points).copy(),
                        'observation_ref': result['request_id']}
                    self.objects[key]['grid'] = deepcopy(geometry)
                result['geometry'] = geometry
            self.revision += 1
            return key

    def view_references(self, ref):
        with self.lock:
            row = self.objects.get(self.references.get(ref), {})
            return {r.split(':')[2]: r for r in row.get('references', []) if len(r.split(':')) == 4}

    def action_result(self, key, skill, result, destination=None):
        with self.lock:
            row = self.objects.get(key)
            if not row: return
            holding = result.get('holding', {})
            row['last_command_id'] = result.get('command_id')
            if holding.get('verified'):
                row.update(state='held', current=False, placement=None)
            elif skill in {'pick_place', 'place_held'} and result.get('ok'):
                row.update(state='placed' if not result.get('review_required') else 'released_unverified',
                    current=False, placement=deepcopy(destination))
            elif row.get('state') == 'held':
                row.update(state='unknown', current=False)
            self.revision += 1

    def snapshot(self):
        with self.lock:
            # Collections retain membership; current poses/state come from the object table.
            collections = []
            for collection in self.collections.values():
                instances = []
                for old in collection['instances']:
                    current = self.objects.get(old.get('track_id'), old)
                    instances.append({**old, 'current': current.get('current', False),
                        'state': current.get('state', 'observed')})
                collections.append({**collection, 'instances': instances})
            return deepcopy({'revision': self.revision, 'objects': list(self.objects.values()),
                'collections': collections, 'complete': False, 'source': 'observed_rgbd_and_action_feedback',
                'stale_means': 'reobserve_before_execution'})
