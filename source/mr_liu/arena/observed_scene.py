"""Collection geometry from model masks and calibrated depth, with no asset catalog."""
from uuid import uuid4
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
    """Latest object tracks; missing observations are unknown, never proof of absence."""
    def __init__(self):
        self.objects = {}
        self.collections = {}
        self.revision = 0

    def update(self, result):
        collection = result.get('collection')
        if collection is None:
            return
        self.revision += 1
        used = set()
        for row in self.objects.values():
            if collection['label'] in row['labels']:
                row['current'] = False
        for item in collection['instances']:
            candidates = [(np.linalg.norm(np.asarray(item['position_m'])-row['position_m']), key)
                          for key,row in self.objects.items() if key not in used]
            distance, key = min(candidates, default=(float('inf'), None))
            if distance > .025:
                key = 'object:'+uuid4().hex
            previous = self.objects.get(key, {})
            self.objects[key] = {**item, 'track_id': key, 'current': True,
                'labels': sorted(set(previous.get('labels', [])) | {collection['label']}),
                'observed_at': result.get('observed_at'), 'observation_ref': result['request_id']}
            item['track_id'] = key
            used.add(key)
        self.collections[collection['label']] = {**collection, 'observed_at': result.get('observed_at'), 'observation_ref': result['request_id']}

    def snapshot(self):
        return {'revision': self.revision, 'collections': list(self.collections.values()),
                'complete': False, 'source': 'observed_rgbd', 'stale_means': 'reobserve_before_execution'}
