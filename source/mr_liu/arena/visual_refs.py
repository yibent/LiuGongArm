"""Frame-grounded references, independent of configured simulation assets."""
import re
import numpy as np

PATTERN = re.compile(r'^obs:([a-f0-9]{32}):(scene_camera|side_camera|wrist_camera):(\d+)$')


def annotate_references(result, frames):
    if result.get('transient'):
        return result
    references = []
    for view in result.get('views', []):
        camera = view['camera']
        height, width = frames[camera+'_rgb'].shape[:2]
        rows = [(x, 'object') for x in view.get('objects', [])]
        rows += [(x, 'region') for x in view.get('regions', [])]
        rows += [(x, 'object') for x in view.get('candidates', [])]
        if view.get('box'):
            rows.append((view, 'object'))
        for index, (row, kind) in enumerate(rows):
            if not row.get('box'): continue
            box = np.asarray(row['box'], dtype=float)
            ref = f"obs:{result['request_id']}:{camera}:{index}"
            row['ref'] = ref
            references.append({'ref': ref, 'label': row.get('label', row.get('description', result['label'])),
                'kind': kind, 'camera': camera, 'box': box.tolist(),
                'box_normalized': (box / [width, height, width, height]).tolist(),
                'frame_sequence': view['sequence'], 'observed_at': result.get('observed_at'),
                'semantic_status': row.get('semantic_status', 'candidate'), 'score': row.get('score')})
    result['references'] = references
    return result


def load_reference(root, ref, scene_id):
    import json
    match = PATTERN.fullmatch(ref)
    if not match: raise ValueError('Invalid visual reference')
    directory = root / match[1]
    request = json.loads((directory/'request.json').read_text())
    if request['scene_id'] != scene_id:
        raise RuntimeError('视觉引用属于另一个场景，请重新观察。')
    result = json.loads((directory/'result.json').read_text())
    row = next((r for r in result.get('references', []) if r['ref'] == ref), None)
    if row is None: raise RuntimeError('视觉引用不存在，请重新观察。')
    return row, directory


def load_snapshot(root, snapshot_ref, scene_id, camera, box_normalized):
    """Ground a user/VLM box against exactly the image it inspected."""
    import json
    if not re.fullmatch(r'[a-f0-9]{32}', snapshot_ref or ''):
        raise ValueError('Invalid image snapshot reference')
    directory = root/snapshot_ref
    request = json.loads((directory/'request.json').read_text())
    if request['scene_id'] != scene_id:
        raise RuntimeError('图像属于另一个场景，请读取当前图片。')
    if camera not in {v['camera'] for v in request['views']}:
        raise ValueError('Image snapshot does not contain this camera')
    box = np.asarray(box_normalized, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any() or (box > 1).any() or np.any(box[2:] <= box[:2]):
        raise ValueError('box_normalized must be [left,top,right,bottom] in [0,1]')
    with np.load(directory/'frames.npz', allow_pickle=False) as frames:
        height, width = frames[camera+'_rgb'].shape[:2]
    return directory, (box * [width,height,width,height]).tolist()
