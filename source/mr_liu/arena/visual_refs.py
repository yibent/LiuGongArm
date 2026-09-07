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
