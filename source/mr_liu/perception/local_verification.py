"""A focused local observer. Detector absence and unknown orientation are not success."""
import json
import re
from pathlib import Path
import numpy as np


def verify_region(root, finder, body):
    ref = body.get('snapshot_ref', '')
    if not re.fullmatch(r'[a-f0-9]{32}', ref):
        raise ValueError('Invalid snapshot reference')
    camera = body.get('camera', 'scene')
    if camera not in {'scene', 'side', 'wrist'}:
        raise ValueError('Unknown camera')
    box = np.asarray(body.get('box_2d'), dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any() or (box > 1000).any() or np.any(box[2:] <= box[:2]):
        raise ValueError('box_2d must be [ymin,xmin,ymax,xmax] in 0..1000')
    label = body.get('target_label', '').strip()
    if not label: raise ValueError('Missing visual target label')
    path = Path(root)/ref
    request = json.loads((path/'request.json').read_text())
    with np.load(path/'frames.npz', allow_pickle=False) as frames:
        rgb = frames[camera+'_camera_rgb']
        h, w = rgb.shape[:2]
        y0, x0, y1, x1 = np.rint(box/1000*[h,w,h,w]).astype(int)
        crop = rgb[y0:y1, x0:x1]
        if not crop.size: raise ValueError('Empty image region')
        # Florence accepts BGR. Only the specified ROI enters the model.
        detections = finder.find(crop[:, :, ::-1].copy(), [label], beams=1)
    rows = []
    for detection in detections:
        bounds = np.asarray(detection.xyxy)
        # Florence's phrase-grounding fallback may emit the entire image even
        # when the requested category is absent. Such a box is not a detection.
        whole_roi = bounds[0] <= 1 and bounds[1] <= 1 and bounds[2] >= x1-x0-2 and bounds[3] >= y1-y0-2
        rows.append({'label': detection.label, 'box_crop': bounds.tolist(),
                     'localized_in_roi': not bool(whole_roi)})
    predicate = body.get('predicate', 'present')
    verdict = 'passed' if any(r['localized_in_roi'] for r in rows) and predicate == 'present' else 'uncertain'
    result = {'verdict': verdict, 'predicate': predicate, 'objects': rows,
              'snapshot_ref': ref, 'observed_at': request['observed_at'], 'camera': camera,
              'box_2d': box.tolist(), 'crop_size': [int(x1-x0), int(y1-y0)], 'provider': 'florence2',
              'decision_scope': 'visibility_only', 'score_calibrated': False,
              'reason': 'target_visible_in_region' if verdict == 'passed' else 'local_detection_cannot_establish_requested_condition'}
    (path/'verification.json').write_text(json.dumps(result, ensure_ascii=False))
    return result
