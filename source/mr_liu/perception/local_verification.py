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
    label = body.get('target_label', '').strip()
    if not label: raise ValueError('Missing visual target label')
    region_label = body.get('region_label', '').strip()
    path = Path(root)/ref
    request = json.loads((path/'request.json').read_text())
    with np.load(path/'frames.npz', allow_pickle=False) as frames:
        rgb = frames[camera+'_camera_rgb']
        h, w = rgb.shape[:2]
        supplied = body.get('box_2d')
        region_rows = []
        if supplied is None:
            if not region_label:
                raise ValueError('Provide box_2d or region_label for local verification')
            candidates = finder.find(rgb[:, :, ::-1].copy(), [region_label], beams=1)
            for detection in candidates:
                bounds = np.asarray(detection.xyxy, dtype=float)
                whole = bounds[0] <= 1 and bounds[1] <= 1 and bounds[2] >= w-2 and bounds[3] >= h-2
                region_rows.append({'label': detection.label, 'box': bounds.tolist(),
                                    'localized': not bool(whole)})
            localized = [row for row in region_rows if row['localized']]
            if len(localized) != 1:
                result = {'verdict': 'uncertain', 'predicate': body.get('predicate', 'present'),
                          'objects': [], 'region_objects': region_rows, 'snapshot_ref': ref,
                          'observed_at': request['observed_at'], 'camera': camera,
                          'provider': 'florence2', 'decision_scope': 'visibility_only',
                          'score_calibrated': False, 'reason': 'verification_region_not_unique'}
                (path/'verification.json').write_text(json.dumps(result, ensure_ascii=False))
                return result
            x0, y0, x1, y1 = np.rint(localized[0]['box']).astype(int)
            margin_x, margin_y = max(3, int((x1-x0)*.08)), max(3, int((y1-y0)*.08))
            x0, x1 = max(0, x0-margin_x), min(w, x1+margin_x)
            y0, y1 = max(0, y0-margin_y), min(h, y1+margin_y)
            box = np.array([y0/h*1000, x0/w*1000, y1/h*1000, x1/w*1000])
        else:
            box = np.asarray(supplied, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any() or (box > 1000).any() or np.any(box[2:] <= box[:2]):
                raise ValueError('box_2d must be [ymin,xmin,ymax,xmax] in 0..1000')
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
    localized = [r for r in rows if r['localized_in_roi']]
    if predicate in {'present', 'inside', 'on'}:
        verdict = 'passed' if localized else 'uncertain'
    elif localized:
        # A 2D Florence box alone cannot prove a 3D upright pose. Keep the
        # observation useful without converting aspect ratio into a false pass.
        verdict = 'uncertain'
    else:
        verdict = 'uncertain'
    result = {'verdict': verdict, 'predicate': predicate, 'objects': rows,
              'region_label': region_label or None, 'region_objects': region_rows,
              'snapshot_ref': ref, 'observed_at': request['observed_at'], 'camera': camera,
              'box_2d': box.tolist(), 'crop_size': [int(x1-x0), int(y1-y0)], 'provider': 'florence2',
              'decision_scope': 'visibility_only', 'score_calibrated': False,
              'reason': 'target_visible_in_region' if verdict == 'passed' else
                        'target_not_visible_in_region' if verdict == 'failed' else
                        'local_detection_cannot_establish_requested_condition'}
    (path/'verification.json').write_text(json.dumps(result, ensure_ascii=False))
    return result
