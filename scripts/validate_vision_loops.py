"""Run image-loop regressions against the local service using frozen real camera frames."""
import argparse
import json
from pathlib import Path
import shutil
import time
import urllib.request
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:5570/observe')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = json.loads((args.frames/'request.json').read_text())
    scene_id = 'regression-'+uuid4().hex
    rows = []

    def observe(name, label, sequence, **options):
        request_id = uuid4().hex
        directory = args.frames.parent/request_id
        directory.mkdir()
        shutil.copyfile(args.frames/'frames.npz', directory/'frames.npz')
        request = dict(base, request_id=request_id, scene_id=scene_id, label=label,
                       command_id=name, scope='target', reset=False, refine=False,
                       observed_at=time.time(), views=[dict(v, sequence=sequence) for v in base['views']])
        request.update(options)
        (directory/'request.json').write_text(json.dumps(request))
        query = urllib.request.Request(args.url, data=json.dumps({'request_id': request_id}).encode(),
                                       headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(query, timeout=85) as response:
            result = json.load(response)
        rows.append({'case': name, 'result': result})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps({'case': name, 'ok': result['ok'], 'loop': result.get('loop'),
                          'semantic_status': result.get('semantic_status'), 'elapsed_s': result['elapsed_s'],
                          'views': [{'camera': v['camera'], 'status': v['status'],
                                     'models': [s['model'] for s in v['stages']],
                                     'operations': [s.get('operation') for s in v['stages']],
                                     'objects': [o['label'] for o in v.get('objects', [])]}
                                    for v in result['views']]}), flush=True)
        return result

    def stages(result):
        return [s for view in result['views'] for s in view['stages']]

    red = observe('red_fast', 'red block', 1, vision_mode='fast', reset=True)
    assert red['ok'] and red['loop'] == 'fast'
    tracked = observe('red_flow', 'red block', 2, vision_mode='fast')
    assert tracked['ok'] and all(s['model'] == 'lk' for s in stages(tracked))
    yellow_auto = observe('yellow_auto', 'yellow cylinder', 3, vision_mode='auto', reset=True)
    assert yellow_auto['ok'] and yellow_auto['semantic_status'] == 'detected'
    yellow = observe('yellow_sam3', 'yellow cylinder', 4, vision_mode='slow', slow_provider='sam3')
    assert yellow['ok'] and yellow['semantic_status'] == 'detected'
    assert all(s['model'] == 'sam3' for s in stages(yellow))
    yellow_flow = observe('yellow_flow', 'yellow cylinder', 5, vision_mode='auto')
    assert yellow_flow['ok'] and all(s['model'] == 'lk' for s in stages(yellow_flow))
    recall = observe('yellow_recall', 'yellow cylinder', 500, vision_mode='auto', reset=True)
    assert recall['ok'] and recall['loop'] == 'fast'
    absent = observe('absent_object', 'purple wrench', 501, vision_mode='auto', reset=True)
    assert not absent['ok']
    assert all(s['model'] != 'florence2' for s in stages(absent))
    inventory = observe('scene_fast_inventory', 'scene', 502, scope='scene', scene_mode='inventory',
                        queries=['red block', 'yellow object', 'blue pad'])
    assert inventory['ok'] and inventory['loop'] == 'fast'
    assert {'red block', 'yellow object', 'blue pad'} <= {o['label'] for v in inventory['views'] for o in v['objects']}
    described = observe('scene_description', 'scene', 503, scope='scene', scene_mode='describe',
                        queries=['red block', 'yellow object', 'blue pad'])
    assert described['ok']
    assert all(len([s for s in v['stages'] if s['model'] == 'florence2']) == 1 for v in described['views'])
    print('PASS: fast detection, flow, SAM3 mask handoff, memory recall, absence, inventory, description', flush=True)


if __name__ == '__main__':
    main()
