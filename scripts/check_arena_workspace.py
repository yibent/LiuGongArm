"""Exercise real workspace controls, then restore the initial scene and robot.

This performs physical simulation writes. Run while the simulation is idle.
"""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:7861')
parser.add_argument('--output', type=Path, default=Path('output/ui-control-validation'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
checks = []

def request(path, payload=None):
    r = urllib.request.Request(args.url + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=15) as response: return json.load(response)
    except urllib.error.HTTPError as e: return json.load(e)

def wait(command_id):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        result = request('/api/commands/' + command_id)
        if result.get('state') not in {'accepted', 'running'}: return result
        time.sleep(.12)
    raise AssertionError('Command timeout: ' + command_id)

def command(**params):
    identifier = 'ui-validation-' + uuid.uuid4().hex[:12]
    result = request('/api/command', {'command_id': identifier, 'skill': 'workspace', 'params': params})
    if result.get('state') in {'accepted', 'running'}: result = wait(identifier)
    assert result.get('ok'), result
    return result

def workspace(): return request('/api/workspace')
def obj(state, id): return next(row for row in state['objects'] if row['id'] == id)
def distance(a, b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def record(name, **evidence):
    checks.append({'check': name, **evidence})
    print('PASS:', name, flush=True)

def stream(target, linear, angular=None, duration=.8, release=True, id=None):
    identifier = 'ui-stream-' + uuid.uuid4().hex[:12]
    velocity = {'linear': linear, 'angular': angular or [0,0,0]}
    params = {'action': 'teleop', 'target': target, **velocity}
    if id: params['id'] = id
    response = request('/api/command', {'command_id': identifier, 'skill': 'workspace', 'params': params})
    assert response.get('ok'), response
    until = time.monotonic() + duration
    while time.monotonic() < until:
        result = request('/api/teleop', {'command_id': identifier, **velocity})
        assert result.get('ok'), result
        time.sleep(.1)
    if release: assert request('/api/teleop', {'command_id': identifier, 'stop': True}).get('ok')
    result = wait(identifier)
    assert result.get('ok'), result
    return result

assert request('/api/status').get('command_id') is None, 'Run this check when no command is active'
try:
    command(action='reset', scope='all')
    baseline = workspace()
    assert baseline.get('api_version') == 2 and baseline['controls']['teleop']
    (args.output/'baseline.json').write_text(json.dumps(baseline, indent=2))
    target = next(o['id'] for o in baseline['objects'] if o['id'] != 'floor')
    row = obj(baseline, target)
    position = row['position'].copy(); position[0] += .025
    rotation = row['rotation'].copy(); rotation[2] += 10
    command(action='object', id=target, position=position, rotation=rotation)
    actual = obj(workspace(), target)
    assert abs(actual['position'][0] - position[0]) < .004, actual
    assert abs(actual['rotation'][2] - rotation[2]) < 2, actual
    record('object absolute translation and rotation', id=target, requested=position, measured=actual['position'])
    command(action='jog', target='object', id=target, translation=[.01,0,0])
    actual = obj(workspace(), target)
    assert abs(actual['position'][0] - position[0] - .01) < .004, actual
    record('object relative jog', measured=actual['position'])
    before = actual['position']
    stream('object', [0,.025,0], duration=.7, id=target)
    after = obj(workspace(), target)['position']
    assert after[1]-before[1] > .008, (before, after)
    record('object joystick', moved_m=distance(before,after))
    command(action='reset', scope='objects')
    reset = workspace()
    assert distance(obj(reset,target)['position'],row['position']) < .004
    assert distance(reset['robot']['position'],baseline['robot']['position']) < .004
    record('object-only reset preserves robot pose')
    before = reset['robot']['position']
    command(action='jog', target='robot', translation=[0,0,.015])
    after = workspace()['robot']['position']
    assert after[2] - before[2] > .006, (before,after)
    record('Arena IK robot jog', before=before, after=after)
    robot = workspace()['robot']
    position = robot['position'].copy(); position[0] += .015
    command(action='robot', position=position, rotation=robot['rotation'])
    actual = workspace()['robot']['position']
    assert distance(position,actual) < .008, (position,actual)
    record('absolute TCP pose through Arena IK', target=position, measured=actual)
    before = actual
    stream('robot', [0,.02,0], duration=.9)
    after = workspace()['robot']['position']
    assert after[1]-before[1] > .008, (before,after)
    time.sleep(.8)
    stationary = workspace()['robot']['position']
    assert distance(stationary,after) < .004, (stationary,after)
    record('robot joystick release stops motion', moved_m=distance(before,after), drift_after_release_m=distance(stationary,after))
    before = stationary
    stream('robot', [0,0,.015], duration=.4, release=False)
    stopped = workspace()['robot']['position']; time.sleep(.8)
    assert distance(workspace()['robot']['position'],stopped) < .004
    record('lost-heartbeat lease stops robot', moved_m=distance(before,stopped))
    command(action='gripper',state='close'); assert workspace()['robot']['gripper'] == -1
    command(action='gripper',state='open'); assert workspace()['robot']['gripper'] == 1
    record('gripper close and open')
    command(action='controller',position_tolerance_m=.004)
    assert workspace()['controller']['position_tolerance_m'] == .004
    command(action='controller',**baseline['controller'])
    record('controller settings apply and restore')
    command(action='object',id=target,position=[row['position'][0]+.02,*row['position'][1:]],rotation=row['rotation'])
    before = obj(workspace(),target)['position']
    command(action='reset',scope='robot')
    after = workspace()
    assert distance(obj(after,target)['position'],before) < .004
    assert distance(after['robot']['position'],baseline['robot']['position']) < .008
    record('robot-only reset preserves object position')
    for scene in ('sorting','precision','initial'):
        command(action='scene',scene_id=scene)
        assert workspace()['scene_id'] == scene
        record('scene preset '+scene)
finally:
    stop = request('/api/command', {'command_id': uuid.uuid4().hex, 'skill': 'stop'})
    time.sleep(.8)
    command(action='scene',scene_id='initial')
    final = workspace()
    (args.output/'final.json').write_text(json.dumps(final,indent=2))
    (args.output/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
for view in ('scene','side','wrist'):
    with urllib.request.urlopen(args.url+'/api/frame/'+view+'.jpg') as response: (args.output/(view+'.jpg')).write_bytes(response.read())
print(json.dumps({'passed':True,'checks':len(checks),'objects':len(final['objects']),'restored':'initial'},ensure_ascii=False))
