"""Live simulation regression: visual pickup, failed destination, stop, continued placement."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7861')
    parser.add_argument('--target', default='red cube')
    parser.add_argument('--destination', default='blue pad')
    parser.add_argument('--interrupt-placement', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('output/validation/holding-live.json'))
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, body=None):
        req = urllib.request.Request(args.url + path, data=json.dumps(body).encode() if body else None,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=10) as response: return json.load(response)

    report = {'target': args.target, 'destination': args.destination,
              'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'commands': []}

    def command(skill, params=None, interrupt=False):
        key = 'holding-test-' + uuid4().hex
        call('/api/command', {'command_id': key, 'skill': skill, 'params': params or {}})
        deadline = time.monotonic() + 150
        interrupted = False
        while time.monotonic() < deadline:
            result = call('/api/commands/' + key)
            if interrupt and not interrupted and result.get('phase') == 'transport':
                call('/api/command', {'command_id': 'transport-stop-' + uuid4().hex, 'skill': 'stop'})
                interrupted = True
            if result['state'] not in ('accepted', 'running'):
                if interrupt:
                    assert interrupted and result['state'] == 'cancelled', result
                report['commands'].append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(json.dumps({k: result.get(k) for k in ('skill', 'ok', 'elapsed_s', 'message', 'route', 'evaluation')}, ensure_ascii=False), flush=True)
                return result
            time.sleep(.25)
        raise TimeoutError(key)

    status = call('/api/status')
    assert status['phase'] == 'idle' and status['held_object'] is None, 'Start with an idle, empty gripper'
    assert status['capabilities']['configured_label_required'] is False
    empty = command('place_held', {'destination': {'label': args.destination}})
    assert not empty['ok'] and not empty['events'], 'An empty gripper must not start transport'
    grasp = command('grasp', {'target': {'category': args.target}})
    assert grasp['ok'], grasp['message']
    held = call('/api/status')['holding']
    assert held['verified'] and held['grasp_command_id'] == grasp['command_id']
    missing = command('place_held', {'destination': {'label': 'purple flying elephant'}})
    assert not missing['ok']
    assert call('/api/status')['holding']['instance_id'] == held['instance_id']
    call('/api/command', {'command_id': 'holding-stop-' + uuid4().hex, 'skill': 'stop'})
    if args.interrupt_placement:
        interrupted = command('place_held', {'destination': {'label': args.destination}}, interrupt=True)
        assert call('/api/status')['holding']['verified']
        assert not set(e['phase'] for e in interrupted['events']) & {'release', 'close_gripper'}
    placed = command('place_held', {'destination': {'label': args.destination}})
    assert placed['ok'], placed['message']
    assert placed['route']['grasp'] == 'retained_from_previous_command'
    phases = [e['phase'] for e in placed['events']]
    assert 'holding_resumed' in phases and 'release' in phases
    assert not set(phases) & {'pregrasp', 'approach', 'close_gripper', 'grasp_candidates', 'lift'}
    assert call('/api/status')['held_object'] is None
    report['passed'] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('HOLDING CONTINUATION PASSED', flush=True)


if __name__ == '__main__': main()
