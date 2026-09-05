"""Opt-in live Isaac motion smoke test. Moves the simulated arm and returns home."""
import argparse
import json
import time
import uuid
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7861')
    args = parser.parse_args()

    def request(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = Request(args.url + path, body, {'Content-Type': 'application/json'})
        with urlopen(req, timeout=10) as response:
            return json.load(response)

    def submit(skill, params=None):
        result = request('/api/command', dict(skill=skill, params=params or {}, command_id='verify-' + uuid.uuid4().hex))
        assert result['ok'], result
        return result['command_id']

    def wait(cid, expected='completed'):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = request('/api/commands/' + cid)
            if result['state'] in {'completed', 'failed', 'cancelled'}:
                print(json.dumps(result, ensure_ascii=False), flush=True)
                assert result['state'] == expected, result
                return result
            time.sleep(.25)
        raise TimeoutError(cid)

    def run(skill, params=None, expected='completed'):
        return wait(submit(skill, params), expected)

    print('BEFORE', json.dumps(request('/api/status')['motion'], ensure_ascii=False), flush=True)
    run('home')
    run('move_joint', {'joint': 'shoulder_pan', 'degrees': 10})
    run('move_joint', {'joint': 'shoulder_pan', 'degrees': 0, 'absolute': True})
    run('move_joint', {'joint': 'wrist_roll', 'degrees': 10})
    run('gripper', {'opening': 1})
    run('gripper', {'opening': 0})
    run('move_cartesian', {'xyz_m': [0, 0, .005]})
    run('rotate', {'axis': 'z', 'degrees': 5, 'frame': 'tool'})
    run('rotate', {'axis': 'z', 'degrees': 90, 'frame': 'world'}, expected='failed')
    run('set_speed', {'degrees_per_second': 10})
    interrupted = submit('move_joint', {'joint': 'wrist_roll', 'degrees': 30})
    time.sleep(1)
    run('hold')
    wait(interrupted, 'cancelled')
    run('resume')
    run('stop')
    run('resume', expected='failed')
    run('set_speed', {'degrees_per_second': 28.65})
    run('home')
    print('ALL LIVE MOTION CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
