"""Opt-in small simulated-joint move and return, measuring endpoint oscillation."""
import json
import time
import uuid
import argparse
from urllib.request import Request, urlopen


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request('http://127.0.0.1:7861'+path, data, {'Content-Type': 'application/json'}), timeout=5) as r:
        return json.load(r)


def move(joint, target):
    cid = 'settling-' + uuid.uuid4().hex
    before = request('/api/status')['motion']
    assert before['mode'] != 'moving' and not before['active_command_id'], 'Arm busy'
    start = before['joint_positions_deg'][joint]
    sign = 1 if target >= start else -1
    result = request('/api/command', dict(skill='move_joint', params=dict(joint=joint, degrees=target, absolute=True), command_id=cid))
    assert result['ok'], result
    began, finished = time.monotonic(), None
    samples = []
    while time.monotonic()-began < 45:
        motion = request('/api/status')['motion']
        record = motion['last_command']
        assert record['command_id'] == cid, 'Another command arrived; stop this test without overriding it'
        samples.append(motion)
        if record['state'] in ('completed', 'failed', 'cancelled'):
            finished = finished or time.monotonic()
            if time.monotonic()-finished >= 1: break
        time.sleep(.025)
    record = request('/api/commands/'+cid)
    if record['state'] not in ('completed', 'failed', 'cancelled'):
        current = request('/api/status')['motion']
        if current['active_command_id'] == cid:
            request('/api/command', dict(skill='hold', params={}, command_id='settling-stop-'+uuid.uuid4().hex))
        raise TimeoutError('Test exceeded 45 seconds; requested hold for its own motion')
    trace = [s for i,s in enumerate(samples) if i == 0 or s['sampled_at'] != samples[i-1]['sampled_at']]
    positions = [s['joint_positions_deg'][joint] for s in trace]
    at_goal = [q for q in positions if abs(q-target) <= 1.5]
    intervals = sorted(b['sampled_at']-a['sampled_at'] for a,b in zip(trace,trace[1:]))
    print(json.dumps(dict(joint=joint, start_deg=start, target_deg=target, state=record['state'],
        wall_seconds=time.monotonic()-began, overshoot_deg=max(0,max(sign*(q-target) for q in positions)),
        near_target_peak_to_peak_deg=max(at_goal)-min(at_goal) if at_goal else None,
        final_error_deg=abs(positions[-1]-target),
        completion_error_deg=record.get('max_joint_error_deg'), completion_speed_deg_s=record.get('max_joint_speed_deg_s'),
        observed_update_median_s=intervals[len(intervals)//2] if intervals else None,
        observed_update_max_s=max(intervals) if intervals else None,
        frames=len(trace), message=record['message']), ensure_ascii=False), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint', choices=['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'], default='shoulder_pan')
    args = parser.parse_args()
    initial = request('/api/status')
    assert not initial['follow_enabled'], 'Disable follow before running this explicit motion test'
    joint = args.joint
    angle = initial['motion']['joint_positions_deg'][joint]
    target = angle+5 if angle < 100 else angle-5
    outcome = move(joint, target)
    assert outcome['state'] in ('completed', 'failed'), outcome
    # Only our own finished command may be followed by the return move.
    assert request('/api/status')['motion']['last_command']['command_id'] == outcome['command_id']
    returned = move(joint, angle)
    assert outcome['state'] == returned['state'] == 'completed', 'Small-move settling test failed'
