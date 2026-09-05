"""Live simulated-ASR interrupt check. Does NOT test microphone/audio recognition."""
import json
import time
import uuid
from urllib.request import Request, urlopen

CONVERSATION = 'voice-motion-check-' + uuid.uuid4().hex


def request(port, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    with urlopen(Request(f'http://127.0.0.1:{port}{path}', body,
                         {'Content-Type': 'application/json'}), timeout=10) as response:
        return json.load(response)


def event(kind, text):
    return request(3100, '/v1/events', dict(source_agent_id='robot.stt',
                   event_type=kind, correlation_id=CONVERSATION,
                   payload={'text': text, 'source': 'stt'}))


def utterance(text):
    event('transcript.final', text)
    event('intent.created', text)


def wait(predicate):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        motion = request(7861, '/api/status')['motion']
        if predicate(motion):
            return motion
        time.sleep(.1)
    raise TimeoutError(motion)


def run(text, skill):
    old = request(7861, '/api/status')['motion']['last_command']['command_id']
    utterance(text)
    result = wait(lambda m: m['last_command']['command_id'] != old and
                  m['last_command']['skill'] == skill and
                  m['last_command']['state'] in {'completed', 'failed'})
    assert result['last_command']['state'] == 'completed', result
    return result


if __name__ == '__main__':
    print('SIMULATED ASR EVENTS', CONVERSATION, flush=True)
    run('速度每秒10度', 'set_speed')
    for repeat in range(2):
        utterance('手腕转到30度')
        active = wait(lambda m: m['mode'] == 'moving')
        time.sleep(.6)
        sent = time.monotonic()
        event('transcript.delta', '停一下')
        held = wait(lambda m: m['last_command']['skill'] == 'hold' and
                    m['last_command']['state'] == 'completed')
        assert held['can_resume'], held
        print('HOLD', repeat, 'latency_seconds', round(time.monotonic()-sent, 3),
              'wrist_deg', held['joint_positions_deg']['wrist_roll'], flush=True)
        event('transcript.final', '停一下')
        event('intent.created', '停一下')
        resumed = run('继续执行', 'resume')
        assert abs(resumed['joint_positions_deg']['wrist_roll']-30) < 1.5
        run('手腕转到0度', 'move_joint')
    run('停止', 'stop')
    assert not request(7861, '/api/status')['motion']['can_resume']
    run('速度每秒28.65度', 'set_speed')
    run('归位', 'home')
    print('SIMULATED ASR INTERRUPT CHECK PASSED', flush=True)
