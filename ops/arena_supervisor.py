"""Readiness adapters for Supervisor; no daemonization or restart loop here."""
import argparse
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import time
import urllib.request

from arena_stack import ROOT, STATE, SERVICES, scene_environment


def wait_for(description, probe):
    started = time.monotonic()
    last_notice = -30.
    while True:
        try:
            if probe(): return
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        elapsed = time.monotonic() - started
        if elapsed - last_notice >= 30:
            print(f'Waiting for {description} ({elapsed:.0f}s)', flush=True)
            last_notice = elapsed
        time.sleep(2)


def command_ready(command, env=None):
    return subprocess.run(command, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=5).returncode == 0


def http_ready(port, path='/health'):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{port}{path}', timeout=3) as response:
        return bool(json.load(response).get('ready'))


def port_ready(port):
    with socket.create_connection(('127.0.0.1', port), timeout=2): return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', choices=['database', *SERVICES])
    name = parser.parse_args().service
    env = {**os.environ, 'BUSAGENT_PORT': '3100', 'BUSAGENT_ROBOT': 'franka_panda'}
    STATE.mkdir(parents=True, exist_ok=True)
    if name == 'database':
        directory = Path('/run/mysqld'); directory.mkdir(parents=True, exist_ok=True)
        user = pwd.getpwnam('mysql'); os.chown(directory, user.pw_uid, user.pw_gid)
        command, cwd = ['/usr/sbin/mariadbd', '--user=mysql', '--bind-address=127.0.0.1', '--port=3307'], ROOT
    else:
        command, cwd = SERVICES[name]
    if name in {'vision', 'graspgenx', 'anyplace', 'arena'}:
        wait_for('NVIDIA driver', lambda: command_ready(['/usr/bin/nvidia-smi', '-L']))
    if name == 'arena':
        env = scene_environment(env)
        runtime = Path(env.get('XDG_RUNTIME_DIR', '/tmp/runtime-xdg'))
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        wait_for('platform X display', lambda: command_ready(
            ['/usr/bin/xdpyinfo', '-display', env.get('DISPLAY', ':20')], env))
        wait_for('vision HTTP', lambda: http_ready(5570))
        wait_for('GraspGenX socket', lambda: port_ready(5556))
        wait_for('AnyPlace HTTP', lambda: http_ready(5590))
    if name == 'busagent':
        wait_for('MariaDB', lambda: command_ready(
            ['/usr/bin/mariadb-admin', '--socket=/run/mysqld/mysqld.sock', 'ping', '--silent']))
        wait_for('Arena HTTP', lambda: http_ready(7861))
    # Keep existing diagnostics compatible. exec preserves the supervised PID.
    (STATE/f'{name}.json').write_text(json.dumps({
        'pid': os.getpid(), 'command': command, 'started_at': time.time(), 'manager': 'supervisor'}))
    os.chdir(cwd)
    print(f'Starting {name} under Supervisor', flush=True)
    os.execvpe(command[0], command, env)


if __name__ == '__main__': main()
