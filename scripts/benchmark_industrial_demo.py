"""Run isolated Isaac demo loads, preserving each measured report (stdlib only)."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isaac-python', required=True, help='Isaac python.bat or python.sh')
    parser.add_argument('--clutter-levels', type=int, nargs='+', default=[0, 96, 240, 500, 1000])
    parser.add_argument('--frames', type=int, default=1800)
    parser.add_argument('--count', type=int, default=48)
    parser.add_argument('--no-vision', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.frames < 1 or not 1 <= args.count <= 96 or any(n < 0 or n > 5000 for n in args.clutter_levels):
        parser.error('frames must be positive; count 1..96; clutter 0..5000')
    root = Path(__file__).resolve().parents[1]
    output = root/'output/industrial_demo'/time.strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for load in args.clutter_levels:
        command = [args.isaac_python, str(root/'scripts/run_vision_follow.py'), '--industrial-demo',
                   '--no-follow', '--demo-autostart', '--demo-count', str(args.count),
                   '--demo-clutter', str(load), '--test-frames', str(args.frames)]
        if args.no_vision:
            command.append('--demo-no-vision')
        if args.headless:
            command.append('--headless')
        started = time.time()
        with (output/f'clutter-{load}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        latest = root/'output/industrial_demo/latest.json'
        report = {'clutter': load, 'exit_code': result.returncode}
        if latest.exists() and latest.stat().st_mtime >= started:
            report.update(json.loads(latest.read_text(encoding='utf-8')))
        else:
            report['error'] = 'No fresh report; inspect log (simulation may not have started).'
        results.append(report)
        (output/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    print('Reports:', output)


if __name__ == '__main__':
    main()
