"""Send real natural-language tasks through the deployed workbench.

This moves the live simulated Panda. It does not mock sockets or reset the scene.
"""
import argparse
import json
import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8991')
parser.add_argument('--browser', default=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
parser.add_argument('--commands', nargs='+', default=['拿起齿轮', '在桌子上随便找个地方放下'])
parser.add_argument('--output', default='output/ui-validation/held-placement.json')
args = parser.parse_args()
out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
rows, errors, wire = [], [], []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=args.browser)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    page.on('pageerror', lambda e: errors.append(str(e)))
    def receive(raw):
        if not isinstance(raw, str): return
        event = json.loads(raw)
        if event.get('type') in {'bus.event', 'reply.final', 'error'}:
            wire.append(event)
    page.on('websocket', lambda ws: ws.on('framereceived', receive))
    page.goto(args.url, wait_until='networkidle')
    # Inspect the rendered entry page before operating it.
    (out.with_suffix('.entry.txt')).write_text(page.locator('body').inner_text())
    print(page.get_by_role('button').all_text_contents(), flush=True)
    page.get_by_role('button', name='基础抓放').click()
    page.get_by_role('button', name='进入仿真', exact=True).click()
    expect(page.get_by_label('Isaac Sim 实时预览')).to_be_visible()
    for command in args.commands:
        start = len(wire); started = time.monotonic()
        page.get_by_label('刘工语音助手', exact=True).hover()
        page.get_by_role('button', name='文字输入', exact=True).click()
        page.get_by_label('输入任务', exact=True).fill(command)
        page.get_by_role('button', name='发送任务', exact=True).click()
        print('SENT:', command, flush=True)
        terminal = None
        while time.monotonic() - started < 180:
            page.wait_for_timeout(300)
            terminal = next((e for e in wire[start:] if e.get('event_type') in
                {'execution.completed', 'execution.failed', 'clarification.requested', 'task.cancelled'}), None)
            if terminal: break
        page.wait_for_timeout(1500)
        status = page.request.get(args.url.rstrip('/')+'/api/status').json()
        row = {'text': command, 'elapsed_s': time.monotonic() - started,
               'terminal': terminal, 'wire': wire[start:],
               'holding': status.get('holding'), 'result': status.get('last_result')}
        rows.append(row)
        out.write_text(json.dumps({'rows': rows, 'browser_errors': errors}, ensure_ascii=False, indent=2))
        page.get_by_role('tab', name='机械臂配置', exact=True).click()
        page.screenshot(path=str(out.with_name(out.stem+f'-{len(rows)}.png')))
        print(json.dumps({k: row[k] for k in ['text', 'elapsed_s', 'holding', 'result']}, ensure_ascii=False), flush=True)
        assert terminal, 'No terminal task event from the real conversation socket'
        assert terminal['event_type'] == 'execution.completed', terminal
        assert row['result']['ok'], row['result']
        assert row['result']['evaluation']['physical_success'], row['result']
        if row['result']['skill'] == 'place_held':
            assert row['result']['route']['grasp'] == 'retained_from_previous_command'
            assert row['result']['evaluation']['released']
            assert not row['holding']['verified']
            execution = terminal['payload']['result']
            assert not any(e['phase'] in {'pregrasp', 'close_gripper', 'selected_grasp'} for e in execution.get('events', []))
    assert not errors, errors
    browser.close()
