"""Exercise icon navigation and split-panel responsiveness without robot writes."""
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:5176')
parser.add_argument('--browser', default=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
args = parser.parse_args()
out = Path(__file__).resolve().parents[1] / 'output/ui-validation'
out.mkdir(parents=True, exist_ok=True)
workspace = {
    'available': True, 'scene_id': 'initial',
    'scenes': [{'id': 'initial', 'name': '基础抓放', 'kind': 'pick', 'description': '桌面物体'}],
    'objects': [
        {'id': 'part_01', 'name': '红色方块', 'position': [.48, -.12, .025], 'rotation': [0, 0, 0], 'size': [.045] * 3, 'color': [.8, .1, .05]},
        {'id': 'part_02', 'name': '黄色圆柱', 'position': [.51, .15, .04], 'rotation': [0, 0, 0], 'size': [.06, .06, .08], 'color': [.9, .7, .1]},
        {'id': 'part_03', 'name': '蓝色方块', 'position': [.62, .08, .03], 'rotation': [0, 0, 0], 'size': [.05] * 3, 'color': [.1, .3, .8]},
    ],
    'controller': {'position_tolerance_m': .006, 'rotation_tolerance_deg': 5, 'max_steps': 360},
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=args.browser)
    page = browser.new_page(viewport={'width': 1440, 'height': 960})
    errors, commands = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.route('**/api/workspace', lambda route: route.fulfill(json=workspace))
    page.route('**/v1/robot/status', lambda route: route.fulfill(json={
        'phase': 'idle', 'motion': {'joint_positions_deg': {f'joint{i}': value for i, value in enumerate([0, -30, 0, -125, 0, 90, 45])}},
    }))
    def command(route):
        commands.append(route.request.post_data_json)
        route.fulfill(json={'ok': True, 'state': 'completed', 'message': 'mocked'})
    page.route('**/api/command', command)
    page.route_web_socket('**/v1/stt', lambda ws: ws.on_message(lambda message: None))
    page.goto(args.url, wait_until='networkidle')
    page.get_by_role('button', name='基础抓放').click()
    page.get_by_role('button', name='进入仿真').click()
    inspector = page.get_by_label('属性与配置')
    rail = page.get_by_role('tablist', name='功能选择')
    expect(rail).to_have_attribute('aria-orientation', 'vertical')
    tabs = rail.get_by_role('tab')
    assert tabs.count() == 4
    boxes = [tab.bounding_box() for tab in tabs.all()]
    assert all(abs(box['x'] - boxes[0]['x']) < 1 for box in boxes)
    assert all(boxes[i + 1]['y'] >= boxes[i]['y'] + boxes[i]['height'] for i in range(3))
    assert tabs.evaluate_all('els => els.every(e => e.textContent.trim() === "" && e.querySelector("svg"))')
    page.get_by_role('tab', name='场景物体').hover()
    expect(page.get_by_role('tooltip')).to_contain_text('场景物体')
    page.get_by_role('tab', name='节点详情').focus()
    page.keyboard.press('ArrowDown')
    expect(page.get_by_role('tab', name='场景物体')).to_have_attribute('aria-selected', 'true')

    def resize(fraction):
        handle = page.locator('[data-panel-resize-handle-id][data-panel-group-direction="horizontal"]')
        box = handle.bounding_box()
        group = handle.locator('..').bounding_box()
        page.mouse.move(box['x'] + 2, box['y'] + 30)
        page.mouse.down()
        page.mouse.move(group['x'] + group['width'] * fraction, box['y'] + 30, steps=12)
        page.mouse.up()
        page.wait_for_timeout(100)

    def check_split(first, second, wide):
        a, b = page.locator(first).bounding_box(), page.locator(second).bounding_box()
        if wide:
            assert abs(a['y'] - b['y']) < 1 and b['x'] >= a['x'] + a['width'] - 1, (a, b)
        else:
            assert abs(a['x'] - b['x']) < 1 and b['y'] >= a['y'] + a['height'] - 1, (a, b)
        assert page.locator('.ui-tabs-content:visible').evaluate('e => e.scrollWidth <= e.clientWidth + 1')

    def check_preview():
        panel = page.get_by_label('Isaac Sim 实时预览').bounding_box()
        for control in [page.get_by_role('button', name='中断', exact=True), page.get_by_role('button', name='重置', exact=True), page.get_by_label('预览精度'), page.get_by_role('button', name='三视图', exact=True)]:
            b = control.bounding_box()
            assert b['x'] >= panel['x'] and b['x'] + b['width'] <= panel['x'] + panel['width'], (b, panel)
        scene, side, wrist = [page.locator('.camera-feed.' + view).bounding_box() for view in ['scene', 'side', 'wrist']]
        assert scene['width'] * scene['height'] > side['width'] * side['height']
        assert abs(side['width'] - wrist['width']) < 1 and abs(side['height'] - wrist['height']) < 1
        assert page.locator('.preview-panel').evaluate('e => e.scrollWidth <= e.clientWidth + 1')

    check_split('.object-browser', '.object-controls', False)
    page.get_by_label('位置 X', exact=True).fill('0.55')
    resize(.9)  # Reach the limit through the real drag gesture.
    assert inspector.bounding_box()['width'] > 1440 * .7
    check_split('.object-browser', '.object-controls', True)
    expect(page.get_by_label('位置 X', exact=True)).to_have_value('0.55')
    check_preview()
    page.screenshot(path=str(out / 'inspector-objects-wide.png'))
    resize(.26)
    check_split('.object-browser', '.object-controls', False)
    expect(page.get_by_label('位置 X', exact=True)).to_have_value('0.55')
    page.get_by_role('tab', name='机械臂配置').click()
    check_split('.robot-summary', '.robot-controls', False)
    resize(.7)
    check_split('.robot-summary', '.robot-controls', True)
    check_preview()
    page.screenshot(path=str(out / 'inspector-robot-wide.png'))
    page.set_viewport_size({'width': 1024, 'height': 768})
    resize(.75)
    check_split('.robot-summary', '.robot-controls', True)
    check_preview()
    page.screenshot(path=str(out / 'inspector-robot-1024-wide.png'))
    resize(.26)
    page.get_by_role('tab', name='场景物体').click()
    check_split('.object-browser', '.object-controls', False)
    field = page.locator('.vector-inputs').first
    inputs = [label.bounding_box() for label in field.locator('label').all()]
    assert inputs[1]['y'] > inputs[0]['y'] and inputs[2]['y'] > inputs[1]['y']
    page.screenshot(path=str(out / 'inspector-objects-1024-narrow.png'))
    assert not errors, errors
    assert not commands, commands  # Resizing and browsing must never move anything.
    (out / 'inspector-regression.json').write_text(json.dumps({'passed': True, 'checks': ['icon-only vertical rail', 'tooltips', 'arrow-key navigation', '75 percent maximum', 'panel-based stacked and side-by-side object and robot layouts', 'draft survives resizing', 'narrow preview controls contained', 'main camera larger and other cameras equal', 'narrow vector fields', 'no commands on browsing'], 'browser_errors': errors}, ensure_ascii=False, indent=2))
    print('PASS: inspector sidebar, wide/narrow split layouts, preserved draft, narrow preview, no robot commands')
    browser.close()
