"""Exercise manual UI gestures against mocked command/teleop endpoints."""
import argparse,json,os,time
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:5176')
parser.add_argument('--browser',default=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
args=parser.parse_args()
out=Path(__file__).resolve().parents[1]/'output/ui-validation';out.mkdir(parents=True,exist_ok=True)
state={'available':True,'api_version':2,'scene_id':'initial','scenes':[{'id':'initial','name':'基础抓放','kind':'pick','description':'当前场景'}], 'controls':{'robot_pose':True,'jog':True,'teleop':True,'gripper':True},'robot':{'position':[.4,0,.3],'rotation':[180,0,0],'gripper':1,'joint_positions_deg':[0]*9},'objects':[{'id':'part_01','name':'红色方块','position':[.5,0,.03],'rotation':[0,0,0],'size':[.05]*3,'color':[.8,.1,.1],'editable':True}], 'controller':{'position_tolerance_m':.006,'rotation_tolerance_deg':5,'max_steps':360}}
state['objects'].append({**state['objects'][0], 'id':'part_02', 'name':'黄色方块'})
commands=[];updates=[];results={};errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path=args.browser)
 page=browser.new_page(viewport={'width':1440,'height':960})
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('console',lambda msg:errors.append(msg.text) if 'same key' in msg.text else None)
 page.route('**/api/workspace',lambda r:r.fulfill(json=state))
 page.route('**/api/status',lambda r:r.fulfill(json={'command_id':None}))
 page.route_web_socket('**/v1/stt',lambda ws:ws.on_message(lambda m:None))
 def command(r):
  body=r.request.post_data_json;commands.append(body);params=body.get('params',{})
  result={'ok':True,'state':'running' if params.get('action')=='teleop' else 'completed','message':'mocked'}
  results[body['command_id']]=result
  r.fulfill(json=result)
 def update(r):
  body=r.request.post_data_json;updates.append(body)
  if body.get('stop') and body['command_id'] in results: results[body['command_id']]['state']='completed'
  r.fulfill(json={'ok':True})
 page.route('**/api/command',command);page.route('**/api/teleop',update)
 page.route('**/api/commands/*',lambda r:r.fulfill(json=results[r.request.url.rsplit('/',1)[1]]))
 page.goto(args.url,wait_until='networkidle')
 assert page.locator('link[rel="icon"]').get_attribute('href')=='/brand/liugong-logo.png'
 expect(page.locator('.brand-logo')).to_be_visible();assert page.locator('.brand-logo').evaluate('e=>e.naturalWidth>1000')
 page.get_by_role('button',name='基础抓放').click();page.get_by_role('button',name='进入仿真').click()
 page.get_by_role('tab',name='机械臂配置').click()
 pad=page.get_by_label('机械臂操作区')
 expect(pad.get_by_role('button',name='移动 X+',exact=True)).to_be_visible()
 pad.get_by_role('button',name='移动 X+',exact=True).click()
 page.wait_for_timeout(180)
 assert commands[-1]['params']['translation']==[.005,0,0]
 # A long press creates one session with replaceable velocity updates; release stops that same ID.
 btn=pad.get_by_role('button',name='移动 Y+',exact=True);box=btn.bounding_box();page.mouse.move(box['x']+15,box['y']+15);page.mouse.down();page.wait_for_timeout(680);page.mouse.up()
 page.wait_for_timeout(300)
 sessions=[c for c in commands if c.get('params',{}).get('action')=='teleop']
 assert len(sessions)==1 and sessions[-1]['params']['linear']==[0,.02,0]
 assert any(u.get('stop') and u['command_id']==sessions[-1]['command_id'] for u in updates)
 count=len(commands);page.wait_for_timeout(400);assert len(commands)==count
 # A dragged joystick sends a direction, then a scoped stop, and never fabricates another command.
 joy=pad.get_by_label('平面摇杆');joy.scroll_into_view_if_needed();box=joy.bounding_box()
 page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2);page.mouse.down();page.mouse.move(box['x']+box['width']*.8,box['y']+box['height']*.3);page.wait_for_timeout(400);page.mouse.up();page.wait_for_timeout(300)
 assert any(u.get('linear',[0,0])[0]>0 and u.get('linear',[0,0])[1]>0 for u in updates)
 assert updates[-1].get('stop')
 # Rotation + alternate plane + tool frame is passed as structured geometry.
 pad.get_by_role('button',name='旋转',exact=True).click();pad.get_by_label('操作平面').select_option('XZ');pad.get_by_label('控制坐标系').select_option('tool')
 pad.get_by_role('button',name='旋转 Z+',exact=True).click();page.wait_for_timeout(200)
 assert commands[-1]['params']['rotation']==[0,0,5] and commands[-1]['params']['frame']=='tool'
 # Space key gets the same one-step control as a pointer click.
 pad.get_by_role('button',name='旋转 X+',exact=True).focus();page.keyboard.press('Space');page.wait_for_timeout(200)
 assert commands[-1]['params']['rotation']==[5,0,0]
 # Losing focus during a hold stops and cannot leave the repeating gesture active.
 btn=pad.get_by_role('button',name='旋转 Z+',exact=True);box=btn.bounding_box();page.mouse.move(box['x']+15,box['y']+15);page.mouse.down();page.wait_for_timeout(400);page.evaluate('window.dispatchEvent(new Event("blur"))');page.mouse.up();page.wait_for_timeout(300)
 assert updates[-1].get('stop')
 page.get_by_role('button',name='打开夹爪').click();page.wait_for_timeout(200);assert commands[-1]['params']['state']=='open'
 page.get_by_role('button',name='闭合夹爪').click();page.wait_for_timeout(200);assert commands[-1]['params']['state']=='close'
 page.get_by_label('末端位置 X',exact=True).fill('0.43');page.wait_for_timeout(1300);expect(page.get_by_label('末端位置 X',exact=True)).to_have_value('0.43')
 page.get_by_role('button',name='移动到此位姿').click();page.wait_for_timeout(200);assert commands[-1]['params']['position'][0]==.43
 page.get_by_role('tab',name='场景物体').click()
 page.locator('.object-list button').filter(has_text='part_02').click();page.wait_for_timeout(1100)
 assert page.get_by_label('物体操作区').count()==1
 page.locator('.object-list button').filter(has_text='part_01').click();page.wait_for_timeout(1100)
 assert page.get_by_label('物体操作区').count()==1
 pad=page.get_by_label('物体操作区');pad.get_by_role('button',name='移动 X+',exact=True).click();page.wait_for_timeout(200)
 assert commands[-1]['params']['id']=='part_01' and commands[-1]['params']['target']=='object'
 page.get_by_label('位置 X',exact=True).fill('0.56');page.wait_for_timeout(1300);expect(page.get_by_label('位置 X',exact=True)).to_have_value('0.56')
 # Wide and narrow controls remain inside their scrollable pane.
 h=page.locator('[data-panel-resize-handle-id][data-panel-group-direction="horizontal"]');box=h.bounding_box();page.mouse.move(box['x']+2,box['y']+25);page.mouse.down();page.mouse.move(1000,box['y']+25,steps=10);page.mouse.up();page.wait_for_timeout(150)
 page.locator('.ui-tabs-content:visible').evaluate('e=>e.scrollTop=0')
 page.screenshot(path=str(out/'manual-objects-wide.png'))
 assert page.locator('.ui-tabs-content:visible').evaluate('e=>e.scrollWidth<=e.clientWidth+1')
 page.get_by_role('tab',name='机械臂配置').click();page.locator('.ui-tabs-content:visible').evaluate('e=>e.scrollTop=0');page.screenshot(path=str(out/'manual-robot-wide.png'))
 page.set_viewport_size({'width':1024,'height':768});h=page.locator('[data-panel-resize-handle-id][data-panel-group-direction="horizontal"]');box=h.bounding_box();page.mouse.move(box['x']+2,box['y']+25);page.mouse.down();page.mouse.move(280,box['y']+25,steps=10);page.mouse.up();page.wait_for_timeout(150)
 page.locator('.ui-tabs-content:visible').evaluate('e=>e.scrollTop=0');page.screenshot(path=str(out/'manual-robot-narrow.png'))
 assert page.locator('.ui-tabs-content:visible').evaluate('e=>e.scrollWidth<=e.clientWidth+1')
 assert not errors,errors
 (out/'manual-controls.json').write_text(json.dumps({'passed':True,'commands':commands,'teleop_updates':updates,'errors':errors},ensure_ascii=False,indent=2))
 print('PASS: logo, tap/hold/joystick/release/blur/keyboard, rotation and tool frame, object and TCP drafts survive refresh, gripper, responsive controls')
 browser.close()
