"""Click the deployed workbench's real controls and restore the initial scene.
This changes the live simulation. All movement originates from the visible UI.
"""
import argparse,json,math,os,time
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:8999')
parser.add_argument('--browser',default=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
args=parser.parse_args();out=Path(__file__).resolve().parents[1]/'output/ui-validation';out.mkdir(parents=True,exist_ok=True)
checks=[];commands=[];errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path=args.browser)
 page=browser.new_page(viewport={'width':1440,'height':960})
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:commands.append(r.post_data_json) if r.method=='POST' and r.url.endswith('/api/command') else None)
 def state():return page.request.get(args.url.rstrip('/')+'/api/workspace').json()
 def result_after(index):
  end=time.monotonic()+40
  while len(commands)<=index and time.monotonic()<end:page.wait_for_timeout(50)
  assert len(commands)>index,'No command was sent'
  command=commands[-1]
  while time.monotonic()<end:
   result=page.request.get(args.url.rstrip('/')+'/api/commands/'+command['command_id']).json()
   if result.get('state') not in ['accepted','running']:
    assert result.get('ok'),result
    page.wait_for_timeout(200)
    return result
   page.wait_for_timeout(150)
  raise AssertionError('Control command did not finish')
 def reset():
  page.get_by_role('button',name='重置',exact=True).click()
  index=len(commands);page.get_by_role('menuitem',name='物体与机械臂',exact=True).click()
  expect(page.get_by_role('status')).to_contain_text('初始状态已恢复',timeout=40000)
  if page.get_by_role('button',name='关闭提示').count():page.get_by_role('button',name='关闭提示').click()
 def record(name,**data):checks.append({'check':name,**data});print('PASS:',name,flush=True)
 def distance(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
 page.goto(args.url,wait_until='domcontentloaded')
 page.get_by_role('button',name='基础抓放').click();page.get_by_role('button',name='进入仿真').click()
 assert page.request.get(args.url.rstrip('/')+'/api/status').json()['command_id'] is None
 try:
  reset();baseline=state()
  page.wait_for_function("[...document.querySelectorAll('canvas')].filter(e=>e.width===800).length===3",timeout=30000)
  assert page.locator('.brand-logo').evaluate('e=>e.naturalWidth===1254')
  assert page.locator('.orb-logo').evaluate('e=>e.naturalWidth===1254')
  record('deployed logo and 3 real cameras')
  handle=page.locator('[data-panel-resize-handle-id][data-panel-group-direction="horizontal"]');b=handle.bounding_box()
  page.mouse.move(b['x']+2,b['y']+30);page.mouse.down();page.mouse.move(1000,b['y']+30,steps=12);page.mouse.up()
  page.get_by_role('tab',name='场景物体').click()
  target='entity_03';page.locator('.object-list button').filter(has_text=target).click()
  def object_position():return next(o['position'] for o in state()['objects'] if o['id']==target)
  before=object_position();index=len(commands)
  page.get_by_label('物体操作区').get_by_role('button',name='移动 X+',exact=True).click();result_after(index)
  after=object_position();assert after[0]-before[0]>.002,(before,after)
  record('real object D-pad',id=target,moved_m=distance(before,after))
  page.locator('.ui-tabs-content:visible').evaluate('e=>e.scrollTop=0');page.screenshot(path=str(out/'live-object-controls.png'))
  page.get_by_role('tab',name='机械臂配置').click();pad=page.get_by_label('机械臂操作区')
  before=state()['robot']['position'];index=len(commands)
  pad.get_by_role('button',name='移动 Z+',exact=True).click();result_after(index)
  after=state()['robot']['position'];assert after[2]-before[2]>.001,(before,after)
  record('real Panda D-pad through Arena IK',moved_m=distance(before,after))
  pad.get_by_role('button',name='旋转',exact=True).click();before=state()['robot']['rotation'][2];index=len(commands)
  pad.get_by_role('button',name='旋转 Z+',exact=True).click();result_after(index)
  after=state()['robot']['rotation'][2];angle=(after-before+180)%360-180
  assert angle>1,(before,after)
  record('real Panda rotation control',measured_rotation_deg=angle)
  pad.get_by_role('button',name='位置',exact=True).click();before=state()['robot']['position'];index=len(commands)
  joy=pad.get_by_label('平面摇杆');box=joy.bounding_box()
  page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2);page.mouse.down();page.mouse.move(box['x']+box['width']*.8,box['y']+box['height']/2);page.wait_for_timeout(900);page.mouse.up();result_after(index)
  after=state()['robot']['position'];page.wait_for_timeout(700);stopped=state()['robot']['position']
  assert distance(before,after)>.003,(before,after)
  assert distance(after,stopped)<.004,(after,stopped)
  record('real browser joystick release',moved_m=distance(before,after),drift_after_release_m=distance(after,stopped))
  index=len(commands);page.get_by_role('button',name='闭合夹爪').click();result_after(index)
  opening=math.radians(sum(state()['robot']['joint_positions_deg'][-2:]));assert opening<.012,opening
  index=len(commands);page.get_by_role('button',name='打开夹爪').click();result_after(index)
  opening=math.radians(sum(state()['robot']['joint_positions_deg'][-2:]));assert opening>.065,opening
  record('gripper fingers physically close and open',open_width_m=opening)
  page.locator('.ui-tabs-content:visible').evaluate('e=>e.scrollTop=0');page.screenshot(path=str(out/'live-robot-controls.png'))
 finally:
  reset()
  final=state()
  (out/'live-controls.json').write_text(json.dumps({'checks':checks,'commands':commands,'errors':errors,'final':final},ensure_ascii=False,indent=2))
 assert distance(final['robot']['position'],baseline['robot']['position'])<.008
 assert all(distance(o['position'],next(row['position'] for row in baseline['objects'] if row['id']==o['id']))<.006 for o in final['objects'])
 assert not errors,errors
 print('PASS: public UI controls and complete initial-state restoration; no browser errors')
 browser.close()
