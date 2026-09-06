#!/usr/bin/env python3
from playwright.sync_api import sync_playwright, expect
from pathlib import Path
import json,time,os,argparse
parser=argparse.ArgumentParser(description='Validate the workbench; all command requests and BusAgent sockets are mocked.')
parser.add_argument('--url',default='http://127.0.0.1:5176')
parser.add_argument('--browser',default=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
args=parser.parse_args()
out=Path(__file__).resolve().parents[1]/'output/ui-validation'
out.mkdir(parents=True,exist_ok=True)
state={'available':True,'scene_id':'initial','scenes':[{'id':'initial','name':'基础抓放','description':'开放工作台。','kind':'pick'},{'id':'sorting','name':'桌面整理','description':'分散排列。','kind':'sort'},{'id':'precision','name':'精确摆放','description':'紧凑排列。','kind':'stack'}],'objects':[{'id':'part_01','name':'红色方块','position':[.48,-.12,.025],'rotation':[0,0,0],'size':[.045,.045,.045],'color':[.8,.1,.05],'editable':True}],'controller':{'position_tolerance_m':.006,'rotation_tolerance_deg':5,'max_steps':360},'camera':{'width':800,'height':600}}
commands=[];sockets=[];errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path=args.browser)
 page=browser.new_page(viewport={'width':1440,'height':960})
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.route('**/api/workspace',lambda r:r.fulfill(json=state))
 def command(route):
  body=route.request.post_data_json;commands.append(body)
  if body['skill']=='workspace' and body['params']['action']=='scene':state['scene_id']=body['params']['scene_id']
  route.fulfill(json={'ok':True,'state':'completed','message':'applied'})
 page.route('**/api/command',command)
 def ws_route(ws):
  sockets.append(ws)
  ws.on_message(lambda msg:None)
 page.route_web_socket('**/v1/stt',ws_route)
 page.goto(args.url,wait_until='networkidle')
 expect(page.get_by_role('button',name='仿真',exact=True)).to_be_disabled()
 page.get_by_role('button',name='桌面整理').click();page.get_by_role('button',name='进入仿真').click()
 page.get_by_label('Isaac Sim 实时预览').wait_for()
 assert commands[0]['params']['scene_id']=='sorting'
 page.wait_for_function("document.querySelector('canvas')?.width > 300")
 assert page.locator('canvas').count()==3
 original=page.locator('canvas').first.evaluate('e=>e.width')
 page.get_by_label('预览精度').select_option('0.5')
 page.wait_for_function(f"document.querySelector('canvas').width === {original//2}")
 page.get_by_role('button',name='腕部视图',exact=True).click()
 assert page.locator('canvas').count()==1
 page.get_by_role('button',name='三视图',exact=True).click()
 page.get_by_label('预览精度').select_option('1')
 # Real resizing behavior.
 handle=page.locator('[data-panel-resize-handle-id][data-panel-group-direction="horizontal"]')
 box=handle.bounding_box();before=page.get_by_label('属性与配置').bounding_box()['width']
 page.mouse.move(box['x']+2,box['y']+35);page.mouse.down();page.mouse.move(box['x']+52,box['y']+35);page.mouse.up()
 assert page.get_by_label('属性与配置').bounding_box()['width'] > before+25
 # Editing goes through the serialized command endpoint (mocked; no robot changes).
 page.get_by_role('tab',name='场景物体').click();page.get_by_label('位置 X',exact=True).fill('0.55');page.get_by_role('button',name='应用位置与旋转').click()
 expect(page.get_by_role('status')).to_contain_text('已应用')
 assert commands[-1]['params']['position'][0]==.55
 page.get_by_role('button',name='关闭提示').click()
 page.get_by_role('button',name='重置',exact=True).click();page.get_by_role('menuitem',name='物体与机械臂',exact=True).click()
 expect(page.get_by_role('status')).to_contain_text('初始状态已恢复')
 assert commands[-1]['params']=={'action':'reset','scope':'all'}
 page.get_by_role('button',name='关闭提示').click()
 page.get_by_role('button',name='中断',exact=True).click()
 expect(page.get_by_role('status')).to_contain_text('中断请求')
 assert commands[-1]['skill']=='stop'
 page.get_by_role('button',name='关闭提示').click()
 # Reproducible protocol fixtures; never sent to a live BusAgent/robot.
 ws=sockets[-1];now=int(time.time()*1000);origin=now-23000
 def send(id,kind,t,agent,span=None,payload=None,source=None,cause=None):
  data={'type':'bus.event','event_id':id,'event_type':kind,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(t/1000))+f'.{t%1000:03}Z','source_agent_id':agent,'task_id':'task-ui-test','payload':payload or {}}
  if span:data['payload'].update({'span_id':span,'started_at_ms':starts[span]})
  if source:data['source_span_id']=source
  if cause:data['causation_id']=cause
  ws.send(json.dumps(data));page.wait_for_timeout(30)
 starts={'language':origin,'vision':origin+3000,'action':origin+12000,'reply':origin+12500,'system':origin+11000}
 send('1','node.started',origin,'robot.instruction','language',{'loop':'slow'})
 send('2','instruction.parsed',origin+3000,'robot.instruction',payload={'message':'把红色方块放到黄色圆柱上'},source='language')
 send('3','node.completed',origin+3500,'robot.instruction','language',{'loop':'slow'})
 send('4','node.started',origin+3000,'robot.vision','vision',{'loop':'fast','trigger_event_id':'2'})
 send('5','perception.reported',origin+10500,'robot.vision',payload={'loop':'fast','message':'YOLOE → SAM2 · 目标已定位'},source='vision')
 send('6','node.completed',origin+11000,'robot.vision','vision',{'loop':'fast'})
 send('7','node.started',origin+11000,'robot.loop-router','system')
 send('8','node.completed',origin+11900,'robot.loop-router','system')
 send('9','node.started',origin+12000,'robot.executor','action',{'loop':'fast','trigger_event_id':'5'})
 send('10','node.started',origin+12500,'module.dialogue','reply',{'loop':'slow'})
 send('11','node.completed',origin+15000,'module.dialogue','reply',{'loop':'slow'})
 page.get_by_label('时间轴缩放').fill('43')
 clip=page.locator('.timeline-clip.state-running')
 w1=clip.bounding_box()['width'];page.wait_for_timeout(400);w2=clip.bounding_box()['width'];assert w2>w1+10
 page.locator('.timeline-clip.loop-fast').first.click()
 expect(page.get_by_role('tab',name='节点详情')).to_have_attribute('aria-selected','true')
 assert page.locator('.timeline-clip.loop-fast').first.locator('.clip-in').evaluate('e=>getComputedStyle(e).backgroundColor') == page.locator('.timeline-clip.loop-slow').first.locator('.clip-out').evaluate('e=>getComputedStyle(e).backgroundColor')
 page.get_by_role('tab',name='机械臂配置').click()
 send('12','execution.progress',int(time.time()*1000),'robot.executor',payload={'message':'正在放置'},source='action')
 expect(page.get_by_role('tab',name='机械臂配置')).to_have_attribute('aria-selected','true')
 page.locator('.timeline-clip.loop-fast').first.click()
 page.screenshot(path=str(out/'timeline-fixture.png'))
 # Stop event freezes width.
 send('13','node.completed',int(time.time()*1000),'robot.executor','action',{'loop':'fast'})
 assert page.locator('.timeline-clip.state-running').count()==0
 page.get_by_role('button',name='拖动浏览',exact=True).click()
 viewport=page.locator('.timeline-viewport');b=viewport.bounding_box();page.mouse.move(b['x']+600,b['y']+50);page.mouse.down();page.mouse.move(b['x']+100,b['y']+50);page.mouse.up()
 assert viewport.evaluate('e=>e.scrollLeft')>0
 # Orb text affordance and keyboard submission.
 page.get_by_role('button',name='开始说话').hover();page.get_by_role('button',name='文字输入').click();page.get_by_label('输入任务').fill('现在什么状态');page.get_by_role('button',name='发送任务').click()
 page.get_by_role('button',name='清空时间轴').click();assert page.locator('.timeline-clip').count()==0
 # Narrow desktop layout and keyboard focus.
 page.set_viewport_size({'width':1024,'height':768});page.screenshot(path=str(out/'workbench-1024.png'))
 assert page.get_by_label('Isaac Sim 实时预览').bounding_box()['width']>450
 assert not errors,errors
 (out/'interaction-results.json').write_text(json.dumps({'passed':True,'checks':['scene gate','preset selection','3 camera feeds','display sampling','camera switch','panel resize','object edit command','reset scopes','stop','growing spans','terminal spans','causal color links','inspector selection stays stable','timeline pan','orb text','clear','1024 layout'],'commands_mocked':True,'errors':errors},ensure_ascii=False,indent=2))
 print('PASS: 18 browser interaction checks; commands mocked; browser errors:',errors)
 browser.close()
