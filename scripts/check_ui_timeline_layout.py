"""Dense timeline / floating controls regression. All API writes and WS traffic
are mocked; uses the existing preview frame endpoints read-only."""
import argparse, json, os, time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:5176')
parser.add_argument('--browser',default=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
args=parser.parse_args()
out=Path(__file__).resolve().parents[1]/'output/ui-validation'
out.mkdir(parents=True,exist_ok=True)
workspace={'available':True,'scene_id':'initial','scenes':[{'id':'initial','name':'基础抓放','kind':'pick','description':'桌面物体'}],'objects':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path=args.browser)
 page=browser.new_page(viewport={'width':1440,'height':960})
 sockets=[];errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.route('**/api/workspace',lambda r:r.fulfill(json=workspace))
 page.route('**/api/command',lambda r:r.fulfill(json={'ok':True,'state':'completed','message':'mocked'}))
 page.route_web_socket('**/v1/stt',lambda ws:sockets.append(ws))
 page.goto(args.url,wait_until='networkidle')
 page.get_by_role('button',name='基础抓放').click();page.get_by_role('button',name='进入仿真').click()
 assert page.locator('.workspace-breadcrumb').count()==0
 page.get_by_role('button',name='开始说话').hover()
 button=page.get_by_role('button',name='文字输入')
 expect(button).to_be_visible()
 assert page.locator('.orb-hover-tools button').count()==1
 assert page.get_by_role('button',name='查看对话').count()==0
 bounds=button.bounding_box();icon=button.locator('svg').bounding_box();label=button.locator('span').bounding_box()
 assert bounds['width']>=100
 assert icon['x']>=bounds['x'] and label['x']+label['width']<=bounds['x']+bounds['width']
 assert page.locator('.orb-status').bounding_box()['y']+page.locator('.orb-status').bounding_box()['height']<page.locator('.timeline-footer').bounding_box()['y']
 button.click();expect(page.get_by_label('输入任务')).to_be_focused();page.get_by_role('button',name='关闭输入').click()
 # 124 rapid invocations across four tracks: every name stays legible.
 kinds=[('instruction.parsed','robot.instruction'),('perception.reported','robot.vision'),('execution.accepted','robot.executor'),('plan.validated','robot.validator')]
 epoch=int(time.time()*1000)-110000
 for i in range(124):
  kind,agent=kinds[i%4];timestamp=epoch+i*800
  sockets[-1].send(json.dumps({'type':'bus.event','event_id':f'dense-{i}','event_type':kind,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(timestamp/1000))+f'.{timestamp%1000:03}Z','source_agent_id':agent,'payload':{'loop':'fast' if i%2 else 'slow'}}))
 page.wait_for_function("document.querySelectorAll('.timeline-clip').length===124")
 def assert_layout():
  # ResizeObserver updates after a splitter gesture; wait for the rendered grid,
  # not an arbitrary delay that races when several browsers run concurrently.
  page.wait_for_function("""() => {
   const rows=document.querySelectorAll('.timeline-track');
   const viewport=document.querySelector('.timeline-viewport');
   return rows.length===4 && Math.abs(rows[3].getBoundingClientRect().bottom-viewport.getBoundingClientRect().bottom)<9;
  }""")
  measurements=page.evaluate('''() => {
   const box=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,bottom:r.bottom}};
   const viewport=document.querySelector('.timeline-viewport');
   return {viewport:box(viewport), canvas:box(document.querySelector('.timeline-canvas')), footer:box(document.querySelector('.timeline-footer')), labels:[...document.querySelectorAll('.track-label')].map(box), rows:[...document.querySelectorAll('.timeline-track')].map(box), minCard:Math.min(...[...document.querySelectorAll('.timeline-clip')].map(e=>e.getBoundingClientRect().width)), scrollHeight:viewport.scrollHeight,clientHeight:viewport.clientHeight, scrollWidth:viewport.scrollWidth,clientWidth:viewport.clientWidth,ruler:box(document.querySelector('.timeline-ruler')),head:box(document.querySelector('.track-label-head'))};}''')
  for label,row in zip(measurements['labels'],measurements['rows']):
   assert abs(label['y']-row['y'])<1
   assert abs(label['h']-row['h'])<1
  assert measurements['minCard']>=139
  assert abs(measurements['rows'][-1]['bottom']-measurements['viewport']['bottom'])<9, measurements
  assert measurements['scrollHeight']-measurements['clientHeight']<=1
  assert abs(measurements['head']['x']-measurements['viewport']['x'])<1
  return measurements
 initial=assert_layout()
 page.get_by_role('button',name='拖动浏览',exact=True).click()
 viewport=page.locator('.timeline-viewport');before=viewport.evaluate('e=>e.scrollLeft');b=viewport.bounding_box()
 page.mouse.move(b['x']+500,b['y']+50);page.mouse.down();page.mouse.move(b['x']+850,b['y']+50,steps=6);page.mouse.up()
 assert viewport.evaluate('e=>e.scrollLeft') < before - 300
 assert_layout()
 page.get_by_role('button',name='跟随',exact=True).click()
 page.get_by_role('button',name='开始说话').hover()
 page.screenshot(path=str(out/'timeline-dense-1440.png'))
 # Resizing both the entire browser and the panel keeps all four rows aligned.
 for width,height in [(1024,768),(2048,900),(1280,820)]:
  page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(100);assert_layout()
  page.screenshot(path=str(out/f'timeline-dense-{width}.png'))
 page.set_viewport_size({'width':1440,'height':960})
 handle=page.locator('[data-panel-resize-handle-id][data-panel-group-direction="vertical"]')
 b=handle.bounding_box();page.mouse.move(b['x']+200,b['y']+2);page.mouse.down();page.mouse.move(b['x']+200,b['y']-90,steps=8);page.mouse.up();page.wait_for_timeout(100)
 assert_layout()
 # Many truly simultaneous spans: one scroll container keeps labels and sticky ruler together.
 page.get_by_role('button',name='清空时间轴').click()
 for i in range(16):
  timestamp=epoch
  sockets[-1].send(json.dumps({'type':'bus.event','event_id':f'parallel-{i}','event_type':'node.started','created_at':time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(timestamp/1000))+f'.{timestamp%1000:03}Z','source_agent_id':'robot.vision','payload':{'span_id':f'p{i}','started_at_ms':timestamp,'loop':'fast'}}))
 page.wait_for_function("document.querySelectorAll('.timeline-clip').length===16")
 page.locator('.timeline-viewport').evaluate('e=>e.scrollTop=250')
 page.wait_for_timeout(50)
 labels=page.locator('.track-label').evaluate_all('els=>els.map(e=>e.getBoundingClientRect().y)');rows=page.locator('.timeline-track').evaluate_all('els=>els.map(e=>e.getBoundingClientRect().y)')
 assert all(abs(a-b)<1 for a,b in zip(labels,rows))
 assert abs(page.locator('.timeline-ruler').bounding_box()['y']-page.locator('.timeline-viewport').bounding_box()['y'])<1
 page.screenshot(path=str(out/'timeline-parallel-scroll.png'))
 assert not errors,errors
 (out/'layout-regression.json').write_text(json.dumps({'passed':True,'events':124,'viewports':[[1440,960],[1024,768],[2048,900],[1280,820]],'parallel_spans':16,'initial':initial,'browser_errors':errors},ensure_ascii=False,indent=2))
 print('PASS: 124 events, four viewport sizes, panel resize, 16 parallel spans, sticky headers, text-only orb with contained icon and label')
 browser.close()
