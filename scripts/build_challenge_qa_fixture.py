"""Build a small Q8 audit fixture, not a task solution or semantic asset map.

Three incoming regions contain 2/5/3 reflective blind sleeves, with reversed
and fallen orientations, plus a fallen sleeve in a six-cell box and a stacking
station. Ground truth goes to a separate audit file, never to BusAgent.
Run with Isaac Sim's Python (usd-core is sufficient).
"""
from pathlib import Path
import json, math
import build_industrial_assets as assets
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/scenes/challenge_qa'

def quaternion(degrees):
    # Rotation about Y, as XYZW expected by this Arena checkout.
    a=math.radians(degrees)/2
    return [0.,math.sin(a),0.,math.cos(a)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assets.OUT=OUT
    p=assets.Part(1,(.58,.60,.63),.95,.13)
    p.cylinder(.014,.004,(0,0,.002));p.ring(.014,.009,.056,z=.004);p.save()
    p=assets.Part(2,(.025,.16,.55),.04,.45)
    p.box((.144,.100,.008),(0,0,.004))
    for x in (-.070,.070):p.box((.004,.100,.028),(x,0,.022))
    for y in (-.048,.048):p.box((.144,.004,.028),(0,y,.022))
    for x in (-.0233,.0233):p.box((.004,.096,.025),(x,0,.0205))
    p.box((.140,.004,.025),(0,0,.0205))
    for y in (-.062,.062):
        p.box((.052,.008,.012),(0,y,.032))
        for x in (-.022,.022):p.box((.008,.019,.009),(x,y*.88,.027))
    p.save()
    p=assets.Part(3,(.30,.33,.35),.65,.3)
    p.box((.19,.17,.07),(0,0,.035));p.save()
    config=json.loads((ROOT/'configs/arena_panda_industrial.json').read_text())
    config['entities']=[]
    config['camera'].update(width=1280,height=960)
    config['camera']['scene'].update(position=[.70,-.40,.76],lookat=[.52,0,.02])
    config['camera']['side'].update(position=[.80,.55,.65],lookat=[.52,0,.02])
    truth={'description':'Reduced Q8 fixture: three regions 2/5/3, six-cell movable bin, fallen bin part, one stack base.',
           'normal_pose':'closed flat end up, local +Z (open bore) down',
           'regions':{},'parts':[],'cell_centres':[], 'scale_note':'10 incoming parts and 6 cells, not a pixel-exact reconstruction or full-scale benchmark.'}
    groups={'A':[(.31,-.14,180),(.35,-.14,180)],
            'B':[(.47,-.22,180),(.514,-.22,0),(.47,-.176,180),(.514,-.176,180),(.562,-.132,90)],
            'C':[(.695,-.16,180),(.735,-.16,180),(.715,-.112,0)]}
    index=0
    for name,rows in groups.items():
        truth['regions'][name]={'count':len(rows),'part_ids':[]}
        for x,y,angle in rows:
            index+=1;identifier=f'qa_body_{index:02d}'
            z=.062 if angle==180 else .002 if angle==0 else .016
            if angle==90:x-=.03
            config['entities'].append({'name':identifier,'usd_path':'assets/scenes/challenge_qa/asset_01.usda',
                'position':[x,y,z],'orientation':quaternion(angle),'dynamic':True,'mass':.08})
            truth['regions'][name]['part_ids'].append(identifier)
            truth['parts'].append({'id':identifier,'region':name,'state':'normal' if angle==180 else 'inverted' if angle==0 else 'fallen'})
    # Drop the fallen sleeve above the dividers, then let physics settle it.
    # Spawning at floor height interpenetrates the divider across its length.
    config['entities'].append({'name':'qa_body_11','usd_path':'assets/scenes/challenge_qa/asset_01.usda',
        'position':[.465,.156,.070],'orientation':quaternion(90),'dynamic':True,'mass':.08})
    truth['parts'].append({'id':'qa_body_11','region':'bin','state':'fallen'})
    for identifier,asset,pos,dynamic,mass in [
        ('qa_body_12',2,[.49,.18,0.],True,.3),
        ('qa_body_13',3,[.75,.21,0.],False,2.),
        ('qa_body_14',2,[.75,.21,.072],True,.3)]:
        config['entities'].append({'name':identifier,'usd_path':f'assets/scenes/challenge_qa/asset_{asset:02d}.usda',
            'position':pos,'orientation':[0,0,0,1],'dynamic':dynamic,'mass':mass})
    truth['incoming_bin_id']='qa_body_12';truth['stack_base_bin_id']='qa_body_14'
    for row,y in enumerate((-.023,.023),1):
        for col,x in enumerate((-.0467,0,.0467),1):
            truth['cell_centres'].append({'row':row,'column':col,'position':[.49+x,.18+y,.008]})
    config['vision']['vocabulary']=['metal cylinder','sleeve','box','tray','robot arm','table']
    (ROOT/'configs/arena_panda_challenge_qa.json').write_text(json.dumps(config,indent=2)+'\n')
    (OUT/'audit_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    print('Fixture built; truth is separate from inference configuration.')
if __name__=='__main__':main()
