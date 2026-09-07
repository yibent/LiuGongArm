"""Offline Q8 evaluator. Its fixture truth must never be sent to BusAgent.

Checks true end orientation and cell seating separately from the controller's
observed-axis evaluator. A geometric snapshot alone cannot prove release,
contact/stability or autonomous target selection; action evidence remains required.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(workspace,truth):
    rows={r['id']:r for r in workspace['objects']}
    bin_=rows[truth['incoming_bin_id']]
    R=Rotation.from_euler('xyz',bin_['rotation'],degrees=True).as_matrix()
    origin=np.asarray(bin_['position'])
    # Centres in the offline fixture truth are in its original world frame.
    initial=np.mean([c['position'] for c in truth['cell_centres']],axis=0);initial[2]=0.
    cells=[{**c,'local':np.asarray(c['position'])-initial} for c in truth['cell_centres']]
    parts=[]
    for spec in truth['parts']:
        row=rows[spec['id']];partR=Rotation.from_euler('xyz',row['rotation'],degrees=True).as_matrix()
        root=np.asarray(row['position']);axis=partR[:,2]
        centre=root+axis*.03
        local=R.T@(centre-origin)
        cell=min(cells,key=lambda c:np.linalg.norm(local[:2]-c['local'][:2]))
        offset=local-cell['local'];bottom=float(min(root[2],(root+axis*.06)[2]))
        angle=float(np.rad2deg(np.arccos(np.clip(-axis[2],-1.,1.))))
        in_cell=bool(np.max(np.abs(offset[:2]))<.009 and abs(bottom-(origin+R@cell['local'])[2])<.008)
        parts.append({'id':spec['id'],'original_region':spec['region'],'closed_end_up':angle<12.,
            'closed_axis_tilt_deg':angle,'cell':[cell['row'],cell['column']] if in_cell else None,
            'centre_world_m':centre.tolist(),'bottom_world_m':bottom})
    seated=[p for p in parts if p['cell'] is not None and p['closed_end_up']]
    filled={tuple(p['cell']) for p in seated}
    return {'parts':parts,'correctly_seated_count':len(seated),'unique_filled_cells':len(filled),
        'capacity':len(cells),'geometrically_full':len(filled)==len(cells),
        'initial_fallen_part_repaired':next(p for p in parts if p['original_region']=='bin') in seated,
        'bin_position_world_m':bin_['position'], 'bin_rotation_deg':bin_['rotation'],
        'evidence_scope':'offline_geometry_only; release/contact/stability and autonomous decisions require action and Bus traces'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path)
    p.add_argument('--truth',type=Path,default=Path(__file__).resolve().parents[1]/'assets/scenes/challenge_qa/audit_truth.json')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=evaluate(json.loads(args.workspace.read_text()),json.loads(args.truth.read_text()))
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='parts'},ensure_ascii=False))
if __name__=='__main__':main()
