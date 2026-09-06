"""Replay an exact recorded model request, never a robot command."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.grasp.backends.m2t2 import M2T2Client,M2T2Backend
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('record',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--surface-range-m',type=float)
    args=parser.parse_args()
    request=json.loads(args.record.read_text(encoding='utf-8'))['request']
    if args.surface_range_m is not None:
        support=request.get('support_z_m')
        if support is None:
            support=M2T2Backend.support_height(np.asarray(request['scene_points']),np.asarray(request['object_points']))
        if support is None:raise ValueError('No observed support mode to normalize')
        request.update(support_z_m=support,input_surface_range_m=args.surface_range_m)
    response=M2T2Client(timeout_s=120.).infer(**request)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'kind':'exact_model_request_replay','physical_success':None,
        'source':str(args.record),'request':request,'response':response}),encoding='utf-8')
    print(json.dumps({k:v for k,v in response.items() if k not in ('grasps','placements')}))
    print('grasp_candidates',len(response['grasps']),'place_candidates',len(response['placements']))


if __name__=='__main__':main()
