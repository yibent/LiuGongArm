"""Run official AnyPlace on a recorded point-cloud pair. No robot commands."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from mr_liu.place.anyplace_runtime import AnyPlaceRuntime


def read_cloud(path):
    if path.suffix=='.npy': return np.load(path,allow_pickle=False)
    vertices=PlyData.read(path)['vertex']
    return np.column_stack([vertices[name] for name in ('x','y','z')])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--child',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--candidates',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=50)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Do not overwrite previous evidence')
    runtime=AnyPlaceRuntime(args.checkpoint,root=ROOT)
    result=runtime.infer(read_cloud(args.parent),read_cloud(args.child),seed=args.seed,
                         candidates=args.candidates,iterations=args.iterations)
    result.update(parent_source=str(args.parent.resolve()),child_source=str(args.child.resolve()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='transforms'},indent=2))


if __name__=='__main__': main()
