"""Local AnyPlace inference service. No robot commands; weights load lazily."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from mr_liu.place.anyplace_runtime import AnyPlaceRuntime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=5590)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'_models/anyplace/anyplace_ckpts/anyplace_multitask/model.pth')
    parser.add_argument('--record-dir',type=Path)
    args=parser.parse_args()
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    state={'runtime':None}
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,data):
            blob=json.dumps(data,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(blob)));self.end_headers();self.wfile.write(blob)
        def do_GET(self):
            self.respond(200 if self.path=='/health' else 404,{'service':'busagent-anyplace',
                'protocol':1,'ready':True,'model_loaded':state['runtime'] is not None})
        def do_POST(self):
            if self.path!='/infer':return self.respond(404,{'error':'not_found'})
            record=None;request=None
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8*1024*1024:raise ValueError('Invalid request size')
                request=json.loads(self.rfile.read(size))
                if request.get('protocol')!=1 or type(request.get('sequence')) is not int:
                    raise ValueError('Invalid protocol/observation sequence')
                if args.record_dir:
                    args.record_dir.mkdir(parents=True,exist_ok=True)
                    record=args.record_dir/f"request_{request['sequence']}_{time.time_ns()}.json"
                    record.write_text(json.dumps({'request':request},allow_nan=False),encoding='utf-8')
                if state['runtime'] is None:state['runtime']=AnyPlaceRuntime(args.checkpoint,root=ROOT)
                result=state['runtime'].infer(request['parent'],request['child'],
                    seed=request.get('seed',0),candidates=request.get('candidates',20),
                    iterations=request.get('iterations',50),
                    init_current_orientation=bool(request.get('init_current_orientation',False)))
                result.update(sequence=request['sequence'],protocol=1,
                              input_geometry=request.get('input_geometry','unspecified'))
                if record:record.write_text(json.dumps({'request':request,'response':result},allow_nan=False),encoding='utf-8')
                self.respond(200,result)
            except Exception as exc:
                error={'error':type(exc).__name__,'message':str(exc)}
                if record:record.write_text(json.dumps({'request':request,'error':error}),encoding='utf-8')
                self.respond(422,error)
    print('[AnyPlace] interface ready; lazy model loading',flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
