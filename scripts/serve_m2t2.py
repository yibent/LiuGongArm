"""Local, isolated official M2T2 inference. No robot commands are accepted."""
import argparse
import json
import sys
import time
import traceback
import base64
import hashlib
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))


def placement_indices(contacts, scores, bounds=None, limit=512):
    """Filter by requested observed region BEFORE top-k; never alter a pose.

    Keep the complete scene as network input. The semantic destination is a
    proposal constraint, not a replacement for footprint/collision validation.
    """
    valid=np.isfinite(scores)
    if bounds is not None:
        bounds=np.asarray(bounds,float)
        if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[0]>bounds[1]):
            raise ValueError('Invalid placement bounds')
        valid &= np.all((contacts>=bounds[0]-.003)&(contacts<=bounds[1]+.003),axis=1)
    indices=np.flatnonzero(valid)
    return indices[np.argsort(-scores[indices],kind='stable')[:limit]],int(valid.sum())


def install_grasp_decoder_compatibility():
    """Use the vendor's pose formula with an explicit vector cross dimension."""
    import m2t2.action_decoder as decoder
    from mr_liu.grasp.backends.m2t2_decoder import build_6d_grasp
    decoder.build_6d_grasp = build_6d_grasp


class Model:
    def __init__(self, args):
        import torch
        from omegaconf import OmegaConf
        if args.ops == 'portable':
            from mr_liu.grasp.backends.m2t2_ops import install_portable_ops
            install_portable_ops()
        sys.path.insert(0, str(args.vendor))
        from m2t2.m2t2 import M2T2
        install_grasp_decoder_compatibility()
        torch.set_num_threads(4)
        self.torch, self.args = torch, args
        self.cfg = OmegaConf.load(args.vendor/'config.yaml')
        self.cfg.eval.world_coord = True
        self.cfg.eval.placement_height = 0.
        self.cfg.m2t2.action_decoder.max_num_pred = 512
        self.model = M2T2.from_config(self.cfg.m2t2)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model = self.model.cuda().eval()
        self.last_metrics = {}
        try:
            vendor_revision=subprocess.check_output(['git','-C',str(args.vendor),'rev-parse','HEAD'],text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            vendor_revision='unavailable'
        self.provenance={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'vendor_revision':vendor_revision,
            'server_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'ops_sha256':hashlib.sha256((ROOT/'source/mr_liu/grasp/backends/m2t2_ops.py').read_bytes()).hexdigest(),
            'torch_version':torch.__version__,
            'pose_compatibility':'decoder_cross_dim_last',
            'decoder_sha256':hashlib.sha256((ROOT/'source/mr_liu/grasp/backends/m2t2_decoder.py').read_bytes()).hexdigest()}

    def segment(self,request):
        import cv2
        torch=self.torch
        if not hasattr(self,'sam'):
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
                str(ROOT/'_models/sam2/sam2.1_hiera_tiny.pt'),device='cuda'))
        rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(request['rgb_png']),np.uint8),1),cv2.COLOR_BGR2RGB)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            self.sam.set_image(rgb)
            point=request.get('point')
            masks,scores,_=self.sam.predict(box=np.asarray(request['box']),
                point_coords=None if point is None else np.asarray([point]),
                point_labels=None if point is None else np.asarray([1]),multimask_output=True)
        index=int(np.argmax(scores));score=float(scores[index])
        _,encoded=cv2.imencode('.png',masks[index].astype(np.uint8)*255)
        return {'backend':'sam2','protocol':1,'sequence':request.get('sequence'),
                'segmentation_score':score,'mask_png':base64.b64encode(encoded).decode()}

    def infer(self, request):
        torch = self.torch
        if request['task'] not in ('pick', 'place'):
            raise ValueError('Unsupported task')
        rng = np.random.default_rng(int(request.get('seed', 0)))
        seed=int(request.get('seed',0))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        def cloud(name):
            value = np.asarray(request[name], np.float32)
            if value.ndim != 2 or value.shape[1] != 3 or not 32 <= len(value) <= 150000:
                raise ValueError('Invalid '+name)
            if not np.isfinite(value).all() or np.max(np.abs(value)) > 20:
                raise ValueError('Non-metric or invalid points')
            return value
        scene, obj = cloud('scene_points'), cloud('object_points')
        ee = np.asarray(request['ee_pose'], np.float32)
        from mr_liu.grasp.transforms import assert_transform
        assert_transform(ee)
        obj_ee = (obj-ee[:3, 3]) @ ee[:3, :3]
        scene_ids = rng.choice(len(scene), self.cfg.data.num_points, replace=len(scene)<self.cfg.data.num_points)
        obj_ids = rng.choice(len(obj), 1024, replace=len(obj)<1024)
        feature_scene=scene.copy()
        surface_range=float(request.get('input_surface_range_m',0.))
        if not 0 <= surface_range <= .02:
            raise ValueError('Input surface normalization exceeds upstream envelope')
        if surface_range:
            support_z=float(request['support_z_m'])
            if not np.isfinite(support_z):raise ValueError('Invalid support height')
            feature_scene[np.abs(feature_scene[:,2]-support_z)<surface_range,2]=support_z
        scene_input = np.c_[feature_scene-feature_scene.mean(0), np.zeros_like(scene)]
        if request['task'] == 'place':
            object_input = np.c_[obj_ee-obj_ee.mean(0), np.zeros_like(obj_ee)][obj_ids]
        else:
            # Match official pick preprocessing (object context is random).
            object_input = rng.random((1024, 6), dtype=np.float32)
        bottom = obj.min(0)
        bottom[:2] = np.unique(np.rint(obj[:, :2]/.01), axis=0).mean(0)*.01
        if 'bottom_center' in request:
            bottom=np.asarray(request['bottom_center'],np.float32)
            if bottom.shape!=(3,) or not np.isfinite(bottom).all():
                raise ValueError('Invalid verified bottom center')
        arrays = {'inputs':scene_input[scene_ids], 'points':scene[scene_ids],
                  'object_inputs':object_input, 'ee_pose':ee, 'bottom_center':bottom,
                  'cam_pose':np.eye(4, dtype=np.float32)}
        data = {k:torch.as_tensor(v, device='cuda', dtype=torch.float32).unsqueeze(0) for k,v in arrays.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        diagnostics={}
        def record_logits(module,inputs,output):
            diagnostics['logits']=output[1][-1]['placement_masks'].detach()
        hook=self.model.transformer.register_forward_hook(record_logits)
        try:
            with torch.inference_mode():
                outputs = self.model.infer(data, self.cfg.eval)
        finally:
            hook.remove()
        torch.cuda.synchronize()
        result = {'backend':'m2t2', 'protocol':1, 'ops':self.args.ops,
                  'model_provenance':self.provenance,
                  'sequence':request.get('sequence'), 'task':request['task'],
                  'inference_ms':(time.perf_counter()-started)*1000,
                  'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
                  'grasps':[], 'placements':[]}
        def cpu(tensor):return tensor.detach().cpu().numpy()
        result['seed']=seed
        result['input_surface_range_m']=surface_range
        result['object_fps_valid_points']=int(np.sum(np.sum(object_input[:,:3]**2,axis=1)>1e-3))
        if request.get('placement_bounds') is not None:
            bounds=np.asarray(request['placement_bounds'],float)
            region=np.all((scene[scene_ids]>=bounds[0]-.003)&(scene[scene_ids]<=bounds[1]+.003),axis=1)
            confidence=cpu(diagnostics['logits'].sigmoid())[0]
            result['destination_sampled_points']=int(region.sum())
            result['destination_max_confidence_by_yaw']=confidence[:,region].max(1).tolist() if region.any() else []
        # Model groups pick candidates by predicted object, placement by yaw.
        if request['task'] == 'pick':
            for poses, scores, contacts in zip(outputs['grasps'][0], outputs['grasp_confidence'][0], outputs['grasp_contacts'][0]):
                for pose, score, contact in zip(cpu(poses), cpu(scores), cpu(contacts)):
                    result['grasps'].append({'pose':pose.tolist(), 'score':float(score), 'contact':contact.tolist()})
        else:
            result['raw_candidates']=0
            result['region_candidates']=0
            for yaw, (poses, scores, contacts) in enumerate(zip(outputs['placements'][0], outputs['placement_confidence'][0], outputs['placement_contacts'][0])):
                poses,scores,contacts=cpu(poses),cpu(scores),cpu(contacts)
                indices,count=placement_indices(contacts,scores,request.get('placement_bounds'))
                result['raw_candidates']+=len(scores)
                result['region_candidates']+=count
                for pose, score, contact in zip(poses[indices],scores[indices],contacts[indices]):
                    result['placements'].append({'pose':pose.tolist(), 'score':float(score), 'contact':contact.tolist(), 'yaw_bin':yaw})
        key = 'grasps' if request['task'] == 'pick' else 'placements'
        if key=='placements':
            # Do not let one yaw's scores erase all candidates of another yaw.
            result[key]=[item for yaw in range(8) for item in
                sorted((x for x in result[key] if x['yaw_bin']==yaw),key=lambda x:x['score'],reverse=True)[:512]]
        else:
            result[key] = sorted(result[key], key=lambda x:x['score'], reverse=True)[:4096]
        self.last_metrics = {k:v for k,v in result.items() if k not in ('grasps','placements')}
        print(json.dumps({**self.last_metrics, 'candidates':len(result[key])}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor', type=Path, default=ROOT/'_vendor/M2T2')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'_models/m2t2/m2t2.pth')
    parser.add_argument('--ops', choices=('portable','native'), default='portable')
    parser.add_argument('--port', type=int, default=5580)
    parser.add_argument('--eager',action='store_true',help='Load weights before serving; default is lazy to reduce startup commit memory')
    args = parser.parse_args()
    model = Model(args) if args.eager else None
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, status=200):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self.reply({'backend':'m2t2','protocol':1,'ops':args.ops,'ready':True,
                        'model_loaded':model is not None,'load_policy':'lazy unless --eager',
                        'model_provenance':None if model is None else model.provenance,
                        'metrics':{} if model is None else model.last_metrics})
        def do_POST(self):
            nonlocal model
            try:
                length = int(self.headers.get('Content-Length', 0))
                if self.path not in ('/infer','/segment') or not 0 < length < 30_000_000:
                    raise ValueError('Invalid request')
                request=json.loads(self.rfile.read(length))
                if model is None:model=Model(args)
                result = model.infer(request) if self.path=='/infer' else model.segment(request)
                self.reply(result)
            except Exception as exc:
                traceback.print_exc()
                self.reply({'backend':'m2t2','protocol':1,'error':repr(exc)}, 500)
    print('[M2T2] ready', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':main()
