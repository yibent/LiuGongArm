"""Official AnyPlace diffusion inference, isolated from robot execution.

No grasp policy, learned success classifier, or collision guarantee is supplied
by this runtime. Output is a transform of the *input child cloud*, not an EE pose.
"""
from pathlib import Path
import hashlib
import os
import random
import sys
import time
import numpy as np


class NullVisualizer:
    """Disable only MeshCat rendering; the upstream diffusion code is unchanged."""
    def __getitem__(self, key): return self
    def set_object(self, *args, **kwargs): pass
    def set_transform(self, *args, **kwargs): pass
    def delete(self, *args, **kwargs): pass


def model_cloud(points, name, minimum=1024):
    """Meet the model tensor size using only measured samples, without new geometry."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f'{name}: expected finite Nx3 cloud in metres')
    observed = len(points)
    if not observed:
        raise ValueError(f'{name}: empty observed cloud')
    if observed < minimum:
        points = points[np.linspace(0, observed, minimum, endpoint=False).astype(int)]
    return points, {'observed_points': observed, 'model_input_points': len(points),
                    'repeated_samples': len(points) - observed}


def rotation_grid(size=10000):
    # Same RING HEALPix centres and Euler convention as upstream util.py.
    # astropy-healpix supplies native Windows wheels where healpy does not.
    from astropy_healpix import HEALPix
    from scipy.spatial.transform import Rotation
    level=max(int(np.round(np.log(size/72.)/np.log(8.))),0)
    nside=2**level
    xyz=np.stack(HEALPix(nside=nside,order='ring').healpix_to_xyz(np.arange(12*nside*nside)),1)
    azimuth=np.arctan2(xyz[:,1],xyz[:,0]); polar=np.arccos(xyz[:,2])
    zeros=np.zeros(len(xyz))
    first=Rotation.from_euler('xyz',np.stack([azimuth,zeros,zeros],1)).as_matrix()
    second=Rotation.from_euler('xyz',np.stack([zeros,zeros,polar],1)).as_matrix()
    return np.concatenate([first@second@Rotation.from_euler('xyz',[tilt,0,0]).as_matrix()
        for tilt in np.linspace(0,2*np.pi,6*2**level,endpoint=False)])


class AnyPlaceRuntime:
    def __init__(self, checkpoint, *, root=None):
        import torch
        self.torch=torch
        root=Path(root) if root else Path(__file__).resolve().parents[3]
        vendor=root/'_vendor/anyplace'
        sys.path[:0]=[str(vendor),str(root/'_vendor/airobot/src')]
        os.environ['ANYPLACE_SOURCE_DIR']=str(vendor/'anyplace')
        from anyplace.utils import config_util
        from anyplace.model.transformer.policy import NSMTransformerSingleTransformationRegression
        from anyplace.utils.anyplace.multistep_pose_regression_anyplace import multistep_regression_scene
        if not torch.cuda.is_available(): raise RuntimeError('AnyPlace requires CUDA')
        checkpoint=Path(checkpoint)
        # This is a downloaded official checkpoint, not an arbitrary user pickle.
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=config_util.load_config(str(vendor/'anyplace/config/full_eval_cfgs/base.yaml'),demo_train_eval='eval')
        refine=state['args']['model']['refine_pose']
        if refine['type']!='nsm_transformer': raise ValueError('Expected AnyPlace diffusion checkpoint')
        kwargs=dict(cfg['model']['nsm_transformer'])
        kwargs.update(refine.get('model_kwargs',{}).get('nsm_transformer',{}))
        self.model=NSMTransformerSingleTransformationRegression(feat_dim=refine['feat_dim'],
            mc_vis=NullVisualizer(),**kwargs).cuda().eval()
        self.model.load_state_dict(state['refine_pose_model_state_dict'],strict=True)
        self.infer_function=multistep_regression_scene
        self.rotations=rotation_grid()
        self.weight_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.checkpoint=str(checkpoint.resolve())

    def infer(self, parent, child, *, seed=0, candidates=20, iterations=50,init_current_orientation=False):
        torch=self.torch
        parent, parent_sampling = model_cloud(parent, 'parent')
        child, child_sampling = model_cloud(child, 'child')
        # 1024 is a tensor size, not a minimum number of independent measurements.
        # Repetition adds no surface information; report the observed counts.
        if not 2<=candidates<=20 or not 1<=iterations<=50: raise ValueError('Invalid inference budget')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            transforms=self.infer_function(NullVisualizer(),parent,child,None,self.model,None,
                scene_scale=1.,scene_mean=np.zeros(3),grid_pts=None,rot_grid=self.rotations,
                viz=False,n_iters=iterations,init_k_val=candidates,no_sc_score=True,
                run_affordance=False,init_parent_mean=False,init_orig_ori=init_current_orientation,
                with_coll=False,add_per_iter_noise=True,
                per_iter_noise_kwargs={'rot':{'angle_deg':20,'rate':6.5},
                                       'trans':{'trans_dist':.03,'rate':5.5}},
                timestep_emb_decay_factor=20,remove_redundant_pose=False)
        torch.cuda.synchronize()
        transforms=np.asarray(transforms)
        if transforms.ndim!=3 or transforms.shape[1:]!=(4,4) or not np.isfinite(transforms).all():
            raise ValueError('Invalid AnyPlace transform output')
        return {'backend':'anyplace','kind':'offline_model_inference_only',
            'physical_success':None,'transforms':transforms.tolist(),
            'transform_convention':'T_world_child_final = relative @ T_world_child_input',
            'seed':seed,'iterations':iterations,'candidates_requested':candidates,
            'init_current_orientation':bool(init_current_orientation),
            'inference_s':time.perf_counter()-started,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'parent_points':parent_sampling['observed_points'],'child_points':child_sampling['observed_points'],
            'input_sampling': {'parent': parent_sampling, 'child': child_sampling},
            'checkpoint':self.checkpoint,'checkpoint_sha256':self.weight_sha256,
            'learned_success_scores':None,'collision_checked':False}
