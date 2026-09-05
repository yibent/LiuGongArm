"""Optional SAM2 contour refinement; Florence still selects the destination."""
import time
import cv2
import numpy as np
from .contracts import PlaceError


class Sam2PayloadRefiner:
    def __init__(self,transport,trace=lambda e:None):
        self.transport,self.trace=transport,trace
        self.key,self.mask=None,None

    def __call__(self,obs,seed):
        key=(obs.camera_frame,obs.sequence)
        if self.key==key:return self.mask.copy()
        seed=np.asarray(seed,bool)
        started=time.monotonic()
        response=self._infer(obs,seed)
        if response.get('sequence')!=obs.sequence:raise PlaceError('held_mask_frame_mismatch')
        score=float(response.get('segmentation_score',0.))
        if response.get('mask') is None or not np.isfinite(score) or score<.60:
            raise PlaceError('held_mask_model_uncertain')
        mask=np.asarray(response['mask'],bool)
        if mask.shape!=seed.shape:raise PlaceError('held_mask_shape_mismatch')
        ratio=float(mask.sum()/seed.sum());overlap=float((mask&seed).sum()/seed.sum())
        if not .5<ratio<3. or overlap<.8:raise PlaceError('held_mask_identity_mismatch')
        count,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        if sum(stats[i,cv2.CC_STAT_AREA]>=32 for i in range(1,count))!=1:
            raise PlaceError('held_mask_ambiguous')
        self.trace({'phase':'place_segment_payload','model':'SAM2.1-hiera-tiny',
                    'sequence':obs.sequence,'elapsed_s':time.monotonic()-started,
                    'score':score,'pixels':int(mask.sum())})
        self.key,self.mask=key,mask.copy()
        return mask

    def warm(self,obs,seed):
        # This old reference image only initializes GPU weights; its mask is
        # discarded, including segmentation ambiguity. It is NOT a live token.
        started=time.monotonic();response=self._infer(obs,seed)
        if response.get('sequence')!=obs.sequence:raise PlaceError('held_mask_frame_mismatch')
        self.trace({'phase':'place_model_warmup','model':'SAM2.1-hiera-tiny',
                    'elapsed_s':time.monotonic()-started,'mask_used_for_control':False})

    def _infer(self,obs,seed):
        seed=np.asarray(seed,bool);yy,xx=np.nonzero(seed)
        if len(xx)<32:raise PlaceError('held_mask_seed_insufficient')
        box=[max(0,int(xx.min())-2),max(0,int(yy.min())-2),
             min(seed.shape[1]-1,int(xx.max())+2),min(seed.shape[0]-1,int(yy.max())+2)]
        index=np.argmin((xx-np.median(xx))**2+(yy-np.median(yy))**2)
        return self.transport._request({'action':'segment','rgb':obs.rgb,'box':box,
            'point':[int(xx[index]),int(yy[index])],'sequence':obs.sequence})
