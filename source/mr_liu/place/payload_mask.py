"""Optional SAM2 contour refinement; Florence still selects the destination."""
import time
import cv2
import numpy as np
from .contracts import PlaceError
from mr_liu.perception.optical_flow import OpticalFlowTracker


class Sam2PayloadRefiner:
    def __init__(self,transport,trace=lambda e:None,*,track_between_inference=False):
        self.transport,self.trace=transport,trace
        self.key,self.mask=None,None
        self.tracking=track_between_inference
        self.flow=OpticalFlowTracker()
        self.model_key=None;self.model_timestamp_s=0.;self.tracked_frames=0

    @staticmethod
    def validate_mask(mask,seed):
        mask=np.asarray(mask,bool)
        if mask.shape!=seed.shape:raise PlaceError('held_mask_shape_mismatch')
        ratio=float(mask.sum()/seed.sum());overlap=float((mask&seed).sum()/seed.sum())
        if not .5<ratio<3. or overlap<.8:raise PlaceError('held_mask_identity_mismatch')
        count,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        if sum(stats[i,cv2.CC_STAT_AREA]>=32 for i in range(1,count))!=1:
            raise PlaceError('held_mask_ambiguous')
        return mask

    def __call__(self,obs,seed):
        key=(obs.camera_frame,obs.sequence)
        if self.key==key:return self.mask.copy()
        seed=np.asarray(seed,bool)
        if seed.sum()<32:raise PlaceError('held_mask_seed_insufficient')
        started=time.monotonic()
        if (self.tracking and self.model_key is not None and key[0]==self.model_key[0]
                and self.tracked_frames<3 and 0<=obs.timestamp_s-self.model_timestamp_s<=2.):
            proposed=self.flow.update(obs.rgb)
            if proposed is not None:
                try:tracked=self.validate_mask(proposed,seed)
                except PlaceError:tracked=None
                if tracked is not None:
                    self.key,self.mask=key,tracked.copy();self.tracked_frames+=1
                    self.trace({'phase':'place_track_payload','sequence':obs.sequence,
                                'model_sequence':self.model_key[1],**self.flow.diagnostic,
                                'elapsed_s':time.monotonic()-started})
                    return tracked
        response=self._infer(obs,seed)
        if response.get('sequence')!=obs.sequence:raise PlaceError('held_mask_frame_mismatch')
        score=float(response.get('segmentation_score',0.))
        if response.get('mask') is None or not np.isfinite(score) or score<.60:
            raise PlaceError('held_mask_model_uncertain')
        mask=self.validate_mask(response['mask'],seed)
        self.trace({'phase':'place_segment_payload','model':'SAM2.1-hiera-tiny',
                    'sequence':obs.sequence,'elapsed_s':time.monotonic()-started,
                    'score':score,'pixels':int(mask.sum())})
        self.key,self.mask=key,mask.copy()
        if self.tracking:
            self.flow.initialize(obs.rgb,mask)
            self.model_key=key;self.model_timestamp_s=obs.timestamp_s;self.tracked_frames=0
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
