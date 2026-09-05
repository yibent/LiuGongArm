"""Bounded background recording; encoding never manufactures camera freshness."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import cv2
import numpy as np


class AsyncRGBDRecorder:
    def __init__(self,output,max_pending=4):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='place-rgbd-writer')
        self.pending=deque();self.max_pending=max_pending;self.errors=[]

    def _collect(self):
        future=self.pending.popleft()
        try:future.result()
        except Exception as exc:self.errors.append(repr(exc))

    def __call__(self,obs):
        while self.pending and (self.pending[0].done() or len(self.pending)>=self.max_pending):
            self._collect()
        # Observations own their numpy snapshots; the writer never calls Kit or
        # a camera. Bounded backpressure may delay a call; freshness gates remain.
        self.pending.append(self.pool.submit(self._write,obs))

    def _write(self,obs):
        frame=re.sub(r'[^A-Za-z0-9_.-]','_',obs.camera_frame)
        tag=f'{frame}_{obs.sequence:04d}'
        if not cv2.imwrite(str(self.output/(tag+'.png')),obs.rgb[:,:,::-1]):
            raise OSError('Cannot write RGB recording')
        robot=obs.metadata.get('robot_self_mask')
        np.savez_compressed(self.output/(tag+'.npz'),rgb=obs.rgb,depth_m=obs.depth_m,
            K=obs.intrinsics.matrix,T_base_camera=obs.T_base_camera,
            robot_self_mask=np.zeros(obs.depth_m.shape,bool) if robot is None else robot,
            has_robot_self_mask=robot is not None,
            T_base_ee=np.empty((0,0)) if obs.T_base_ee is None else obs.T_base_ee,
            T_ee_camera=np.empty((0,0)) if obs.T_ee_camera is None else obs.T_ee_camera,
            sequence=obs.sequence,timestamp_s=obs.timestamp_s)

    def close(self):
        self.pool.shutdown(wait=True)
        while self.pending:self._collect()
        return self.errors.copy()
