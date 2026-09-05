"""Bounded appearance association, not semantic recognition or a 6D tracker."""
import cv2
import numpy as np


def appearance_matches(rgb, chromaticity):
    colors=np.asarray(rgb,np.float32).reshape(-1,3)
    normalized=colors/np.maximum(colors.sum(axis=1,keepdims=True),12)
    anchor=np.asarray(chromaticity,np.float32)
    anchor_hsv=cv2.cvtColor(anchor.reshape(1,1,3),cv2.COLOR_RGB2HSV)[0,0]
    # Neutral objects have undefined hue: keep the stricter chromaticity check.
    if anchor_hsv[1]<.25:
        return np.linalg.norm(normalized-anchor,axis=1)<.10
    hsv=cv2.cvtColor((colors/255).reshape(-1,1,3),cv2.COLOR_RGB2HSV)[:,0]
    hue=np.abs(hsv[:,0]-anchor_hsv[0]);hue=np.minimum(hue,360-hue)
    # Hue tolerates shading/saturation changes but excludes neutral highlights,
    # near-black unobservable pixels, and materially different object colors.
    chroma=np.linalg.norm(normalized-anchor,axis=1)<.13
    return (chroma | (hue<12))&(hsv[:,1]>.25)&(hsv[:,2]>.06)
