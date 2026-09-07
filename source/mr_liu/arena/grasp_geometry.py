"""Rank Panda approaches against observed depth, without asset identities."""
import numpy as np
from scipy.spatial.transform import Rotation


def finger_sweep_points(scene, pose, approach=.12):
    """Observed points in the open fingers' straight approach volume.

    Panda fingers move along hand Y. Bounds cover the repository's official
    finger meshes (X ±10.6 mm, outward Y 26.4 mm, hand Z 58.4–112.3 mm),
    expressed about the Arena TCP at hand Z 103.4 mm, with a 1 mm margin.
    This ranks local approaches; it does not certify whole-arm clearance.
    """
    local=(np.asarray(scene)-pose[:3,3])@pose[:3,:3]
    occupied=(np.abs(local[:,0])<.0116)&(np.abs(local[:,1])>.039)
    occupied&=(np.abs(local[:,1])<.0674)&(local[:,2]>-.046-approach)&(local[:,2]<.010)
    return int(np.count_nonzero(occupied))


def top_grasp_orientation(position, scene, current_rotation):
    """Choose a yaw using the actual gaps around the observed target."""
    ranked=[]
    scene=np.asarray(scene)
    scene=scene[np.linalg.norm(scene-position,axis=1)<.23]
    for yaw in range(0,360,15):
        rotation=(Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_euler('x',np.pi)).as_matrix()
        pose=np.eye(4);pose[:3,3]=position;pose[:3,:3]=rotation
        occupied=finger_sweep_points(scene,pose)
        angle=Rotation.from_matrix(rotation@current_rotation.T).magnitude()
        ranked.append((occupied,angle,yaw,rotation))
    occupied,_,yaw,rotation=min(ranked,key=lambda row:(row[0],row[1]))
    return np.roll(Rotation.from_matrix(rotation).as_quat(),1), {'yaw_deg':yaw,
        'observed_sweep_points':occupied, 'alternatives':[{'yaw_deg':r[2],'observed_sweep_points':r[0]} for r in ranked]}
