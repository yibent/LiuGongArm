"""Host snapshots of Isaac Lab Torch and Warp sensor arrays."""
import numpy as np
from scipy.spatial.transform import Rotation

def numpy_data(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def pose_matrix(position, quaternion_xyzw):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    pose[:3, 3] = position
    return pose
