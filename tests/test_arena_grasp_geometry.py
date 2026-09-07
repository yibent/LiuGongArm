import numpy as np
from scipy.spatial.transform import Rotation
from mr_liu.arena.grasp_geometry import finger_sweep_points, top_grasp_orientation


def test_dense_neighbours_change_approach_yaw_without_changing_target_position():
    pick=np.array([.5,-.2,.048])
    theta,z=np.meshgrid(np.linspace(0,2*np.pi,50),np.linspace(.002,.06,32))
    part=np.c_[.014*np.cos(theta.ravel()),.014*np.sin(theta.ravel()),z.ravel()]
    scene=np.r_[part+[.5,-.2,0],part+[.5,-.156,0],part+[.544,-.2,0],part+[.544,-.156,0]]
    original=np.diag([1.,-1.,-1.])
    pose=np.eye(4);pose[:3,:3]=original;pose[:3,3]=pick
    before=finger_sweep_points(scene,pose)
    quat,info=top_grasp_orientation(pick,scene,original)
    pose[:3,:3]=Rotation.from_quat(np.roll(quat,-1)).as_matrix()
    assert before>100
    assert finger_sweep_points(scene,pose)<before*.1
    assert info['observed_sweep_points']==finger_sweep_points(scene,pose)
    np.testing.assert_allclose(pose[:3,2],[0,0,-1],atol=1e-9)


def test_unobstructed_approach_preserves_current_yaw():
    current=(Rotation.from_euler('z',45,degrees=True)*Rotation.from_euler('x',np.pi)).as_matrix()
    quat,info=top_grasp_orientation(np.array([.4,0,.04]),np.empty((0,3)),current)
    np.testing.assert_allclose(Rotation.from_quat(np.roll(quat,-1)).as_matrix(),current,atol=1e-9)
    assert info['observed_sweep_points']==0
