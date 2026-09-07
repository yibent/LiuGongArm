import numpy as np
from scipy.spatial.transform import Rotation
from mr_liu.arena.orientation import align_vector, placement_pose, endpoint_check


def test_horizontal_and_inverted_axes_align_without_reflection():
    for axis in ([1.,0,0], [0,0,-1.], [.2,.3,.4]):
        value=align_vector(axis,[0,0,1])
        np.testing.assert_allclose(value @ (np.asarray(axis)/np.linalg.norm(axis)),[0,0,1],atol=1e-8)
        assert np.isclose(np.linalg.det(value),1.)


def test_sideways_payload_becomes_upright_and_keeps_actual_tcp_offset():
    points=np.array([[x,y,z] for x in [-.03,.03] for y in [-.01,.01] for z in [-.01,.01]])+[.5,.1,.2]
    tcp=np.eye(4);tcp[:3,3]=[.515,.1,.2];tcp[:3,:3]=Rotation.from_euler('x',180,degrees=True).as_matrix()
    delta=align_vector([1,0,0],[0,0,1])
    pose,placed=placement_pose(points,tcp,delta,[.45,.2,.008])
    np.testing.assert_allclose(placed.min(0),[.44,.19,.011],atol=1e-8)
    np.testing.assert_allclose(placed.max(0),[.46,.21,.071],atol=1e-8)
    before=(points-tcp[:3,3])@tcp[:3,:3]
    after=(placed-pose[:3,3])@pose[:3,:3]
    np.testing.assert_allclose(before,after,atol=1e-8)


def test_physical_orientation_evaluation_rejects_wrong_end_and_fallen_part():
    assert endpoint_check([0,0,1],[0,0,0,1],1)['satisfied']
    assert not endpoint_check([0,0,1],[0,0,0,1],0)['satisfied']
    assert not endpoint_check([0,0,1],Rotation.from_euler('y',90,degrees=True).as_quat(),1)['satisfied']
    assert endpoint_check([0,0,1],Rotation.from_euler('y',180,degrees=True).as_quat(),0)['satisfied']
