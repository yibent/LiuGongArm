import numpy as np
from mr_liu.arena.placement_geometry import inspect_grid, inspect_grid_views, principal_axis, cell_fit, retained_grid


def grid_cloud(rows=2, columns=3, angle=0.):
    u=np.linspace(-columns*.024,columns*.024,columns*40)
    v=np.linspace(-rows*.024,rows*.024,rows*40)
    x,y=np.meshgrid(u,v)
    floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,.008)]
    lines=[]
    for along,cross in [(0,np.linspace(u.min(),u.max(),columns+1)),(1,np.linspace(v.min(),v.max(),rows+1))]:
        other=v if along==0 else u
        for line in cross:
            for width in [-.001,0,.001]:
                for z in [.028,.030,.032,.034,.036]:
                    p=np.zeros((len(other),3));p[:,along]=line+width;p[:,1-along]=other;p[:,2]=z;lines.append(p)
    points=np.concatenate([floor,*lines])
    c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s],[s,c]])
    points[:,:2]=points[:,:2]@R.T+[.5,.1]
    return points,R


def test_grid_is_inferred_for_rotated_and_differently_sized_trays():
    for rows,columns,angle in [(2,3,0.),(3,4,.4),(2,5,-.6)]:
        points,R=grid_cloud(rows,columns,angle)
        result=inspect_grid(points,camera_position=[.5,-1.,1.],camera_right=R[:,0])
        assert result['status']=='observed'
        assert (result['rows'],result['columns'])==(rows,columns)
        assert len(result['cells'])==rows*columns
        assert all(c['occupancy']=='empty' for c in result['cells'])


def test_occluded_floor_is_unknown_and_actual_raised_points_are_occupied():
    parent,_=grid_cloud()
    result=inspect_grid(parent)
    cell=result['cells'][0];center=np.array(cell['position_m'])
    inside=(np.abs(parent[:,:2]-center[:2])<.017).all(1)
    scene=parent[~inside]
    assert inspect_grid(parent,scene)['cells'][0]['occupancy']=='unknown'
    raised=center+np.c_[np.linspace(-.01,.01,20),np.zeros(20),np.full(20,.04)]
    assert inspect_grid(parent,np.r_[scene,raised])['cells'][0]['occupancy']=='occupied'


def test_plain_surface_does_not_invent_cells():
    p,_=grid_cloud()
    assert inspect_grid(p[p[:,2]<.01])['status']=='unknown'


def test_occluded_view_does_not_discard_grid_from_clear_view():
    p,_=grid_cloud();T=np.eye(4);T[:3,3]=[.5,-1,1]
    result=inspect_grid_views({'side':{'points':p[p[:,2]<.01],'T':T},'top':{'points':p,'T':T}},p)
    assert result['status']=='observed' and result['camera']=='top'
    assert result['rows']==2 and result['columns']==3
    assert result['view_diagnostics'][0]['status']=='unknown'


def test_geometry_memory_requires_current_confirmation_and_rechecks_occupancy():
    p,_=grid_cloud();grid=inspect_grid(p)
    visible=p[p[:,0]>.5]
    cell=grid['cells'][0];centre=np.asarray(cell['position_m'])
    obstacle=centre+np.c_[np.linspace(-.01,.01,20),np.zeros(20),np.full(20,.04)]
    result=retained_grid(grid,p,visible,np.r_[p,obstacle])
    assert result and result['cells'][0]['occupancy']=='occupied'
    assert grid['cells'][0]['occupancy']=='empty'
    assert retained_grid(grid,p,visible+[.12,0,0],p) is None
    assert retained_grid(grid,p,visible[:10],p) is None


def test_observed_axis_endpoints_follow_a_fallen_part():
    t=np.linspace(0,2*np.pi,60);x=np.linspace(-.03,.03,40)
    xx,tt=np.meshgrid(x,t)
    points=np.c_[xx.ravel(),.014*np.cos(tt.ravel()),.014*np.sin(tt.ravel())]+[.5,0,.02]
    result=principal_axis(points)
    assert abs(np.dot(result['axis_world'],[1,0,0]))>.99
    assert result['axis_confidence']>2
    assert np.linalg.norm(np.diff(result['endpoints_m'],axis=0))>.055


def test_partial_shaft_and_visible_end_cap_do_not_tilt_the_estimated_axis():
    from scipy.spatial.transform import Rotation
    theta,z=np.meshgrid(np.linspace(-1.25,1.25,45),np.linspace(0,.06,48))
    side=np.c_[.014*np.cos(theta.ravel()),.014*np.sin(theta.ravel()),z.ravel()]
    x,y=np.meshgrid(np.linspace(-.014,.014,50),np.linspace(-.014,.014,50))
    disk=x*x+y*y<.014**2
    cap=np.c_[x[disk],y[disk],np.full(disk.sum(),.06)]
    points=np.r_[side,cap]
    points+=np.random.default_rng(27).normal(0,.000035,points.shape)
    for angles in [[0,0,0],[70,42,19],[90,0,0]]:
        rotation=Rotation.from_euler('xyz',angles,degrees=True)
        observed=rotation.apply(points)+[.5,.1,.04]
        actual=rotation.apply([0,0,1])
        _,vectors=np.linalg.eigh(np.cov(observed.T))
        assert abs(vectors[:,-1]@actual)<np.cos(np.deg2rad(10))
        result=principal_axis(observed)
        assert result['axis_method']=='side_surface_normals'
        assert abs(np.dot(result['axis_world'],actual))>np.cos(np.deg2rad(2))


def test_one_flat_face_retains_covariance_axis_when_normals_are_ambiguous():
    x,z=np.meshgrid(np.linspace(-.01,.01,24),np.linspace(0,.06,50))
    points=np.c_[x.ravel(),np.zeros(x.size),z.ravel()]
    result=principal_axis(points)
    assert result['axis_method']=='point_covariance'
    assert abs(result['axis_world'][2])>.999


def test_cell_evaluation_rejects_wrong_cell_overhang_and_resting_on_dividers():
    points,R=grid_cloud(angle=.4)
    grid=inspect_grid(points,camera_right=R[:,0])
    cell=grid['cells'][0];centre=np.asarray(cell['position_m'])
    t=np.linspace(0,2*np.pi,80)
    shape=np.c_[.014*np.cos(t),.014*np.sin(t),np.zeros(80)]
    shape[:,:2]=shape[:,:2]@R.T
    assert cell_fit(shape+centre,cell,grid['basis_xy'])['fits']
    assert not cell_fit(shape+centre+np.r_[R[:,0]*.048,0.],cell,grid['basis_xy'])['fits']
    assert not cell_fit(shape+centre+[0,0,.028],cell,grid['basis_xy'])['fits']
    shape[:,:2]*=3
    assert not cell_fit(shape+centre,cell,grid['basis_xy'])['fits']


def test_grid_peaks_reject_weak_occluder_but_keep_strong_irregular_wall_unknown():
    from mr_liu.arena.placement_geometry import regular_dividers
    positions=np.array([0.,.048,.061,.096])
    np.testing.assert_allclose(regular_dividers(positions,[1000,1100,300,900]),[0,.048,.096])
    assert regular_dividers(positions,[1000,1100,1050,900]) is None
    assert regular_dividers([0,.032,.09],[900,1000,1000]) is None
