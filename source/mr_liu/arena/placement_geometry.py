"""Observed regular grid cells and object axes, independent of simulation names."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


def cell_fit(points, cell, basis_xy, tolerance=.002):
    """Check the payload itself against an observed cell, including resting height."""
    points = np.asarray(points, dtype=float)
    centre = np.asarray(cell['position_m'])
    local = (points[:,:2]-centre[:2])@np.asarray(basis_xy)
    low,high = np.quantile(local,[.02,.98],axis=0)
    margin = np.asarray(cell['interior_size_m'])/2-np.maximum(np.abs(low),np.abs(high))
    bottom_gap = float(np.quantile(points[:,2],.02)-centre[2])
    return {'fits': bool(np.all(margin >= -tolerance) and -.004 <= bottom_gap <= .012),
            'minimum_wall_margin_m': float(margin.min()), 'bottom_gap_m': bottom_gap,
            'row':cell['row'], 'column':cell['column']}


def inspect_grid_views(views, scene):
    """Fit each view independently; use all views to check cell occupancy.

    A side view can reveal an obstacle while obscuring the parallel divider
    structure. Merging its walls into the fitting cloud can destroy a good fit.
    """
    proposals = [{**inspect_grid(view['points'],scene,
        camera_position=view['T'][:3,3],camera_right=view['T'][:2,0]),'camera':camera}
        for camera,view in views.items()]
    observed = [p for p in proposals if p['status']=='observed']
    result = max(observed,key=lambda p:sum(c['floor_points'] for c in p['cells'])) if observed else proposals[0]
    return {**result,'view_diagnostics':[{'camera':p['camera'],'status':p['status'],
        'reason':p.get('reason'),'rows':p.get('rows'),'columns':p.get('columns')} for p in proposals]}


def cell_occupancy(scene, cell, basis_xy):
    observed=np.asarray(scene,dtype=float)
    centre=np.asarray(cell['position_m']);half=np.asarray(cell['interior_size_m'])/2-.004
    local=(observed[:,:2]-centre[:2])@np.asarray(basis_xy)
    inside=(np.abs(local)<half).all(1)
    raised=inside & (observed[:,2]>centre[2]+.012)&(observed[:,2]<centre[2]+.15)
    floor_points=inside & (np.abs(observed[:,2]-centre[2])<.004)
    centre_floor=floor_points & (np.abs(local)<half*.5).all(1)
    coverage=np.floor((local[floor_points]+half)/(2*half)*3).astype(int)
    patches=len(np.unique(coverage,axis=0)) if len(coverage) else 0
    state='occupied' if np.count_nonzero(raised)>=8 else 'empty' if np.count_nonzero(centre_floor)>=4 and patches>=5 else 'unknown'
    return {'occupancy':state,'floor_points':int(np.count_nonzero(floor_points)),
            'raised_points':int(np.count_nonzero(raised))}


def retained_grid(previous, previous_cloud, current_cloud, scene):
    """Reuse an observed grid only while current depth confirms its unchanged pose.

    This is geometry memory, not an assumption that a named/configured tray is
    stationary. Moved or poorly observed containers need a new fit/view.
    """
    old=np.asarray(previous_cloud);new=np.asarray(current_cloud)
    if previous.get('status')!='observed' or min(len(old),len(new))<40:return None
    current_match=float(np.mean(cKDTree(old).query(new)[0]<.0025))
    previous_coverage=float(np.mean(cKDTree(new).query(old)[0]<.0025))
    if current_match<.8 or previous_coverage<.25:return None
    return {**previous,'source':'prior_rgbd_grid_with_current_geometry_confirmation',
        'pose_confirmation':{'current_match':current_match,'previous_coverage':previous_coverage,'tolerance_m':.0025},
        'cells':[{**cell,**cell_occupancy(scene,cell,previous['basis_xy'])} for cell in previous['cells']]}


def principal_axis(points, view=None):
    points = np.asarray(points, dtype=float)
    centre = np.quantile(points, [.02,.98], axis=0).mean(0)
    eigenvalues, vectors = np.linalg.eigh(np.cov((points-centre).T))
    axis = vectors[:, -1]
    method = 'point_covariance'
    # A visible end cap and one side of a shaft bias point covariance toward
    # the camera. Local side-surface normals constrain the shaft direction
    # without assuming it is vertical or reading a configured asset shape.
    if len(points) >= 48 and eigenvalues[-1] > 1.5*eigenvalues[-2]:
        sample = points[np.linspace(0,len(points)-1,min(len(points),2400),dtype=int)]
        neighbours = cKDTree(points).query(sample,k=16)[1]
        local = points[neighbours]-points[neighbours].mean(1,keepdims=True)
        values, bases = np.linalg.eigh(np.einsum('nki,nkj->nij',local,local))
        normals = bases[:,:,0]
        sides = (np.abs(normals@axis)<.6)&(values[:,0]<.15*values[:,1])
        if sides.sum() >= 32:
            spread, directions = np.linalg.eigh(normals[sides].T@normals[sides])
            refined = directions[:,0]
            span = np.ptp(sample[sides]@axis)
            # One visible flat face cannot determine a long axis. Retain PCA
            # when normal directions or length coverage do not constrain it.
            if (spread[1]>.04*spread[2] and spread[0]<.08*spread[1]
                    and span>.5*np.ptp(sample@axis) and abs(refined@axis)>.85):
                axis = refined
                method = 'side_surface_normals'
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    projected = (points-centre) @ axis
    ends = centre + np.quantile(projected, [.02,.98])[:, None]*axis
    result = {'axis_world': axis.tolist(), 'endpoints_m': ends.tolist(),
              'axis_method': method,
              'axis_confidence': float(eigenvalues[-1]/max(eigenvalues[-2], 1e-9))}
    if view:
        T,K = np.asarray(view['T']),np.asarray(view['K'])
        camera = (ends-T[:3,3]) @ T[:3,:3]
        pixels = camera @ K.T
        result['endpoints_normalized'] = (pixels[:,:2]/pixels[:,2:]/np.asarray(view['size'])).tolist()
    return result


def regular_dividers(positions, strength):
    """Reject weak handle/occluder peaks using the observed line spacing.

    Every returned divider must have a measured peak. Never fill a missing
    divider or discard a strong irregular wall to manufacture a regular grid.
    """
    positions=np.asarray(positions);strength=np.asarray(strength)
    for count in range(len(positions),1,-1):
        expected=np.linspace(positions[0],positions[-1],count)
        spacing=(positions[-1]-positions[0])/(count-1)
        if spacing<.012:continue
        nearest=np.abs(expected[:,None]-positions[None,:]).argmin(1)
        if len(set(nearest)) != count or np.max(np.abs(expected-positions[nearest])) > max(.0025,spacing*.12):continue
        unused=np.ones(len(positions),bool);unused[nearest]=False
        if unused.any() and np.max(strength[unused]) > .55*np.median(strength[nearest]):continue
        return positions[nearest]
    return None


def inspect_grid(parent, scene=None, *, camera_position=None, camera_right=None, resolution=.002):
    """Infer a rectangular grid from observed floor and parallel divider peaks.

    An uncertain grid returns no cells; the caller can inspect another view or
    select an individual region instead. Empty space needs positive floor evidence.
    """
    parent = np.asarray(parent, dtype=float)
    parent = parent[np.isfinite(parent).all(1)]
    if len(parent) < 40:
        return {'status':'unknown','reason':'insufficient_depth','cells':[]}
    levels,counts = np.unique(np.round(parent[:,2]/resolution).astype(int), return_counts=True)
    floor = float(levels[np.argmax(counts)]*resolution)
    flat = parent[np.abs(parent[:,2]-floor) < resolution*1.5]
    walls = parent[(parent[:,2] > floor+.012) & (parent[:,2] < floor+.06)]
    if len(walls) < 40:
        return {'status':'unknown','reason':'no_observed_dividers','cells':[]}
    centre = np.median(flat[:,:2],axis=0)
    _,vectors = np.linalg.eigh(np.cov(flat[:,:2].T))
    initial = np.arctan2(vectors[1,-1],vectors[0,-1])
    best = None
    for angle in initial+np.deg2rad(np.arange(-15,15.1,.5)):
        c,s = np.cos(angle),np.sin(angle)
        axes = np.array([[c,-s],[s,c]])
        p,w = (flat[:,:2]-centre)@axes,(walls[:,:2]-centre)@axes
        low,high = np.quantile(p,[.005,.995],axis=0)
        # Handles outside the observed floor are not divider lines.
        w = w[((w >= low-.008)&(w <= high+.008)).all(1)]
        histograms,score = [],0.
        for axis in range(2):
            edges = np.arange(low[axis]-.008,high[axis]+.010,resolution)
            hist,_ = np.histogram(w[:,axis],edges)
            hist = gaussian_filter1d(hist.astype(float),.55)
            score += float(np.sum(hist**2)/max(np.sum(hist)**2,1.))
            histograms.append((hist,edges))
        if best is None or score > best[0]:
            best = score,axes,histograms
    _,axes,histograms = best
    lines=[]
    for hist,edges in histograms:
        peaks,_ = find_peaks(hist,prominence=max(hist)*.18,distance=max(2,int(.008/resolution)))
        centres = (edges[peaks]+edges[peaks+1])/2
        if len(centres) < 2:
            return {'status':'unknown','reason':'incomplete_grid_boundaries','cells':[]}
        centres = regular_dividers(centres,hist[peaks])
        if centres is None:
            return {'status':'unknown','reason':'irregular_or_occluded_dividers','cells':[]}
        lines.append(centres)
    # Column order points right in the selected camera; rows start near it.
    right=np.asarray(camera_right if camera_right is not None else [1.,0.])[:2]
    if axes[:,0]@right<0:
        axes[:,0]*=-1;lines[0]=-lines[0][::-1]
    if camera_position is not None and axes[:,1]@(np.asarray(camera_position)[:2]-centre)>0:
        axes[:,1]*=-1;lines[1]=-lines[1][::-1]
    cells=[]
    observed = parent if scene is None else np.asarray(scene,dtype=float)
    for r,(v0,v1) in enumerate(zip(lines[1][:-1],lines[1][1:]),1):
        for c,(u0,u1) in enumerate(zip(lines[0][:-1],lines[0][1:]),1):
            uv=np.array([(u0+u1)/2,(v0+v1)/2])
            cell={'row':r,'column':c,'position_m':np.r_[centre+axes@uv,floor].tolist(),
                  'interior_size_m':[float(u1-u0-.004),float(v1-v0-.004)]}
            cells.append({**cell,**cell_occupancy(observed,cell,axes)})
    return {'status':'observed','rows':len(lines[1])-1,'columns':len(lines[0])-1,'capacity':len(cells),
            'cells':cells,'floor_height_m':floor,'basis_xy':axes.tolist(),
            'row_order':'near_to_far_in_reference_camera','column_order':'left_to_right_in_reference_camera',
            'source':'rgbd_floor_and_divider_lines'}
