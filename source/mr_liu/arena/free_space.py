"""Rank observed horizontal placements with payload/gripper footprint and yaw."""
import numpy as np
from mr_liu.arena.failure import PlacementSpaceUnavailable
from scipy.ndimage import binary_closing, binary_erosion, distance_transform_edt
from scipy.spatial import ConvexHull


def support_grid(support, scene, child, resolution=.006):
    support, scene, child = (np.asarray(p, dtype=float) for p in (support, scene, child))
    support, scene, child = [p[np.isfinite(p).all(axis=1)] for p in (support, scene, child)]
    if min(len(support), len(child)) < 3:
        raise RuntimeError('尚无足够的实测几何来选择空闲放置位置。')
    bands = np.round(support[:, 2] / .004).astype(int)
    levels, counts = np.unique(bands, return_counts=True)
    height = float(np.median(support[bands == levels[counts.argmax()], 2]))
    plane = support[np.abs(support[:, 2] - height) < .008]
    low, high = np.quantile(child, [.02, .98], axis=0)
    padding = max(.15, np.linalg.norm(high[:2]-low[:2]) / 2 + .05)
    origin = plane[:, :2].min(0) - padding
    shape = np.ceil((plane[:, :2].max(0) + padding - origin) / resolution).astype(int) + 1
    observed = np.zeros(tuple(shape), bool)
    ij = np.floor((plane[:, :2] - origin) / resolution).astype(int)
    observed[ij[:, 0], ij[:, 1]] = True
    observed = binary_closing(observed, iterations=2)
    obstacle = scene[(scene[:, 2] > height + .009) &
                     (scene[:, 2] < height + max(.16, high[2] - low[2] + .10))]
    occupied = np.zeros_like(observed)
    ij = np.floor((obstacle[:, :2] - origin) / resolution).astype(int)
    ij = ij[((ij >= 0) & (ij < shape)).all(1)]
    occupied[ij[:, 0], ij[:, 1]] = True
    visible = np.zeros_like(observed)
    ij = np.floor((scene[:, :2] - origin) / resolution).astype(int)
    ij = ij[((ij >= 0) & (ij < shape)).all(1)]
    visible[ij[:, 0], ij[:, 1]] = True
    visible = binary_closing(visible, iterations=2)
    return observed & ~occupied, visible & ~occupied, origin, height, low, high, len(plane), len(obstacle)


def footprint(low, high, yaw, resolution, tool_offset=None, include_tool=True):
    # Union of measured payload bounds and Panda open-jaw footprint (10 x 6 cm,
    # including clearance). Unlike a circumscribed circle, narrow slots remain usable.
    half = (high[:2] - low[:2]) / 2 + .01
    corners = np.array([[x, y] for x in [-half[0], half[0]] for y in [-half[1], half[1]]])
    tool = np.array([[x, y] for x in [-.03, .03] for y in [-.05, .05]])
    tool += np.asarray(tool_offset if tool_offset is not None else [0., 0.])[:2]
    c, s = np.cos(yaw), np.sin(yaw)
    vertices = (np.r_[corners, tool] if include_tool else corners) @ np.array([[c, s], [-s, c]])
    extent = np.ceil(np.abs(vertices).max(0) / resolution).astype(int) + 1
    x, y = np.meshgrid(np.arange(-extent[0], extent[0]+1), np.arange(-extent[1], extent[1]+1), indexing='ij')
    xy = np.c_[x.ravel(), y.ravel()] * resolution
    hull = ConvexHull(vertices)
    kernel = (xy @ hull.equations[:, :2].T + hull.equations[:, 2] <= resolution/2).all(1).reshape(x.shape)
    return kernel, vertices


def project(points, view):
    points = np.asarray(points)
    if view is None: return points[:, :2]
    T, K = np.asarray(view['T']), np.asarray(view['K'])
    camera = (points-T[:3, 3]) @ T[:3, :3]
    uvw = camera @ K.T
    return uvw[:, :2] / np.maximum(uvw[:, 2:], 1e-6)


def free_support_candidates(support, scene, child, preferred, *, resolution=.006,
                            preference='nearest', region=None, view=None, tool_offset=None,
                            allow_rotation=True, limit=6):
    free, collision_free, origin, height, low, high, plane_count, obstacle_count = support_grid(support, scene, child, resolution)
    clearance = distance_transform_edt(free) * resolution
    candidates = []
    for yaw in ([0., np.pi/2, -np.pi/2, np.pi/4, -np.pi/4] if allow_rotation else [0.]):
        kernel, vertices = footprint(low, high, yaw, resolution, tool_offset)
        payload, _ = footprint(low, high, yaw, resolution, include_tool=False)
        # Only the payload needs supporting. The jaws may extend past a support
        # edge if that space is observed and free of obstacles.
        valid = binary_erosion(free, structure=payload, border_value=0) & binary_erosion(collision_free, structure=kernel, border_value=0)
        ij = np.argwhere(valid)
        if not len(ij): continue
        xyz = np.c_[origin + (ij+.5)*resolution, np.full(len(ij), height)]
        uv = project(xyz, view)
        if region is not None:
            box = np.asarray(region['box'])
            regional_uv = project(xyz, region.get('view', view))
            inside = ((regional_uv >= box[:2]) & (regional_uv <= box[2:])).all(1)
            xyz, uv, ij = xyz[inside], uv[inside], ij[inside]
            if not len(ij): continue
        distance = np.linalg.norm(xyz[:, :2]-np.asarray(preferred)[:2], axis=1)
        scores = distance + .025*abs(yaw)
        normalized = (uv-uv.min(0)) / np.maximum(np.ptp(uv, axis=0), 1e-6)
        if preference in ['left', 'right', 'near', 'far']:
            axis, flip = {'left': (0,False), 'right': (0,True), 'near': (1,True), 'far': (1,False)}[preference]
            priority_weight = np.linalg.norm(np.ptp(xyz[:, :2], axis=0)) + .1
            scores += priority_weight*((1-normalized[:,axis]) if flip else normalized[:,axis])
        elif preference == 'center':
            scores += .4*np.linalg.norm(normalized-.5, axis=1)
        elif preference == 'compact':
            scores += 2*clearance[ij[:,0], ij[:,1]]
        elif preference not in ['nearest', 'any']:
            raise ValueError('Unknown placement preference')
        for i in np.argsort(scores)[:max(100, limit)]:
            candidates.append({'position_world_m': xyz[i].tolist(), 'yaw_delta_rad': float(yaw),
                'score': float(scores[i]), 'clearance_m': float(clearance[tuple(ij[i])]),
                'footprint_radius_m': float(np.linalg.norm(vertices, axis=1).max()),
                'footprint_size_m': np.ptp(vertices, axis=0).tolist(),
                'support_points': plane_count, 'obstacle_points': obstacle_count,
                'source': 'observed_rgbd_footprint', 'preference': preference})
    chosen = []
    for row in sorted(candidates, key=lambda x:x['score']):
        if any(np.linalg.norm(np.asarray(row['position_world_m'])-old['position_world_m']) < .03
               and abs(row['yaw_delta_rad']-old['yaw_delta_rad']) < .1 for old in chosen): continue
        chosen.append(row)
        if len(chosen) >= limit: break
    if not chosen:
        raise PlacementSpaceUnavailable('当前可见支撑面没有容纳持物和夹爪的空位，请换个区域或视角。')
    for row in chosen: row['candidate_count'] = len(candidates)
    return chosen


def choose_free_support(support, scene, child, preferred, **options):
    candidates = free_support_candidates(support, scene, child, preferred, **options)
    selected = candidates[0]
    return np.asarray(selected['position_world_m']), {**selected, 'alternatives': candidates[1:]}


def placement_is_free(support, scene, placed, tool_position, *, resolution=.006):
    free, collision_free, origin, _, low, high, _, _ = support_grid(support, scene, placed, resolution)
    center = (low+high)/2
    kernel, _ = footprint(low, high, 0., resolution, np.asarray(tool_position)[:2]-center[:2])
    payload, _ = footprint(low, high, 0., resolution, include_tool=False)
    valid = binary_erosion(free, structure=payload, border_value=0) & binary_erosion(collision_free, structure=kernel, border_value=0)
    ij = np.floor((center[:2]-origin)/resolution).astype(int)
    return bool((ij >= 0).all() and (ij < valid.shape).all() and valid[tuple(ij)])
