"""Choose a horizontal support patch using observed RGB-D geometry only."""
import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt


def choose_free_support(support, scene, child, preferred, *, resolution=.006):
    support, scene, child = (np.asarray(p, dtype=float) for p in (support, scene, child))
    support = support[np.isfinite(support).all(axis=1)]
    scene = scene[np.isfinite(scene).all(axis=1)]
    if len(support) < 3 or len(child) < 3:
        raise RuntimeError('尚无足够的实测几何来选择空闲放置位置。')
    # The largest horizontal depth band is the support surface; rims and
    # objects above it must not raise the estimated table height.
    bands = np.round(support[:, 2] / .004).astype(int)
    levels, counts = np.unique(bands, return_counts=True)
    height = float(np.median(support[bands == levels[counts.argmax()], 2]))
    plane = support[np.abs(support[:, 2] - height) < .008]
    low, high = np.quantile(child, [.02, .98], axis=0)
    # Account for the current object's footprint and Panda fingers.
    radius = max(.05, float(np.linalg.norm(high[:2] - low[:2]) / 2 + .015))
    origin = plane[:, :2].min(axis=0) - resolution * 3
    shape = np.ceil((plane[:, :2].max(axis=0) - origin) / resolution).astype(int) + 4
    observed = np.zeros(tuple(shape), dtype=bool)
    indices = np.floor((plane[:, :2] - origin) / resolution).astype(int)
    observed[indices[:, 0], indices[:, 1]] = True
    # Close only camera sampling holes, not unseen areas or large occlusions.
    observed = binary_closing(observed, iterations=2)
    occupied = np.zeros_like(observed)
    obstacle = scene[(scene[:, 2] > height + .009) &
                     (scene[:, 2] < height + max(.16, high[2] - low[2] + .10))]
    indices = np.floor((obstacle[:, :2] - origin) / resolution).astype(int)
    indices = indices[((indices >= 0) & (indices < shape)).all(axis=1)]
    occupied[indices[:, 0], indices[:, 1]] = True
    clearance = distance_transform_edt(observed & ~occupied) * resolution
    candidates = np.argwhere(clearance >= radius + resolution)
    if not len(candidates):
        raise RuntimeError('当前可见支撑面没有容纳持物和夹爪的空位，请腾出位置或换个视角。')
    xy = origin + (candidates + .5) * resolution
    index = np.argmin(np.linalg.norm(xy - np.asarray(preferred)[:2], axis=1))
    centre = np.r_[xy[index], height]
    return centre, {'position_world_m': centre.tolist(), 'footprint_radius_m': radius,
                    'clearance_m': float(clearance[tuple(candidates[index])]),
                    'support_points': len(plane), 'obstacle_points': len(obstacle),
                    'candidate_count': len(candidates), 'source': 'model_support_mask_and_scene_rgbd'}
