"""Observed support/free-space checks. Missing depth NEVER means free space."""
from __future__ import annotations
from itertools import product
import cv2
import numpy as np
from scipy.spatial import cKDTree
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import transform_points
from .contracts import PlaceError, DestinationEvidence


def scene_cloud(obs):
    valid = np.isfinite(obs.depth_m) & (obs.depth_m > .02) & (obs.depth_m < 3.)
    robot = obs.metadata.get("robot_self_mask")
    if robot is not None:
        if np.asarray(robot).shape != valid.shape:
            raise PlaceError("invalid_robot_mask")
        valid &= ~np.asarray(robot, bool)
    rows, cols = np.nonzero(valid)
    points = transform_points(obs.T_base_camera, deproject_depth(obs.depth_m, obs.intrinsics, valid))
    return points, rows, cols


def polygon_mask(item, shape):
    """Use Florence polygon evidence, not an entire bounding box as free space."""
    block = item.get("segmentation", {}).get("<REGION_TO_SEGMENTATION>", {})
    mask = np.zeros(shape, np.uint8)
    def polygons(value):
        if not isinstance(value,(list,tuple,np.ndarray)) or len(value)==0:
            return
        if np.isscalar(value[0]):
            yield np.asarray(value,float)
        else:
            for child in value:
                yield from polygons(child)
    for polygon in polygons(block.get("polygons", [])):
        if len(polygon) >= 6 and len(polygon) % 2 == 0 and np.isfinite(polygon).all():
            cv2.fillPoly(mask, [np.rint(polygon.reshape(-1, 2)).astype(np.int32)], 1)
    box = np.asarray(item.get("xyxy"), float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise PlaceError("invalid_destination_box")
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1,x2=np.clip([x1,x2],0,shape[1])
    y1,y2=np.clip([y1,y2],0,shape[0])
    if x2<=x1 or y2<=y1:
        raise PlaceError("invalid_destination_box")
    roi = np.zeros(shape, np.uint8)
    roi[max(0,y1):min(shape[0],y2), max(0,x1):min(shape[1],x2)] = 1
    mask &= roi
    if mask.sum() < 80:
        raise PlaceError("destination_segmentation_uncertain")
    return mask.astype(bool)


def support_evidence(obs, mask, *, relation="on", offset=(0., 0.)):
    points, rows, cols = scene_cloud(obs)
    local = points[mask[rows, cols]]
    if len(local) < 80:
        raise PlaceError("insufficient_destination_depth")
    anchor = np.median(local, axis=0)
    if relation == "relative":
        center_xy = anchor[:2] + offset
        region = np.linalg.norm(points[:,:2] - center_xy, axis=1) < .09
        local = points[region]
    else:
        center_xy = anchor[:2]
    if len(local) < 80:
        raise PlaceError("insufficient_support_depth")
    # Horizontal support only. Enumerate measured height modes; never invent a
    # floor underneath an occluded object or fill a depth hole with a plane.
    bins = np.rint(local[:,2] / .004).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    candidates = sorted(zip(counts, values), reverse=True)
    surface = None
    for count, level in candidates:
        if count < 50:
            continue
        z = np.median(local[np.abs(local[:,2] - level*.004) < .004, 2])
        plane = local[np.abs(local[:,2] - z) < .003]
        if len(plane) < 80 or np.any(np.ptp(plane[:,:2], axis=0) < .035):
            continue
        if relation == "inside":
            # Require observed elevated boundary in all four sectors. This V1
            # supports open trays, not arbitrary hidden/transparent containers.
            middle = np.median(plane[:,:2], axis=0)
            rim = local[local[:,2] > z + .012]
            if len(rim) < 40:
                continue
            delta = rim[:,:2] - middle
            if any(np.sum(delta[:,axis]*sign > .025) < 8 for axis in (0,1) for sign in (-1,1)):
                continue
        surface = plane
        break
    if surface is None:
        raise PlaceError("support_or_container_uncertain")
    z = float(np.median(surface[:,2]))
    if relation != "relative":
        center_xy = np.median(surface[:,:2], axis=0)
    return DestinationEvidence(obs, surface, points, np.r_[center_xy,z], z, mask)


def box_corners(half):
    return np.asarray(list(product((-1.,1.), repeat=3))) * half


def footprint_supported(evidence, xy, radius_xy, margin=.010):
    """Require dense measured support beneath the whole padded object footprint."""
    extent = np.asarray(radius_xy) + margin
    axes = [np.linspace(-v, v, max(3, int(np.ceil(2*v/.004))+1)) for v in extent]
    grid = np.asarray(list(product(*axes))) + xy
    distances, _ = cKDTree(evidence.surface_points[:,:2]).query(grid)
    return bool(np.all(distances < .006))


def select_site(evidence, half_extents, margin=.010):
    # A set of measured alternatives, not one hard-coded pixel/scene coordinate.
    center = evidence.center_base_m[:2]
    offsets = [np.array([0.,0.])] + [np.array(v)*.004 for v in product(range(-8,9),repeat=2) if v != (0,0)]
    offsets.sort(key=np.linalg.norm)
    for offset in offsets:
        xy = center + offset
        if footprint_supported(evidence, xy, half_extents[:2], margin):
            return xy
    raise PlaceError("no_supported_placement_region")
