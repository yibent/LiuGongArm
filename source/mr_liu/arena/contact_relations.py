"""Observed-geometry contact placement built on AnyPlace full-pose proposals.

AnyPlace selects the payload orientation.  The local RGB-D geometry snaps that
proposal to the opening, peg or hook and supplies the approach axis; no scene
asset name or configured destination coordinate is used for control.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_components

from mr_liu.arena.contracts import placement_to_tcp
from mr_liu.grasp.transforms import assert_transform, transform_points


CONTACT_RELATIONS = {"insert", "sleeve_on_peg", "hang"}


def _bounds_centre(points):
    low, high = np.quantile(points, [.02, .98], axis=0)
    return (low + high) / 2


def _dominant_support_height(points, resolution=.003):
    """Return the densest visible horizontal level in the lower 60%."""
    z = np.asarray(points)[:, 2]
    cutoff = np.quantile(z, .6)
    lower = z[z <= cutoff]
    origin = lower.min()
    bins = np.floor((lower - origin) / resolution).astype(int)
    return float(origin + (np.bincount(bins).argmax() + .5) * resolution)


def _xy_components(points, resolution=.008):
    low = points[:, :2].min(0) - resolution
    ij = np.floor((points[:, :2] - low) / resolution).astype(int)
    shape = tuple(ij.max(0) + 2)
    grid = np.zeros(shape, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    labels, count = connected_components(grid, structure=np.ones((3, 3), dtype=bool))
    rows = []
    for index in range(1, count + 1):
        selected = labels[ij[:, 0], ij[:, 1]] == index
        if selected.sum() < 6:
            continue
        cloud = points[selected]
        rows.append({"centre": _bounds_centre(cloud), "points": int(len(cloud))})
    return rows


def fixture_feature(parent, relation, preferred):
    """Find a contact feature using only the current destination point cloud."""
    points = np.asarray(parent, dtype=float)
    if relation not in CONTACT_RELATIONS or points.ndim != 2 or points.shape[1] != 3 or len(points) < 32:
        raise ValueError("A supported relation and observed destination cloud are required")
    base_z = _dominant_support_height(points)
    elevated = points[points[:, 2] > base_z + .016]
    if len(elevated) < 20:
        raise RuntimeError("目标区域没有观测到可用于接触放置的孔、销或挂钩。")
    top_z = float(np.quantile(elevated[:, 2], .98))
    if relation == "insert":
        # Socket walls form one connected elevated component around the void.
        centre = _bounds_centre(elevated)
        return {"centre": centre.tolist(), "axis": [0., 0., 1.],
                "base_z": base_z, "top_z": top_z, "source": "observed_socket_walls"}
    if relation == "sleeve_on_peg":
        components = _xy_components(elevated)
        if not components:
            raise RuntimeError("目标区域没有分离出可套入的定位销。")
        preferred = np.asarray(preferred, dtype=float)
        selected = min(components, key=lambda row: np.linalg.norm(np.asarray(row["centre"])[:2] - preferred[:2]))
        return {"centre": np.asarray(selected["centre"]).tolist(), "axis": [0., 0., 1.],
                "base_z": base_z, "top_z": top_z, "source": "observed_vertical_peg",
                "feature_count": len(components)}
    # The hook is the highest horizontal feature.  PCA supplies its insertion
    # axis and is invariant to the fixture's world yaw.
    high = elevated[elevated[:, 2] >= np.quantile(elevated[:, 2], .82)]
    centre = _bounds_centre(high)
    covariance = np.cov(high - high.mean(0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    horizontal = [vectors[:, i] for i in np.argsort(values)[::-1] if abs(vectors[2, i]) < .45]
    axis = horizontal[0] if horizontal else np.array([0., 1., 0.])
    axis[2] = 0.; axis /= np.linalg.norm(axis)
    return {"centre": centre.tolist(), "axis": axis.tolist(),
            "base_z": base_z, "top_z": top_z, "source": "observed_hook_axis"}


def contact_pose_candidates(transforms, child, parent, object_input, tcp_to_object,
                            relation, preferred):
    """Snap AnyPlace orientations to an observed contact feature and rank them."""
    child = np.asarray(child, dtype=float)
    feature = fixture_feature(parent, relation, preferred)
    target = np.asarray(feature["centre"], dtype=float)
    feature_axis = np.asarray(feature["axis"], dtype=float)
    rows = []
    for index, value in enumerate(transforms):
        relative = assert_transform(np.asarray(value, dtype=float)).copy()
        placed = transform_points(relative, child)
        covariance = np.cov(placed - placed.mean(0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        if relation in {"insert", "sleeve_on_peg"}:
            payload_axis = vectors[:, np.argmax(values)]
            orientation_cost = 1. - abs(float(payload_axis @ feature_axis))
            # A sleeve must be engaged while the fingers still clear the
            # fixture.  Releasing after partial insertion lets the observed peg
            # guide gravity settling; commanding the payload all the way to the
            # base would require the Panda fingers to pass through the plate.
            target_bottom = (feature["top_z"] - .018 if relation == "sleeve_on_peg"
                             else feature["base_z"] + .002)
            shift = np.r_[target[:2] - _bounds_centre(placed)[:2],
                          target_bottom - np.quantile(placed[:, 2], .02)]
        else:
            # A ring/handle's thinnest PCA axis is normal to its opening plane;
            # align it with the hook so the contact motion passes through it.
            opening_axis = vectors[:, np.argmin(values)]
            orientation_cost = 1. - abs(float(opening_axis @ feature_axis))
            shift = target - _bounds_centre(placed)
        relative[:3, 3] += shift
        snapped = transform_points(relative, child)
        pose = placement_to_tcp(relative, object_input, tcp_to_object)
        rows.append({"pose": pose, "placed": snapped, "feature": feature,
                     "score": 3. * orientation_cost + .01 * index,
                     "proposal_index": index,
                     "engagement_depth_m": (.018 if relation == "sleeve_on_peg" else None)})
    return sorted(rows, key=lambda row: row["score"])
