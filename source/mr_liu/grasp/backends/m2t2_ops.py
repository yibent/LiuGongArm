"""Inference-only portable PointNet++ operators for M2T2 on Windows.

This is real sampling/grouping/interpolation, NOT a grasp-model fallback.
The published CUDA implementation remains available as ``native``. FPS keeps
its origin exclusion; CPU/GPU tie-breaking may differ. No training supported.
"""
import sys
import types
import numpy as np
import torch
from scipy.spatial import cKDTree


def furthest_point_sampling(xyz, count):
    result = []
    for points in xyz.detach().cpu().numpy():
        distances = np.full(len(points), 1e10, np.float32)
        valid = np.sum(points * points, axis=1) > 1e-3
        indices = np.zeros(count, np.int64)
        old = 0
        for i in range(1, count):
            delta = points - points[old]
            distances = np.minimum(distances, np.einsum('ij,ij->i', delta, delta))
            old = int(np.argmax(np.where(valid, distances, -1.)))
            indices[i] = old
        result.append(indices)
    return torch.as_tensor(np.stack(result), device=xyz.device, dtype=torch.int32)


def gather_points(features, indices):
    return torch.gather(features, 2, indices.long().unsqueeze(1).expand(-1, features.shape[1], -1))


def group_points(features, indices):
    flat = gather_points(features, indices.reshape(indices.shape[0], -1))
    return flat.reshape(features.shape[0], features.shape[1], *indices.shape[1:])


def ball_query(new_xyz, xyz, radius, nsample):
    out = []
    for points, centers in zip(xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy()):
        neighbors = cKDTree(points).query_ball_point(centers, radius, return_sorted=True)
        indices = np.zeros((len(centers), nsample), np.int32)
        for i, found in enumerate(neighbors):
            if found:
                found = np.asarray(found, np.int64)
                # Native kernel uses a strict radius test and input-index order.
                delta = points[found] - centers[i]
                found = found[np.sum(delta * delta, axis=1) < radius * radius][:nsample]
                if len(found):
                    indices[i, :] = found[0]
                    indices[i, :len(found)] = found
        out.append(indices)
    return torch.as_tensor(np.stack(out), device=xyz.device)


def three_nn(unknown, known):
    distances, indices = [], []
    for query, points in zip(unknown.detach().cpu().numpy(), known.detach().cpu().numpy()):
        d, idx = cKDTree(points).query(query, k=min(3, len(points)))
        d, idx = np.reshape(d, (len(query), -1)), np.reshape(idx, (len(query), -1))
        if d.shape[1] < 3:
            d = np.pad(d, ((0, 0), (0, 3-d.shape[1])), mode='edge')
            idx = np.pad(idx, ((0, 0), (0, 3-idx.shape[1])), mode='edge')
        distances.append(d*d)
        indices.append(idx)
    return (torch.as_tensor(np.stack(distances), device=unknown.device, dtype=unknown.dtype),
            torch.as_tensor(np.stack(indices), device=unknown.device, dtype=torch.int32))


def three_interpolate(features, indices, weights):
    return (group_points(features, indices) * weights.unsqueeze(1)).sum(-1)


def install_portable_ops():
    if 'm2t2.pointnet2_utils' in sys.modules:
        raise RuntimeError('Select PointNet++ operators before importing M2T2')
    package = types.ModuleType('pointnet2_ops')
    ext = types.ModuleType('pointnet2_ops._ext')
    for name in ('furthest_point_sampling', 'gather_points', 'group_points',
                 'ball_query', 'three_nn', 'three_interpolate'):
        setattr(ext, name, globals()[name])
    package._ext = ext
    sys.modules['pointnet2_ops'] = package
    sys.modules['pointnet2_ops._ext'] = ext
