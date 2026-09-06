"""Associate a model mask with a physical witness, without semantic asset names.

Renderer IDs never supply masks, labels or geometry to perception/planning.
They identify the simulator body whose motion/contact the task will measure.
"""
from collections import Counter
import numpy as np


class InstanceConflict(RuntimeError):
    pass


def consistent_witness(view_votes):
    witnesses = {physical_witness(votes) for votes in view_votes.values() if votes}
    if len(witnesses) > 1:
        raise InstanceConflict('不同视角定位到了不同实例，需要重新识别，不能合并这些点云。')
    return physical_witness(sum(view_votes.values(), Counter()))


def instance_votes(mask, instance_image, labels, bodies):
    counts = Counter()
    values, pixels = np.unique(np.asarray(instance_image).squeeze()[mask], return_counts=True)
    for value, count in zip(values, pixels):
        path = labels.get(str(int(value)), '')
        if isinstance(path, dict):
            path = path.get('primPath', path.get('prim_path', ''))
        if not isinstance(path, str):
            continue
        matches = [(len(body['prim_path']), name) for name, body in bodies.items()
                   if path == body['prim_path'] or path.startswith(body['prim_path'] + '/')]
        if matches:
            counts[max(matches)[1]] += int(count)
    return counts


def physical_witness(votes):
    ranked = votes.most_common()
    if not ranked:
        raise RuntimeError('已取得视觉目标，但没有对应的仿真刚体可供物理评测。')
    # A mixed mask must not silently select a different object on another view.
    if len(ranked) > 1 and ranked[0][1] <= sum(votes.values()) / 2:
        raise InstanceConflict('视觉区域同时覆盖多个物理实例，尚不能确定目标实例。')
    return ranked[0][0]
