"""Scene assets have physical properties, not exclusive pick/place roles."""


def scene_entities(config):
    return list({row['name']: row for row in config['objects'] + config['destinations']}.values())


def resolve_entity(config, label, *, graspable=False):
    normalize = lambda value: str(value).strip().lower().replace('_', ' ')
    rows = config['objects'] if graspable else scene_entities(config)
    matches = [row for row in rows if normalize(label) in
               [normalize(value) for value in [row['name'], row['label'], *row.get('aliases', [])]]]
    if len(matches) != 1:
        raise ValueError(f'无法唯一关联场景物体：{label}。需要定位到具体场景实体；这不是视觉未检测到的结论。')
    return matches[0]
