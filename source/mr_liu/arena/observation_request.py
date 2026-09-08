"""Read-only perception: capture on the sim thread, inference on an HTTP worker."""


def capture_observation(runtime, body):
    params = body.get('params', {})
    if not isinstance(params, dict):
        raise ValueError('Observation params must be an object')
    if params.get('tracking'):
        raise ValueError('Persistent tracking uses the perceive skill')
    scene = params.get('scope') == 'scene'
    label, ref = (None, None) if scene else runtime.target_value(
        params if params.get('ref') else params.get('category'))
    if not scene and not label:
        raise ValueError('Specify a category, ref or scope=scene')
    color = params.get('attributes', {}).get('color', '')
    if label and color and color not in label:
        label = color + ' ' + label
    packet = runtime.perception.capture(
        runtime, label, cameras=params.get('cameras'),
        vision_mode=params.get('vision_mode', 'auto'),
        slow_provider=params.get('slow_provider'),
        scene_mode=params.get('scene_mode', 'inventory'),
        visual_ref=ref, grounding=params.get('grounding'), inspect=params.get('inspect'),
        collection=not scene and not ref and not params.get('grounding') and params.get('selection', 'all') != 'one',
    )
    # Do not attribute a read-only query to the physical action in flight.
    packet['command_id'] = body['command_id']
    packet['correlation_id'] = body.get('correlation_id')
    packet.pop('task_id', None)
    packet.pop('task_version', None)
    import json
    (runtime.perception.root / packet['request_id'] / 'request.json').write_text(json.dumps(packet))
    return packet


def observation_result(runtime, packet):
    observed = runtime.perception.request(packet)
    result = {
        'ok': bool(observed.get('ok')), 'state': 'completed' if observed.get('ok') else 'failed',
        'command_id': packet['command_id'], 'vision': observed,
        'message': 'Read-only observation completed' if observed.get('ok') else str(observed.get('error', 'Observation failed')),
        'scope': observed.get('scope'), 'request_id': packet['request_id'],
    }
    for key in ('collection', 'geometry', 'references', 'semantic_status'):
        if key in observed:
            result[key] = observed[key]
    # Keep geometry/identity available to later manipulation without touching
    # the current command's target, phase, visual_result or holding state.
    if observed.get('ok') and packet.get('scope') == 'target':
        points, _ = runtime.perception.cloud(observed)
        runtime.perception.remember_target(observed, points, packet.get('visual_ref'))
    return result
