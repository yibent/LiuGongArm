"""Explicit simulation test faults. Never installed in the live service."""
from mr_liu.arena.cascade import FastPathFailure


def inject_once(runtime, fault):
    if fault == "miss_grasp":
        original = runtime.cloud
        fired = False

        def cloud(name, **kwargs):
            nonlocal fired
            points = original(name, **kwargs)
            if not fired and name == runtime.target_name:
                fired = True
                points = points.copy()
                points[:, 1] += .08
                runtime.event("test_fault", fault=fault, offset_world_m=[0., .08, 0.])
            return points
        runtime.cloud = cloud
    else:
        original = runtime.event
        fired = False

        def event(phase, **data):
            nonlocal fired
            original(phase, **data)
            if not fired and phase == "lift_verified":
                fired = True
                original("test_fault", fault=fault)
                raise FastPathFailure("Injected failure after a real successful lift")
        runtime.event = event
