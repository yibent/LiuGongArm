"""Measured support relation; a supported object may overhang its base."""
import numpy as np


def support_metrics(position, support_position, force_world, support_velocity):
    position, support_position = np.asarray(position), np.asarray(support_position)
    force_world = np.asarray(force_world)
    contact = bool(np.linalg.norm(force_world) > 0.)
    # An upward contact from the requested support distinguishes a stack from
    # a nearby object on the table. No footprint-containment requirement.
    supported = bool(contact and force_world[2] > 0. and position[2] > support_position[2])
    return {'destination_xy_error_m': float(np.linalg.norm(position[:2] - support_position[:2])),
            'destination_position_world_m': support_position.tolist(),
            'support_contact': contact, 'support_force_world_n': force_world.tolist(),
            'supported_region': supported,
            'support_linear_speed_mps': float(np.linalg.norm(support_velocity))}
