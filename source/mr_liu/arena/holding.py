"""Measured holding continuity across commands; independent of task wording."""
import numpy as np


def holding_measurement(reference_tcp_to_object, tcp_to_object, opening):
    drift = float(np.linalg.norm(np.asarray(tcp_to_object)[:3, 3] -
                                 np.asarray(reference_tcp_to_object)[:3, 3]))
    return {'verified': bool(opening < .075 and drift < .035),
            'gripper_opening_m': float(opening), 'relative_translation_drift_m': drift}
