"""Measured holding continuity across commands; independent of task wording."""
import numpy as np
from scipy.spatial.transform import Rotation


def holding_measurement(reference_tcp_to_object, tcp_to_object, opening, reference_opening=None):
    drift = float(np.linalg.norm(np.asarray(tcp_to_object)[:3, 3] -
                                 np.asarray(reference_tcp_to_object)[:3, 3]))
    angular_drift = float(np.rad2deg(Rotation.from_matrix(
        np.asarray(tcp_to_object)[:3,:3] @ np.asarray(reference_tcp_to_object)[:3,:3].T).magnitude()))
    # Closed empty fingers are not evidence of a grasp. Retain the measured
    # thickness at closure so losing a small part near the TCP is observable.
    minimum = max(.001, .5 * reference_opening) if reference_opening is not None else .001
    jaw_retained = minimum < opening < .075
    return {'verified': bool(jaw_retained and drift < .035),
            'jaw_retained': bool(jaw_retained),
            'reference_opening_m': reference_opening,
            'relative_rotation_drift_deg': angular_drift,
            'gripper_opening_m': float(opening), 'relative_translation_drift_m': drift}


class HoldMonitor:
    """Debounce measured loss only while transporting with closed fingers."""
    def __init__(self):
        self.losses = 0

    def update(self, verified, phase, gripper_closed):
        watching = gripper_closed and phase in {'lift', 'transport', 'reorient_transport', 'place_approach'}
        self.losses = self.losses + 1 if watching and not verified else 0
        return self.losses >= 3
