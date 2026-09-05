"""Optional stopped-arm release budget; never authorizes Cartesian movement."""
import numpy as np
from mr_liu.grasp.transforms import assert_transform


def stationary_release_state(before, after, grip_before, grip_after, *, now, max_age_s):
    """Require timestamped measured poses/joints and unchanged gripper aperture.

    This is a bounded static-workcell policy, not a dynamic-obstacle guarantee.
    Missing proprioception fails closed rather than assuming stop() settled.
    """
    try:
        for state in (before, after):
            assert_transform(state.T_base_ee)
            if not state.joint_positions or not np.isfinite(state.timestamp_s):
                return False
        if not -.02 <= now-before.timestamp_s <= 1.5:
            return False
        if not -.02 <= now-after.timestamp_s <= max_age_s:
            return False
        if after.timestamp_s < before.timestamp_s:
            return False
        if before.joint_positions.keys() != after.joint_positions.keys():
            return False
        deltas = np.array([after.joint_positions[k]-v for k,v in before.joint_positions.items()])
        if not np.isfinite(deltas).all() or np.max(np.abs(deltas)) > .003:
            return False
        a,b = before.T_base_ee,after.T_base_ee
        angle = np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1))
        if np.linalg.norm(a[:3,3]-b[:3,3]) > .0005 or angle > np.deg2rad(.25):
            return False
        widths = np.array([grip_before.width_m,grip_after.width_m],float)
        return bool(np.isfinite(widths).all() and np.all(widths > .002)
                    and abs(widths[1]-widths[0]) <= .0005
                    and -.02 <= now-grip_after.timestamp_s <= max_age_s)
    except (AttributeError, TypeError, ValueError):
        return False
