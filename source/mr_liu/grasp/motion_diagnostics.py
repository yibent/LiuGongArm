"""Read-only error decomposition. All input positions share the world frame."""
import numpy as np


def endpoint_error_chain(requested, planned, measured_fk, measured_usd):
    requested, planned, measured_fk, measured_usd = (
        np.asarray(value, dtype=np.float64).reshape(3)
        for value in (requested, planned, measured_fk, measured_usd)
    )
    vectors = {
        "ik_residual": planned - requested,
        "joint_tracking": measured_fk - planned,
        "fk_usd_discrepancy": measured_usd - measured_fk,
        "total": measured_usd - requested,
    }
    # Vectors add; scalar lengths generally do not (errors can cancel).
    return {name: {"vector_mm": (value * 1000).tolist(),
                   "norm_mm": float(np.linalg.norm(value) * 1000)}
            for name, value in vectors.items()}
