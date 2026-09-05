"""Opt-in physical perturbations for simulation tests, never robot drivers.

The injected callback changes only a simulated object's physical position.
No target truth is returned to the grasp controller, verifier or retry policy.
"""
from __future__ import annotations

import math


class OneShotPreCloseShift:
    def __init__(self, shift_x_m: float, apply_shift):
        if not math.isfinite(shift_x_m) or abs(shift_x_m) > 0.05:
            raise ValueError("Test-only target perturbation must be finite and <= 5 cm")
        self.shift_x_m = float(shift_x_m)
        self.apply_shift = apply_shift
        self.applied = False

    def on_event(self, event):
        if self.applied or self.shift_x_m == 0.0 or event.get("phase") != "close":
            return None
        self.applied = True
        self.apply_shift((self.shift_x_m, 0.0, 0.0))
        return {"phase": "test_fault", "event": "target_shift_before_first_close",
                "shift_base_m": [self.shift_x_m, 0.0, 0.0],
                "simulation_only": True}


class OneShotCoarseShift(OneShotPreCloseShift):
    """Move the scene after planning segment two, before its execution guard."""
    def on_event(self, event):
        if (self.applied or self.shift_x_m == 0.0 or event.get("event") != "coarse_move"
                or event.get("iteration") != 1):
            return None
        self.applied = True
        self.apply_shift((self.shift_x_m, 0.0, 0.0))
        return {"phase": "test_fault", "event": "target_shift_during_coarse",
                "shift_base_m": [self.shift_x_m, 0.0, 0.0], "simulation_only": True}
