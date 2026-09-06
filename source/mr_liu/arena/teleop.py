"""Latest-input mailbox. Only the simulation thread applies motion.

A short renewable lease stops motion after pointer release, a lost connection,
or a closed tab. Updates cannot queue up or cancel a different command.
"""
import math
import threading
import time


def finite_vector(value, count=3):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"Expected {count} numeric coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("Coordinates must be finite numbers")
    return value


class TeleopInput:
    def __init__(self, lease_s=.6, clock=time.monotonic):
        self.lease_s, self.clock = lease_s, clock
        self.lock = threading.Lock()
        self.session = None

    @staticmethod
    def velocities(params):
        return {key: finite_vector(params.get(key, [0., 0., 0.]))
                for key in ("linear", "angular")}

    def start(self, command_id, params):
        values = self.velocities(params)
        with self.lock:
            if self.session is not None:
                raise ValueError("Manual control is already active")
            self.session = {"command_id": command_id, **values,
                            "expires": self.clock() + self.lease_s, "stopped": False}

    def update(self, command_id, params):
        values = None if params.get("stop") else self.velocities(params)
        with self.lock:
            row = self.session
            if row is None or row["command_id"] != command_id:
                return False
            if values is None:
                row["stopped"] = True
                return True
            # An expired/released gesture cannot be revived by a delayed packet.
            if row["stopped"] or row["expires"] <= self.clock():
                return False
            row.update(values, expires=self.clock() + self.lease_s)
            return True

    def read(self, command_id):
        with self.lock:
            row = self.session
            if row is None or row["command_id"] != command_id or row["stopped"] or row["expires"] <= self.clock():
                return None
            return dict(row)

    def finish(self, command_id):
        with self.lock:
            if self.session and self.session["command_id"] == command_id:
                self.session = None
