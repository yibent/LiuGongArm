"""Separate simulation integration frequency from the desktop render loop."""
from __future__ import annotations

import math


def render_interval(physics_dt: float, render_hz: float) -> float:
    if not math.isfinite(render_hz) or render_hz <= 0 or physics_dt <= 0:
        raise ValueError("Physics step and render frequency must be positive")
    dt = 1.0 / render_hz
    ratio = dt / physics_dt
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-6):
        raise ValueError("Rendering interval must contain an integer number of physics steps")
    return dt


def configure_render_timing(physics_dt: float, render_hz: float) -> float:
    import carb
    import omni.timeline
    import omni.usd
    from omni.kit.loop import _loop

    dt = render_interval(physics_dt, render_hz)
    timeline = omni.timeline.get_timeline_interface()
    omni.usd.get_context().get_stage().SetTimeCodesPerSecond(render_hz)
    timeline.set_time_codes_per_second(render_hz)
    settings = carb.settings.get_settings()
    settings.set("/persistent/simulation/minFrameRate", render_hz)
    if settings.get("/app/runLoops/main/rateLimitEnabled"):
        settings.set("/app/runLoops/main/rateLimitFrequency", render_hz)
        timeline.set_target_framerate(render_hz)
    loop = _loop.acquire_loop_interface()
    loop.set_manual_step_size(dt)
    loop.set_manual_mode(True)
    print(f"[mr_liu] Timing: physics={1/physics_dt:g} Hz render={render_hz:g} Hz "
          f"substeps={round(dt/physics_dt)}", flush=True)
    return dt
