"""Simulation-only fixture geometry; never destination coordinates for control."""
from dataclasses import dataclass
import math


FIXTURE_RELATIONS = {"region": "on", "tray": "inside", "socket": "insert", "rack": "hang"}


@dataclass(frozen=True)
class FixtureBox:
    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float] = (.05, .12, .85)


def fixture_boxes(kind, *, x=.25, y=-.198, clutter=0):
    if kind not in FIXTURE_RELATIONS:
        raise ValueError("Unknown fixture")
    if not (math.isfinite(x) and math.isfinite(y) and .20 <= x <= .30 and -.24 <= y <= -.15):
        raise ValueError("Fixture outside development workspace")
    if clutter not in (0, 1, 2):
        raise ValueError("Clutter level must be 0, 1, or 2")
    boxes = [FixtureBox("Floor", (x,y,1.051), (.10,.10,.002))]
    if kind in {"tray", "socket"}:
        # A 38 mm square socket admits a 32 mm cylinder with 3 mm radial
        # clearance. Independent walls preserve the hole in collision geometry.
        inner, thickness, height = (.094,.006,.036) if kind == "tray" else (.038,.014,.050)
        for axis in (0, 1):
            for sign in (-1, 1):
                position = [x,y,1.05+height*.5]
                position[axis] += sign*(inner+thickness)*.5
                size = [inner+2*thickness,inner+2*thickness,height]
                size[axis] = thickness
                boxes.append(FixtureBox(f"Wall{axis}{'p' if sign>0 else 'n'}",tuple(position),tuple(size),(.04,.10,.80)))
    if kind == "rack":
        # Forward bar and raised tip form a collision-bearing hook for a mug
        # handle. Physical hanging needs a validated 6D/contact controller.
        boxes.extend([
            FixtureBox("Post",(x,y+.036,1.135),(.014,.014,.17)),
            FixtureBox("Hook",(x,y+.009,1.195),(.010,.068,.010)),
            FixtureBox("HookTip",(x,y-.020,1.204),(.010,.010,.028)),
        ])
    for index, (dx,dy,height) in enumerate(((-.072,.060,.08),(-.065,-.068,.12))[:clutter]):
        boxes.append(FixtureBox(f"Clutter{index}",(x+dx,y+dy,1.05+height*.5),(.018,.018,height),(.75,.55,.10)))
    return boxes
