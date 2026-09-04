"""Standalone Isaac Sim 6.0 scene: lab table with an SO-101 arm on top."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.simulation_manager import SimulationManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import scene_config
from mr_liu.sim.spawn import spawn_table_and_so101

stage_utils.create_new_stage()
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
DistantLight("/World/DistantLight").set_intensities(float(scene_config()["distant_intensity"]))
spawn_table_and_so101()

SimulationManager.set_physics_dt(1.0 / 60.0)
app_utils.play()

print("[mr_liu] Hello World scene is running. Close the window to exit.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
