import sys
from pathlib import Path

import omni.ext
import omni.ui as ui
import omni.usd

_REPO = Path(__file__).resolve().parents[3]
_SOURCE = _REPO / "source"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from mr_liu.sim.spawn import apply_environment_map, mount_so101_to_table, spawn_table_and_so101


class MrLiuProjectExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._window = ui.Window("MR Liu Project", width=380, height=280)
        scene_path = _REPO / "scenes" / "world.usda"

        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("MR Liu + YOLOE", height=24)
                ui.Label("SO-101 + cuMotion follow-target. Vision: scripts\\run_vision_follow.bat", word_wrap=True, height=40)

                def load_scene() -> None:
                    if not scene_path.is_file():
                        print(f"[mr_liu] Scene not found: {scene_path}")
                        return
                    omni.usd.get_context().open_stage(str(scene_path))
                    print(f"[mr_liu] Opened {scene_path}")

                def _run(fn, label: str) -> None:
                    try:
                        fn()
                    except Exception as exc:
                        print(f"[mr_liu] {label} failed: {exc}")

                ui.Button("Load Default Scene", height=32, clicked_fn=load_scene)
                ui.Button("Place Table + SO-101", height=32, clicked_fn=lambda: _run(spawn_table_and_so101, "spawn"))
                ui.Button("Apply Environment Map", height=32, clicked_fn=lambda: _run(apply_environment_map, "HDRI"))
                ui.Button("Mount SO-101 to Table", height=32, clicked_fn=lambda: _run(mount_so101_to_table, "mount"))

        print(f"[mr_liu] Extension started ({ext_id})")
        print("[mr_liu] Follow-target: scripts\\run_follow_target.bat")
        print("[mr_liu] Vision follow: scripts\\run_vision_follow.bat")

    def on_shutdown(self) -> None:
        if self._window:
            self._window.destroy()
            self._window = None
        print("[mr_liu] Extension shutdown")
