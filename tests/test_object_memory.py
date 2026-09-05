"""Object memory persistence and YOLOE image-prompt behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from find_and_track import Detection, FindTrackPipeline, ObjectMemoryStore, RuntimeConfig


class _FakeFinder:
    ready = True
    loaded_id = "fake"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> None:
        return

    def find(self, _frame, _prompts):
        self.calls += 1
        return []


class _FakeYoloe:
    ready = True

    def __init__(self) -> None:
        self.has_prompt = False
        self.image_prompt_calls = 0

    def load(self) -> None:
        return

    def reset(self) -> None:
        self.has_prompt = False

    def set_image_prompt(self, _image, _label, conf=None) -> None:
        self.image_prompt_calls += 1
        self.has_prompt = True

    def track(self, _frame, conf=None, persist=True):
        return [Detection(np.array([4, 4, 20, 20], dtype=np.float32), "cube", .9, source="fast")]

    def detect(self, _frame, conf=None):
        return self.track(_frame, conf, False)


class ObjectMemoryTests(unittest.TestCase):
    def test_simulation_vision_facade_exports_memory_store(self) -> None:
        from mr_liu.vision import ObjectMemoryStore as exported_store
        self.assertIs(exported_store, ObjectMemoryStore)

    def test_memory_and_grasp_capabilities_coexist_after_merge(self) -> None:
        from mr_liu.vision.control import VisionRuntimeControl, execute_control_command
        control = VisionRuntimeControl(RuntimeConfig())
        self.assertNotIn("remember", control.capabilities()["skills"])
        with tempfile.TemporaryDirectory() as directory:
            control.memory_store = ObjectMemoryStore(directory)
            control.grasp = Mock()
            control.grasp.status.return_value = {"state": "idle"}
            control.grasp.capabilities.return_value = {"backend": "graspgenx"}
            control.grasp.submit.return_value = {"ok": True, "state": "accepted"}
            status = control.status()
            self.assertEqual(status["memories"], [])
            self.assertEqual(status["grasp"], {"state": "idle"})
            self.assertIn("grounding_diagnostics", status)
            for skill in ("remember", "recall", "forget", "grasp"):
                self.assertIn(skill, status["capabilities"]["skills"])
            self.assertNotIn("grasp", status["capabilities"]["unsupported"])
            self.assertTrue(execute_control_command(control, "grasp", {}, "test")["ok"])
            control.grasp.submit.assert_called_once_with({}, "test")

    def test_selecting_new_target_clears_previous_visual_memory(self) -> None:
        from mr_liu.vision.control import VisionRuntimeControl, execute_control_command
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectMemoryStore(directory)
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            memory = store.remember("red cube", {"scene": frame},
                {"scene": Detection(np.array([5, 5, 24, 24], dtype=np.float32), "red cube")})
            control = VisionRuntimeControl(RuntimeConfig())
            control.memory_store = store
            control.recall(memory.memory_id, "scene")
            execute_control_command(control, "perceive")
            self.assertEqual(control.snapshot().memory_id, memory.memory_id)
            before = control.snapshot().prompt_version
            execute_control_command(control, "select_target", {"category": "nut", "attributes": {"color": "yellow"}})
            cfg = control.snapshot()
            self.assertEqual(cfg.prompt, "yellow nut")
            self.assertIsNone(cfg.memory_id)
            self.assertIsNone(cfg.memory_view)
            self.assertGreater(cfg.prompt_version, before)

    def test_remember_round_trip_keeps_multi_view_crops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectMemoryStore(directory)
            frames = {
                "scene": np.zeros((40, 40, 3), dtype=np.uint8),
                "wrist": np.full((40, 40, 3), 255, dtype=np.uint8),
            }
            detections = {
                "scene": Detection(np.array([5, 6, 24, 25], dtype=np.float32), "cube", 0.91),
                "wrist": Detection(np.array([8, 9, 21, 22], dtype=np.float32), "cube", 0.83),
            }
            memory = store.remember("cube", frames, detections, aliases=["block"])
            loaded = store.get(memory.memory_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.views), 2)
            self.assertEqual(store.find("BLOCK")[0].memory_id, memory.memory_id)
            self.assertTrue(all(Path(view.image).is_file() for view in loaded.views))

    def test_recall_arms_yoloe_image_prompt_without_florence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectMemoryStore(directory)
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            memory = store.remember(
                "cube", {"scene": frame},
                {"scene": Detection(np.array([5, 5, 24, 24], dtype=np.float32), "cube")},
            )
            finder, yoloe = _FakeFinder(), _FakeYoloe()
            pipe = FindTrackPipeline("unused.pt", finder=finder, yoloe_tracker=yoloe, memory_store=store)
            result = pipe.process_frame(frame, RuntimeConfig(prompt="cube", memory_id=memory.memory_id))
            self.assertEqual(finder.calls, 0)
            self.assertEqual(yoloe.image_prompt_calls, 1)
            self.assertEqual(result.stats.path, "fast")


if __name__ == "__main__":
    unittest.main()
