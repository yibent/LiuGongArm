"""Object memory persistence and YOLOE image-prompt behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

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
