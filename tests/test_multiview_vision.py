"""Dual-view FIND/TRACK state tests using lightweight fake models."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from find_and_track import Detection, FindTrackPipeline, MultiViewFindTrackPipeline, RuntimeConfig


class FakeFinder:
    ready = True
    loaded_id = "fake-florence"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self) -> None:
        return

    def find(self, _frame, prompts) -> list[Detection]:
        label = list(prompts)[0]
        self.calls.append(label)
        return [Detection(np.asarray([8, 8, 24, 24], dtype=np.float32), label=label)]


class FakeYoloeTracker:
    ready = True

    def __init__(self) -> None:
        self.has_prompt = False
        self.label = ""
        self.track_calls = 0
        self.imgsz = 640
        self.conf = 0.25

    def load(self) -> None:
        return

    def reset(self) -> None:
        self.has_prompt = False

    def set_visual_prompt(self, _frame, detections, conf=None) -> None:
        self.has_prompt = True
        self.label = detections[0].label

    def _detection(self) -> list[Detection]:
        return [
            Detection(
                np.asarray([8, 8, 24, 24], dtype=np.float32),
                label=self.label,
                track_id=7,
                source="fast",
            )
        ]

    def detect(self, _frame, conf=None) -> list[Detection]:
        return self._detection()

    def track(self, _frame, conf=None, persist=True) -> list[Detection]:
        self.track_calls += 1
        return self._detection()


class MultiViewVisionTests(unittest.TestCase):
    def test_memory_uses_per_camera_defaults_or_explicit_view(self) -> None:
        for requested_view in (None, "wrist"):
            with self.subTest(view=requested_view):
                pipelines = {view: Mock() for view in ("scene", "wrist")}
                multi = MultiViewFindTrackPipeline(pipelines=pipelines)
                frame = np.zeros((40, 40, 3), dtype=np.uint8)
                cfg = RuntimeConfig(memory_id="test-memory", memory_view=requested_view)
                multi.process_frames({"scene": frame, "wrist": frame}, cfg)
                for view, pipeline in pipelines.items():
                    view_cfg = pipeline.process_frame.call_args.args[1]
                    self.assertEqual(view_cfg.memory_id, "test-memory")
                    self.assertEqual(view_cfg.memory_view, requested_view or view)
                    self.assertIsNot(view_cfg, cfg)
                self.assertEqual(cfg.memory_view, requested_view)

    def test_prompt_change_refinds_then_both_views_resume_fast_tracking(self) -> None:
        finder = FakeFinder()
        trackers = {name: FakeYoloeTracker() for name in ("scene", "wrist")}
        pipelines = {
            name: FindTrackPipeline(
                "unused.pt",
                finder=finder,
                yoloe_tracker=trackers[name],
            )
            for name in trackers
        }
        multi = MultiViewFindTrackPipeline(pipelines=pipelines)
        multi.load("yoloe")
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frames = {"scene": frame, "wrist": frame.copy()}
        cfg = RuntimeConfig(prompt="red cube", prompt_version=0, slow_interval=0)

        first = multi.process_frames(frames, cfg)
        self.assertEqual({result.stats.path for result in first.values()}, {"slow"})
        self.assertEqual(finder.calls, ["red cube", "red cube"])

        second = multi.process_frames(frames, cfg)
        self.assertEqual({result.stats.path for result in second.values()}, {"fast"})
        self.assertTrue(all(tracker.track_calls == 1 for tracker in trackers.values()))

        cfg.prompt = "orange power drill"
        cfg.prompt_version += 1
        cfg.find_epoch += 1
        changed = multi.process_frames(frames, cfg)
        self.assertEqual({result.stats.path for result in changed.values()}, {"slow"})
        self.assertEqual(finder.calls[-2:], ["orange power drill", "orange power drill"])
        self.assertTrue(all(result.fast[0].label == "orange power drill" for result in changed.values()))


if __name__ == "__main__":
    unittest.main()
