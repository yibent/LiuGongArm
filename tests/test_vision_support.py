"""No model downloads, camera hardware or Isaac application required."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source")]

import vision_main
from find_and_track.cv_tracker import CvFeatureTracker
from find_and_track.florence_finder import FlorenceFinder, OVD
from find_and_track.pipeline import FindTrackPipeline
from find_and_track.settings import default_florence, default_yoloe
from find_and_track.types import Detection, RuntimeConfig


class VisionSupportTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(2).integers(0, 255, (100, 120, 3), dtype=np.uint8)
        self.det = Detection(np.array([20, 20, 60, 65], dtype=np.float32), "part")

    def test_cv_accepts_modern_none_init_return(self):
        mil = Mock()
        mil.init.return_value = None
        with patch("find_and_track.cv_tracker.cv2.TrackerMIL_create", return_value=mil):
            tracker = CvFeatureTracker()
            tracker.init(self.frame, [self.det])
        self.assertIs(tracker.tracks[0].mil, mil)

    def test_florence_decode_preserves_adjacent_location_tokens(self):
        finder = FlorenceFinder(device="cpu")
        finder.model = Mock()
        finder.processor = Mock(return_value={"input_ids": torch.tensor([[0, 1]])})
        finder.processor.batch_decode.return_value = ["bus<loc_1><loc_2><loc_3><loc_4>"]
        finder._run_task(Image.fromarray(self.frame), OVD, "bus", 1)
        self.assertFalse(finder.processor.batch_decode.call_args.kwargs["spaces_between_special_tokens"])

    def test_cv_rejects_explicit_failed_init(self):
        mil = Mock()
        mil.init.return_value = False
        with patch("find_and_track.cv_tracker.cv2.TrackerMIL_create", return_value=mil):
            tracker = CvFeatureTracker()
            tracker.init(self.frame, [self.det])
        self.assertIsNone(tracker.tracks[0].mil)

    def test_cli_cv_does_not_require_yolo_weights(self):
        with patch.object(sys, "argv", ["vision_main.py", "--source", "sample.mp4", "--fast", "cv", "--weights", "no-such-weights.pt"]), patch.object(vision_main, "run_cli", return_value=0) as run:
            self.assertEqual(vision_main.main(), 0)
            run.assert_called_once()

    def test_cli_yolo_requires_weight_file(self):
        with patch.object(sys, "argv", ["vision_main.py", "--source", "sample.mp4", "--weights", "no-such-weights.pt"]), patch.object(vision_main, "run_cli") as run:
            self.assertEqual(vision_main.main(), 1)
            run.assert_not_called()

    def test_webui_receives_cli_config_and_model(self):
        with patch.object(sys, "argv", ["vision_main.py", "--webui", "--fast", "cv", "--prompt", "wrench", "--florence", "local-florence", "--interval", "0"]), patch("find_and_track.webui.serve") as serve:
            self.assertEqual(vision_main.main(), 0)
        kwargs = serve.call_args.kwargs
        self.assertEqual(kwargs["florence_id"], "local-florence")
        self.assertEqual(kwargs["config"].fast_backend, "cv")
        self.assertEqual(kwargs["config"].prompt, "wrench")
        self.assertEqual(kwargs["config"].slow_interval, 0)

    def test_model_environment_overrides(self):
        with patch.dict(os.environ, {"BUSAGENT_FLORENCE_MODEL": "my-florence", "BUSAGENT_YOLOE_WEIGHTS": "my-yolo.pt"}):
            self.assertEqual(default_florence(), "my-florence")
            self.assertEqual(default_yoloe(), Path("my-yolo.pt"))

    def test_cv_pipeline_load_never_loads_yolo(self):
        pipe = FindTrackPipeline(device="cpu")
        pipe.florence = Mock()
        pipe.yoloe = Mock()
        pipe.load("cv")
        pipe.florence.load.assert_called_once()
        pipe.yoloe.load.assert_not_called()

    def test_switching_backend_reinitializes_with_latest_frame(self):
        pipe = FindTrackPipeline(device="cpu")
        pipe.florence = Mock()
        pipe.florence.find.return_value = [self.det]
        pipe.cv = Mock()
        pipe.cv.update.return_value = [self.det]
        pipe.yoloe = Mock()
        pipe.yoloe.detect.return_value = [self.det]
        cfg = RuntimeConfig(prompt="part", fast_backend="cv", slow_interval=0)
        pipe.process_frame(self.frame, cfg)
        cfg.fast_backend = "yoloe"
        current = self.frame.copy()
        pipe.process_frame(current, cfg)
        self.assertEqual(pipe.florence.find.call_count, 2)
        self.assertIs(pipe.yoloe.set_visual_prompt.call_args.args[0], current)
        self.assertEqual(pipe.cv.init.call_count, 1)

    def test_closing_video_generator_releases_camera(self):
        pipe = FindTrackPipeline(device="cpu")
        pipe._loaded = True
        pipe.process_frame = Mock(return_value=Mock(image=self.frame))
        cap = Mock()
        cap.get.return_value = 15
        cap.read.return_value = (True, self.frame)
        with patch("find_and_track.pipeline.cv2.VideoCapture", return_value=cap):
            frames = pipe.process_source(0, RuntimeConfig(fast_backend="cv"))
            next(frames)
            frames.close()
        cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
