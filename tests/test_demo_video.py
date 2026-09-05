import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.sim.demo_video import DemoVideoRecorder


class VideoTimingTests(unittest.TestCase):
    def test_coarse_fault_is_not_labeled_no_fault_or_rounded_to_two_cm(self):
        recorder = object.__new__(DemoVideoRecorder)
        recorder.fault_m, recorder.coarse_fault_m = 0., .025
        self.assertEqual(recorder._fault_note(), "粗接近目标移位 2.5 cm")
        recorder.fault_m = -.01
        self.assertIn("闭爪前目标移位 -1 cm", recorder._fault_note())
        recorder.fault_m = recorder.coarse_fault_m = 0.
        self.assertEqual(recorder._fault_note(), "无故障注入")

    def test_holds_gaps_and_does_not_accelerate_with_frequent_callbacks(self):
        recorder = object.__new__(DemoVideoRecorder)
        recorder.started, recorder.frames, recorder.last_frame, recorder.fps = None, 0, None, 10
        recorder.compose = lambda *a: np.zeros((2, 2, 3), dtype=np.uint8)
        def write(frame):
            recorder.frames += 1
        recorder._write = write
        with patch("mr_liu.sim.demo_video.time.monotonic", side_effect=[20., 20.001, 20.002, 21.]):
            for _ in range(3):
                recorder.capture(None, None, None)
            self.assertEqual(recorder.frames, 1)
            recorder.capture(None, None, None)
            self.assertEqual(recorder.frames, 11)

    def test_missing_panels_render_without_error(self):
        from PIL import Image
        canvas = Image.new("RGB", (32, 24))
        DemoVideoRecorder._panel(canvas, None, (0, 0), (32, 24))
        self.assertEqual(canvas.getbbox(), None)


if __name__ == "__main__":
    unittest.main()
