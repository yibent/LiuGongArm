"""Worker tests without loading Torch, Isaac or model weights."""
import importlib.util
import threading
import time
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('vision_worker', Path(__file__).resolve().parents[1]/'source/mr_liu/vision/worker.py')
worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_module)


class WorkerTests(unittest.TestCase):
    def test_inference_does_not_block_or_accumulate_frames(self):
        release = threading.Event()
        def process(value):
            release.wait(2)
            return value
        worker = worker_module.VisionWorker(process)
        try:
            self.assertTrue(worker.submit('first'))
            self.assertFalse(worker.submit('obsolete'))
            self.assertIsNone(worker.poll())
            release.set()
            result = None
            for _ in range(100):
                result = worker.poll()
                if result is not None: break
                time.sleep(.01)
            self.assertEqual(result, 'first')
            self.assertTrue(worker.available)
        finally:
            release.set()
            worker.close()

    def test_unprojection_keeps_capture_pose_after_camera_moves(self):
        origin = np.array([1., 2., 3.])
        def camera(pixels, depth):
            rays = np.c_[pixels[:, 0]/500., pixels[:, 1]/500., np.ones(len(pixels))]
            return origin + depth[:, None]*rays
        frozen = worker_module.frozen_unprojector(camera)
        pixels = np.array([[120., 240.], [320., 40.]])
        depth = np.array([1.3, .8])
        expected = camera(pixels, depth).copy()
        origin[:] = 100
        np.testing.assert_allclose(frozen(pixels, depth), expected, atol=1e-10)
