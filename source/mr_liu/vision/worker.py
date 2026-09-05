"""One inference job in flight. No Kit/USD calls or unbounded frame backlog."""
from concurrent.futures import ThreadPoolExecutor


class VisionWorker:
    def __init__(self, process):
        self.process = process
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
        self.future = None
        self.closed = False

    @property
    def available(self):
        return not self.closed and self.future is None

    def submit(self, packet):
        if not self.available:
            return False
        self.future = self.executor.submit(self.process, packet)
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return future.result()

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)


def frozen_unprojector(unproject):
    """Capture a pinhole camera's affine rays on the simulation thread.

    Inference may finish after the camera moves; never re-read its new pose
    when interpreting depth captured with an older RGB frame.
    """
    import numpy as np
    points = np.asarray(unproject(np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float),
                                   np.array([0, 1, 1, 1], dtype=float))).copy()
    origin, ray = points[0], points[1]-points[0]
    dx, dy = points[2]-points[1], points[3]-points[1]

    def project(pixels, depth):
        pixels, depth = np.asarray(pixels), np.asarray(depth)
        return origin + depth[:, None]*(ray + pixels[:, :1]*dx + pixels[:, 1:2]*dy)
    return project
