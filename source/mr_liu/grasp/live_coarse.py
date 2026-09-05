"""Live assembly of the colleague's label approach; no demo scene or joint presets."""
from concurrent.futures import ThreadPoolExecutor

from mr_liu.control.coarse_approach import CoarseApproach
from mr_liu.perception.semantic_target import SemanticFlowTarget
from mr_liu.grasp.segmentation import SeededDepthSegmenter


class ResponsiveLocator:
    """Only HTTP inference leaves Kit's thread; keep rendering and cancellation live."""
    def __init__(self, locator, advance, selector=None):
        self.locator, self.advance, self.selector = locator, advance, selector

    def locate(self, observation, label):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="label-locator")
        future = pool.submit(self.locator.locate, observation, label)
        try:
            while not future.done():
                self.advance()
            result = future.result()
            if self.selector in {"leftmost", "rightmost"} and result.get("boxes"):
                boxes = sorted(result["boxes"], key=lambda b: b["xyxy"][0])
                result = {**result, "boxes": [boxes[0 if self.selector == "leftmost" else -1]]}
            return result
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


def approach_for_live_grasp(*, scene_camera, wrist_camera, motion, gripper,
                            locator, target, table_height_m, trace, recorder):
    # The colleague's finger geometry assumes this observed opening.
    if not gripper.open(.075, speed_mps=.04):
        raise ValueError("夹爪未到达观察开度，尚未执行接近或闭爪。")
    tracker = SemanticFlowTarget(scene_camera, locator, target, table_height_m,
                                 trace=trace, recorder=recorder)
    updated = CoarseApproach(tracker, motion, trace=trace).run()
    observation = wrist_camera.capture(updated)
    mask = None if observation is None else SeededDepthSegmenter().segment(observation, updated)
    tracker.validate_wrist_handoff(observation, mask)
    return updated
