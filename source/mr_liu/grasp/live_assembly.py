"""Live Web grasp assembly, sharing the tested recovery node (no demo scene)."""
from __future__ import annotations

from copy import deepcopy

from mr_liu.grasp.debug import DebugSegmenter, FineGraspDebugRecorder
from mr_liu.grasp.identity import GeometricIdentitySegmenter, RobotMaskedSegmenter
from mr_liu.grasp.node import GeneralGraspNode
from mr_liu.grasp.recovery import RecoveringGraspNode, RecoveryConfig
from mr_liu.grasp.scene_observer import SceneAssistedVerifier, SceneTargetObserver
from mr_liu.grasp.segmentation import AppearanceDepthTrackerSegmenter, SeededDepthSegmenter
from mr_liu.grasp.settings import FineGraspSettings


def assemble_live_grasp(*, config, camera, scene_camera, motion, gripper, backend,
                        recorder, trace, target):
    if scene_camera is None:
        raise ValueError("双相机抓取需要固定 RGB-D 相机，未降级为单相机。")
    observer = SceneTargetObserver(
        scene_camera, table_height_m=config["selection"]["table_height_m"],
        require_robot_mask=True,
        recorder=FineGraspDebugRecorder(recorder.output_dir / "overhead"),
    )

    def attempt_factory(index, failed, current_target):
        settings = deepcopy(config)
        multiview = index > 0
        settings["fusion"] = {**settings.get("fusion", {}), "enabled": multiview}
        settings["active_views"] = {**settings.get("active_views", {}), "enabled": multiview}
        segmenter = AppearanceDepthTrackerSegmenter(SeededDepthSegmenter())
        if multiview:
            segmenter = GeometricIdentitySegmenter(segmenter)
        segmenter = RobotMaskedSegmenter(segmenter, require_mask=True)
        attempt_recorder = FineGraspDebugRecorder(recorder.output_dir / f"attempt_{index + 1}" / "debug")

        def attempt_trace(event):
            attempt_recorder.trace(event)
            trace({**event, "attempt": index + 1})

        return GeneralGraspNode(
            camera=camera, motion=motion, gripper=gripper, backend=backend,
            segmenter=DebugSegmenter(segmenter, attempt_recorder),
            verifier=SceneAssistedVerifier(observer, motion, current_target),
            settings=FineGraspSettings.from_mapping(settings), trace=attempt_trace,
            failed_grasps=failed,
        )

    return RecoveringGraspNode(
        attempt_factory=attempt_factory, observer=observer, motion=motion, gripper=gripper,
        config=RecoveryConfig(max_attempts=2), trace=trace, require_retry_authorization=True,
    )
