"""Warm the GraspGenX sweep sampler while Isaac Sim is starting.

The model server lazily constructs a sampler for each gripper sweep volume.
Doing that work in parallel with Isaac startup keeps the first wrist-camera
request inside the normal online inference timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import fine_grasp_config  # noqa: E402
from mr_liu.grasp.backends.graspgenx import (  # noqa: E402
    GraspGenXConfig,
    ZmqGraspGenXTransport,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--timeout-ms", type=int, default=45000)
    args = parser.parse_args()

    config = GraspGenXConfig.from_mapping(fine_grasp_config()["backends"]["graspgenx"])
    config = replace(config, port=args.port, timeout_ms=args.timeout_ms)
    transport = ZmqGraspGenXTransport(config.host, config.port, config.timeout_ms)
    rng = np.random.default_rng(731)
    points = rng.uniform(
        [-0.018, -0.018, 0.16],
        [0.018, 0.018, 0.20],
        size=(1024, 3),
    ).astype(np.float32)
    started = time.perf_counter()
    try:
        poses, scores, _tags = transport.infer_object(
            points,
            config.sweep_volume.to_wire(),
            planner=config.planner,
            num_grasps=config.num_grasps,
            grasp_threshold=config.min_confidence,
            topk_num_grasps=config.max_candidates,
            return_branch_tags=True,
        )
    finally:
        transport.close()
    if not len(poses) or not len(scores) or not np.isfinite(scores).all():
        raise RuntimeError("GraspGenX warmup returned no valid scored candidates")
    print(
        json.dumps(
            {
                "status": "ready",
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "candidate_count": len(poses),
                "top_score": float(np.max(scores)) if len(scores) else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
