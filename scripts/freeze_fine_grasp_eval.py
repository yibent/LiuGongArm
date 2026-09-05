"""Freeze object-level development/acceptance cases; refuse overwrites.

Held out from BusAgent tuning, not asserted absent from model pretraining.
"""
import argparse
import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.grasp.benchmark import BenchmarkCase, default_unseen_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "configs/eval/fine_grasp_v2")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen evaluation: {args.output}")
    development = default_unseen_cases(0)
    wrench = next(c for c in development if c.name == "wrench")
    development.append(replace(wrench, name="wrench_12mm", dimensions_m=(0.125, 0.038, 0.012),
                               position_m=(*wrench.position_m[:2], 1.056)))
    asset_root = Path("_models/vgn/official_data/data/urdfs/pile/test")
    assets = ["lemon_poisson_000_visual.obj", "mushroom_poisson_005_visual.obj",
              "pliers_poisson_014_visual.obj", "hammer_poisson_012_visual.obj",
              "spray_can_poisson_001_visual.obj", "bottle_new_poisson_000_visual.obj"]
    acceptance = []
    for asset in assets:
        path = asset_root / asset
        digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for seed in (101, 202, 303):
            rng = random.Random(f"heldout-v2:{asset}:{seed}")
            acceptance.append(BenchmarkCase(
                name=f"heldout_{Path(asset).stem}", shape="mesh", dimensions_m=(0.070,)*3,
                position_m=(0.324 + rng.uniform(-0.015, 0.015),
                            -0.198 + rng.uniform(-0.015, 0.015), 1.085),
                seed=seed, yaw_deg=rng.uniform(-180, 180), mass_kg=0.08,
                mesh_path=path.as_posix(), light_scale=(0.65, 1.0, 1.4)[(101, 202, 303).index(seed)],
                metadata={"asset_sha256": digest, "split": "acceptance", "training_overlap": "unknown"}))
    for seed in (101, 202, 303):
        mug = next(c for c in default_unseen_cases(seed) if c.name == "coffee_mug")
        acceptance.append(replace(mug, name="heldout_mug_proxy", yaw_deg=float(seed % 180 - 90),
                                  metadata={"split": "acceptance", "asset_kind": "compound_proxy"}))
    args.output.mkdir(parents=True)
    manifests = {}
    for split, cases in (("development", development), ("acceptance", acceptance)):
        directory = args.output / split
        directory.mkdir()
        paths = []
        for index, case in enumerate(cases):
            path = directory / f"{index:03d}_{case.name}_{case.seed}.json"
            path.write_text(json.dumps(case.to_dict(), indent=2) + "\n", encoding="utf-8")
            paths.append({"path": path.relative_to(args.output).as_posix(),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifests[split] = paths
    (args.output / "manifest.json").write_text(json.dumps({
        "version": 2, "pretraining_holdout_proven": False,
        "acceptance_gate": {"minimum_feasible_success_rate": 0.80,
                            "maximum_false_successes": 0, "maximum_family_zero_successes": 0,
                            "minimum_attempt_coverage": 0.80},
        "splits": manifests}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"development": len(development), "acceptance": len(acceptance)}))


if __name__ == "__main__":
    main()
