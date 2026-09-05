# FineGrasp unseen-object physical benchmark

The benchmark runs every trial in a fresh Isaac Sim process and exercises the
same eye-in-hand `GeneralGraspNode` used by the normal demo. It is not a pose-only
or dry-run evaluation: success requires both the node's close/lift verification
and a ground-truth rigid-body lift of at least 25 mm.

## Run

From the repository root on the Isaac workstation:

```powershell
$env:ISAAC_PYTHON = 'D:\isaac\env_isaacsim60\python.exe'
& $env:ISAAC_PYTHON scripts\run_fine_grasp_benchmark.py
```

The default suite contains a cube, sphere, vertical cylinder, 7 mm thin part,
and a reflective metallic cylinder. Positions and yaw are reproducibly varied
by seed:

```powershell
& $env:ISAAC_PYTHON scripts\run_fine_grasp_benchmark.py `
  --seeds 0,1,2 --cases cube,sphere,cylinder,thin,metallic_part
```

Select the slow grasp generator without changing the stable BusAgent contract
with `--backend geometric` (default) or `--backend graspgenx`. The latter expects
the configured local GraspGenX ZMQ server to be running; its configured geometric
fallback remains active for unavailable/empty neural proposals.

Run a quick case or inspect the expanded scene definitions without starting
Isaac:

```powershell
& $env:ISAAC_PYTHON scripts\run_fine_grasp_benchmark.py --cases sphere --seeds 4
& $env:ISAAC_PYTHON scripts\run_fine_grasp_benchmark.py --list --seeds 4
```

Custom objects can be supplied with repeatable `--case-json` arguments. The
JSON fields follow `BenchmarkCase` in `source/mr_liu/grasp/benchmark.py`; supported
primitive shapes are `cube`, `sphere`, `cylinder`, and `thin`, while material is
`matte`, `metallic`, or `reflective`.

## Artifacts and scoring

Each case directory contains the exact `case.json`, replay command, process
logs, wrist RGB/depth/segmentation debugging, the normal FineGrasp `report.json`,
and a normalized `benchmark_result.json`. The benchmark root is updated after
every trial so an interrupted long suite retains useful results:

- `summary.json`: machine-readable results, failure histogram, grouped shape
  success, and mean/median/P95 metrics.
- `SUMMARY.md`: human-readable result table.
- `NNN_<case>_seed<seed>/`: fully replayable per-trial evidence.

`success` is deliberately stricter than `result.success` in the normal report.
It requires `actual_target_lift_m >= 0.025`. A node success without physical
lift is classified `false_positive_no_physical_lift`; a physically lifted object
rejected by verification is `verification_false_negative`. Other failures retain
the stable `FailureCode` such as `ik_unreachable`, `insufficient_depth`, or
`gripper_width_infeasible`.

Use `--dry-run` only to diagnose candidate/IK filtering. Dry runs are always
classified `dry_run_not_physical` and never contribute a physical success.
