# FineGrasp generalization audit — 2026-09-05

Status: experimental, NOT validated for real hardware or general industrial use.
The upstream GraspGenX weights have not been fine-tuned in this project.

## Archived evidence

- `output/unseen_benchmark/a7d72fe_seed0_full`: 8/8 feasible grasps and 1/1 expected rejection.
- `output/unseen_benchmark/7d38bf9_seeds1_2`: 12/16 feasible grasps and 2/2 expected rejections.
- Combined historical evidence: 20/24 feasible successes. Mug 1/3, hammer 2/3,
  reflective cylinder 2/3. These are small primitive/compound proxy tests,
  with only +/-3 mm XY and +/-18 degrees yaw. They are not held-out real objects.
- `output/unseen_benchmark/axis18_wrench_seed1`: wrench failed, no physical lift.
- GraspGenX and VGN were both executed, but in different physical benchmarks;
  their success rates are not a controlled head-to-head comparison.

## Experimental checkpoint warning

The checkpoint preceding the multi-view implementation preserves development
experiments for reproducibility, not as a release candidate. It includes width,
axis and handle heuristics that have not passed full regression. In particular:

- Some handle postprocessing moves a pose without rescoring it.
- Tabletop completion changes candidate height based on a support assumption.
- Isaac excludes the target from the planner; local finger/target checks are incomplete.
- A stalled gripper is not an independently measured target contact.
- Lift verification can mistake fingers/background for the target.
- The fast tracker only estimates translation; the rotation replan condition is ineffective.
- Coarse-position seeding is still used repeatedly, despite the tracker's older docstring.
- There is no active viewpoint planner or multi-frame geometric fusion.
- Adding cases changes the random sequence of later cases. Model sampling is not seeded.
- The wrench thickness changed from 12 to 16 mm during development; these are different cases.

69 unit tests pass at this checkpoint. Unit tests do not establish physical success.

## Implementation and evaluation gate

1. Preserve this checkpoint. Freeze case manifests independently of ordering and
   record configuration, code revision, model weights and inference seeds.
2. Fix false-positive verification and evaluate actual finger/target approach
   geometry separately from global robot/world collision planning.
3. Add bounded informative wrist views and fusion using each observation's
   synchronized hand-eye chain. Unknown space remains unknown. Invalidate old
   geometry on object motion, identity mismatch or inconsistent registration.
4. Maintain image identity independently of a fixed upstream coarse point;
   expose optional open-vocabulary initialization through a replaceable segmenter.
5. Refresh observations after slow inference and before movement; implement
   observable rotation change detection and bounded latest-frame servo updates.
6. Distinguish uncertainty from demonstrated geometric infeasibility. Missing
   depth and a thin semantic label alone do not prove physical infeasibility.
7. Use separate development and untouched acceptance manifests. Test multiple
   object instances and families, poses, lighting, depth degradation and seeds.
   Report attempt coverage, false rejection, false success, physical lift, cycle
   time, selected proposal branch and failures, not only conditional success.

The original push gate required preliminary generalization evidence. On
2026-09-05 the user explicitly requested a BusAgent README and GitHub push of
the development baseline before further enhancements. This authorizes publishing
the checkpoint, not claiming that it passes the generalization gate. No real
hardware success may be inferred from simulation. See BUSAGENT_README.md for the
current handoff and FINE_GRASP_BASELINE.md for the latest failed verification.
