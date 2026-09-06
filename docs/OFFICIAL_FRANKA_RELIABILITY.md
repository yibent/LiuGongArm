# Official Franka PickPlace reliability runs

These runs use NVIDIA Isaac Sim's official `PickPlaceController` with the
official Franka `RMPFlowController`. The project payload and release validators
are not in this execution path.

## Batch results

The 12-case matrix in
`configs/eval_official_franka_pick_place_v1.json` covers mirrored and edge
workspace coordinates, short and long transfers, small/large/tall/flat
rectangular parts, a 45-degree yaw part, a raised platform, and a 120 gram
part. With a 10 mm final center tolerance, the split runs in
`output/franka/official_reliability_20260906_b` and
`output/franka/official_reliability_20260906_c` completed **12/12** physical
pick-and-place trials. Final center errors were 1.5–6.2 mm except the yaw
part, which was 1.72 mm after commanding the official controller with a 45°
end-effector yaw.

The first yaw-part run without that yaw alignment ended 39.8 mm away from the
requested target. This is why the object orientation is now an explicit input
to the official controller wrapper.

## Videos

- [standard Franka pick and place](../output/franka/official_videos_20260906_a/01_standard_b/official_pick_place.mp4)
- [45° rotated rectangular part](../output/franka/official_videos_20260906_a/02_rotated_block/official_pick_place.mp4)
- [industrial multi-part scene](../output/franka/official_videos_20260906_a/03_industrial/official_pick_place.mp4)
- [target moved during approach](../output/franka/official_videos_20260906_a/04_target_moved_during_approach/official_pick_place.mp4)
- [target moved immediately before closing](../output/franka/official_videos_20260906_a/05_target_moved_before_close_b/official_pick_place.mp4)

The industrial video keeps six additional dynamic parts in the scene while the
blue cube is the commanded target: multiple rectangular blocks and a cylinder.

## Target moved during grasp

The fault-injection runs move the cube by 80 mm and stop its velocity.

- During official phase 1 (approach), the controller reads the current object
  position every cycle. It followed the moved cube and still placed it with a
  2.74 mm final error.
- During official phase 2 (the short settle immediately before closing), the
  official state machine had already latched the picking XY. It completed all
  ten phases, but closed at the old location; the cube stayed at the moved
  start position and the physical success check was false.

This behavior is recorded as a limitation of the official baseline. The next
industrial layer should watch the target identity and restart the official
controller when a target is lost or moves after the approach phase; it should
not treat phase completion alone as a successful grasp.

## Shape boundary checks

The same official controller also succeeded on a compact 50 mm diameter,
50 mm-high dynamic cylinder (`output/franka/official_shapes_20260906_b/cylinder_compact`)
with 3.02 mm final error. A free sphere and an over-tall cylinder were not
reliable with the flat-ground target: the sphere rolled after release and the
over-tall cylinder was not stably captured. They need shape-specific fixtures
or grasp orientation/height profiles rather than being silently counted as
successful cube transfers.
