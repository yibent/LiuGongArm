from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from mr_liu.sim.clutter import (  # noqa: E402
    ClutterBody,
    ClutterScene,
    default_clutter_scenes,
    load_scene,
    write_scene,
)


class ClutterSceneTests(unittest.TestCase):
    def test_default_suite_covers_clutter_collision_and_semantic_cases(self) -> None:
        scenes = default_clutter_scenes()
        self.assertEqual(
            {scene.name for scene in scenes},
            {"mixed_tools", "occluded_target", "narrow_corridor", "duplicate_label", "moving_blocker"},
        )
        self.assertTrue(any(scene.expected_outcome == "safe_reject" for scene in scenes))
        self.assertTrue(any(scene.metadata.get("requires_reobserve") for scene in scenes))
        self.assertTrue(any(scene.metadata.get("self_collision_risk") for scene in scenes))
        self.assertTrue(any(any(body.dynamic for body in scene.bodies) for scene in scenes))

    def test_scene_json_round_trip_preserves_tuple_geometry(self) -> None:
        scene = default_clutter_scenes()[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.json"
            write_scene(path, scene)
            loaded = load_scene(path)
        self.assertEqual(loaded, scene)
        self.assertIsInstance(loaded.bodies[0].position_m, tuple)
        self.assertEqual(loaded.name, scene.name)

    def test_invalid_body_and_duplicate_names_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ClutterBody("bad", "mesh", (0, 0, 1), (0.1, 0.1, 0.1))
        body = ClutterBody("same", "box", (0, 0, 1), (0.1, 0.1, 0.1))
        with self.assertRaises(ValueError):
            ClutterScene("bad", "duplicate", (body, body))

    def test_mapping_defaults_are_explicit_and_validated(self) -> None:
        scene = ClutterScene.from_mapping({
            "name": "minimal",
            "description": "one blocker",
            "bodies": [{
                "name": "wall",
                "shape": "box",
                "position_m": [0.3, 0.0, 1.1],
                "dimensions_m": [0.05, 0.2, 0.1],
            }],
        })
        self.assertEqual(scene.expected_outcome, "grasp_or_replan")
        self.assertFalse(scene.bodies[0].dynamic)
        self.assertEqual(scene.bodies[0].semantic_label, "obstacle")


if __name__ == "__main__":
    unittest.main()
