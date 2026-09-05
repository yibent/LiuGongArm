"""Tabletop prop configuration checks that do not require Isaac Sim."""

from __future__ import annotations

import sys
import unittest
from math import isclose, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import scene_config  # noqa: E402


class TabletopPropConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.props = scene_config()["tabletop_props"]

    def test_expected_manipulation_categories_are_present(self) -> None:
        labels = {prop["semantic_label"] for prop in self.props}
        self.assertTrue({"block", "wrench", "bolt", "nut", "gear"}.issubset(labels))

    def test_prim_paths_and_names_are_unique(self) -> None:
        names = [prop["name"] for prop in self.props]
        paths = [prop["prim_path"] for prop in self.props]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(path.startswith("/World/TabletopProps/") for path in paths))

    def test_library_assets_are_official_isaac_paths(self) -> None:
        assets = [prop for prop in self.props if prop["kind"] == "asset"]
        self.assertGreaterEqual(len(assets), 5)
        self.assertTrue(all(prop["asset_rel"].startswith("/Isaac/") for prop in assets))

    def test_all_props_are_physical_and_start_above_table(self) -> None:
        table_z = float(scene_config()["table_pos"][2])
        for prop in self.props:
            self.assertGreater(float(prop["mass"]), 0.0, prop["name"])
            self.assertGreater(float(prop["position"][2]), table_z, prop["name"])
            self.assertEqual(len(prop["orientation_wxyz"]), 4, prop["name"])

    def test_robot_starts_rotated_counter_clockwise_90_degrees_about_z(self) -> None:
        orientation = scene_config()["robot_orient_wxyz"]
        half_sqrt = sqrt(0.5)
        expected = [half_sqrt, 0.0, 0.0, half_sqrt]
        self.assertTrue(all(isclose(a, b, abs_tol=1e-7) for a, b in zip(orientation, expected)))


if __name__ == "__main__":
    unittest.main()
