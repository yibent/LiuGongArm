import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.arena.arrays import semantic_selection


def test_semantic_numeric_and_sim6_color_mappings():
    expected = np.array([[False, True], [True, False]])
    ids = np.array([[0, 2], [2, 0]], dtype=np.uint32)
    assert np.array_equal(semantic_selection(ids, {"2": {"class": "red_block"}}, "red_block"), expected)
    rgba = np.array([33, 243, 3, 255], np.uint8)
    colors = np.zeros((2, 2, 4), np.uint8); colors[expected] = rgba
    labels = {"(33, 243, 3, 255)": {"class": "red_block"}}
    assert np.array_equal(semantic_selection(colors, labels, "red_block"), expected)
    assert np.array_equal(semantic_selection(colors.view(np.uint32), labels, "red_block"), expected)
    assert not semantic_selection(colors, labels, "yellow_cylinder").any()
