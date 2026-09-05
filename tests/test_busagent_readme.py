"""Keep the BusAgent handoff's public Python example executable and safe."""

from __future__ import annotations

import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "tests"))

from test_fine_grasp_node import FakeCamera, FakeGripper, FakeMotion


class BusAgentReadmeTests(unittest.TestCase):
    def test_python_example_builds_skill_and_dry_runs_without_actions(self):
        doc = (ROOT / "docs/BUSAGENT_README.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", doc, flags=re.DOTALL)
        self.assertEqual(len(blocks), 1)
        namespace = {}
        exec(compile(blocks[0], "BUSAGENT_README.md", "exec"), namespace)
        motion = FakeMotion(time.monotonic)
        gripper = FakeGripper(time.monotonic)
        motion.move_to = Mock(side_effect=AssertionError("dry-run attempted motion"))
        motion.servo_to = Mock(side_effect=AssertionError("dry-run attempted servo"))
        gripper.open = Mock(side_effect=AssertionError("dry-run attempted gripper open"))
        gripper.close = Mock(side_effect=AssertionError("dry-run attempted gripper close"))
        with tempfile.TemporaryDirectory(prefix="busagent-readme-test-") as run_dir:
            skill = namespace["build_fine_grasp_skill"](
                camera=FakeCamera(time.monotonic), motion=motion, gripper=gripper,
                run_dir=run_dir, backend="geometric",
            )
            result = namespace["execute_nearby_target"](
                skill, object_id="unseen-part", label="part",
                coarse_position_base_m=(0.0, 0.0, 1.2), request_id="readme-smoke",
            )
            self.assertTrue(result.success, result.message)
            self.assertEqual(result.request_id, "readme-smoke")
            self.assertIsNotNone(result.selected_grasp)
            self.assertEqual(skill.name, "fine_grasp")
            self.assertTrue((Path(run_dir) / "debug/trace.jsonl").is_file())
        motion.move_to.assert_not_called()
        motion.servo_to.assert_not_called()
        gripper.open.assert_not_called()
        gripper.close.assert_not_called()

    def test_local_document_links_exist(self):
        for relative in ("README.md", "docs/BUSAGENT_README.md"):
            path = ROOT / relative
            doc = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", doc):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(document=relative, link=target):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).is_file())


if __name__ == "__main__":
    unittest.main()
