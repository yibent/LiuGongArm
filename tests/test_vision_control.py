"""Runtime prompt API tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from find_and_track import RuntimeConfig
from mr_liu.vision.control import (
    VisionControlServer,
    VisionRuntimeControl,
    execute_control_command,
)


class VisionControlTests(unittest.TestCase):
    def test_semantic_commands_update_target_and_follow_state(self) -> None:
        control = VisionRuntimeControl(RuntimeConfig(prompt="red cube"))
        result = execute_control_command(
            control,
            "select_target",
            {"category": "block", "attributes": {"color": "green"}},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(control.snapshot().prompt, "green block")
        execute_control_command(control, "stop")
        self.assertFalse(control.follow_enabled())
        execute_control_command(control, "follow", {"enabled": True})
        self.assertTrue(control.follow_enabled())

    def test_unimplemented_physical_skill_is_rejected(self) -> None:
        control = VisionRuntimeControl(RuntimeConfig())
        with self.assertRaises(NotImplementedError):
            execute_control_command(control, "grasp")

    def test_prompt_update_bumps_versions_for_both_view_pipelines(self) -> None:
        control = VisionRuntimeControl(RuntimeConfig(prompt="red cube"))
        before = control.snapshot()
        after = control.set_prompt("orange power drill")
        self.assertEqual(after.prompt, "orange power drill")
        self.assertEqual(after.prompt_version, before.prompt_version + 1)
        self.assertEqual(after.find_epoch, before.find_epoch + 1)

    def test_http_prompt_update_and_force_find(self) -> None:
        control = VisionRuntimeControl(RuntimeConfig(prompt="red cube"))
        server = VisionControlServer(control, host="127.0.0.1", port=0)
        _, port = server.start()
        try:
            body = json.dumps({"prompt": "wrench"}).encode()
            request = Request(
                f"http://127.0.0.1:{port}/api/prompt",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                result = json.load(response)
            self.assertEqual(result["prompt"], "wrench")

            request = Request(
                f"http://127.0.0.1:{port}/api/find",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                result = json.load(response)
            self.assertEqual(result["find_epoch"], 2)

            with urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as response:
                status = json.load(response)
            self.assertEqual(status["prompt"], "wrench")
            self.assertEqual(status["prompt_version"], 1)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
