import sys
import unittest
import tempfile
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.arena.service import CommandQueue


class CommandQueueTests(unittest.TestCase):
    def test_stop_and_claim_are_serialized(self):
        queue = CommandQueue(); event = threading.Event()
        queue.submit({"command_id": "a", "skill": "grasp"})
        queue.interrupt(event)
        self.assertFalse(queue.claim("a"))
        queue.submit({"command_id": "b", "skill": "grasp"})
        self.assertTrue(queue.claim("b"))
        queue.interrupt(event)
        self.assertTrue(event.is_set())

    def test_restart_does_not_replay_an_unfinished_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            request = {"command_id": "a", "skill": "grasp"}
            CommandQueue(path).submit(request)
            restarted = CommandQueue(path)
            self.assertFalse(restarted.submit(request)["ok"])
            self.assertEqual(restarted.queue.qsize(), 0)

    def test_idle_stop_does_not_cancel_a_future_command(self):
        queue = CommandQueue(); event = threading.Event()
        queue.interrupt(event)
        self.assertFalse(event.is_set())
        queue.submit({"command_id": "a", "skill": "grasp"})
        queue.interrupt(event)
        self.assertEqual(queue.result("a")["state"], "cancelled")
        queue.submit({"command_id": "b", "skill": "grasp"})
        self.assertEqual(queue.result("b")["state"], "accepted")

    def test_replay_does_not_repeat_physical_action(self):
        queue = CommandQueue()
        request = {"command_id": "a", "skill": "grasp"}
        queue.submit(request)
        queue.submit(request.copy())
        self.assertEqual(queue.queue.qsize(), 1)
        queue.update("a", {"ok": True, "state": "completed"})
        self.assertEqual(queue.submit(request)["state"], "completed")

    def test_conflicting_replay_and_overlapping_commands_are_rejected(self):
        queue = CommandQueue(); queue.submit({"command_id": "a", "skill": "grasp"})
        for request in [{"command_id": "a", "skill": "home"}, {"command_id": "b", "skill": "grasp"}]:
            with self.assertRaises(ValueError): queue.submit(request)
