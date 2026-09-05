"""Checker tests only. Run with Python -B; no GPU, model or robot calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_deployment as audit


class AuditTests(unittest.TestCase):
    def test_text_hash_accepts_crlf(self):
        with tempfile.TemporaryDirectory(prefix="grasp-audit-test-") as folder:
            path = Path(folder) / "config.yaml"
            path.write_bytes(b"a: 1\r\nb: 2\r\n")
            self.assertEqual(audit.text_hash(path), hashlib.sha256(b"a: 1\nb: 2\n").hexdigest())

    def test_snapshot_manifest_integrity(self):
        manifest = json.loads((audit.HERE / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            self.assertEqual(audit.text_hash(audit.HERE / entry["snapshot"]), entry["sha256_lf_utf8"])

    def test_missing_changed_and_weight_mismatch_are_reported(self):
        manifest = json.loads((audit.HERE / "manifest.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="grasp-audit-test-") as folder:
            root = Path(folder)
            changed = root / manifest["files"][0]["target"]
            changed.parent.mkdir(parents=True)
            changed.write_text("changed: true\n", encoding="utf-8")
            weight = root / manifest["weights"][0]["target"]
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"not a checkpoint")
            with patch.object(audit, "git", return_value={"exit_code": 1, "stdout": ""}):
                report = audit.collect(root, manifest, hash_weights=True)
            self.assertFalse(report["files"][0]["matches_reference"])
            self.assertFalse(report["files"][1]["exists"])
            self.assertEqual(report["weights"][0]["hash_status"], "mismatch")
            self.assertFalse(report["weights"][0]["size_matches_reference"])
            self.assertFalse(report["weights"][1]["exists"])

    def test_missing_command_is_structured(self):
        with patch.object(audit.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(audit.run_readonly(["absent"])["error"], "FileNotFoundError")

    def test_probe_does_not_import_models(self):
        with patch.object(audit, "run_readonly", return_value={}) as mocked:
            audit.probe_model_python("trusted-python")
        argv = mocked.call_args.args[0]
        self.assertEqual(argv[:4], ["trusted-python", "-I", "-B", "-c"])
        self.assertNotIn("import torch", argv[4])
        self.assertNotIn("import graspgenx", argv[4])


if __name__ == "__main__":
    unittest.main()
