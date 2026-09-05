import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.benchmark import (
    BenchmarkCase,
    aggregate_runs,
    classify_report,
    default_demo_case,
    default_unseen_cases,
    load_case,
    render_markdown,
    write_case,
    validate_frozen_case_bytes,
)


class BenchmarkCaseTests(unittest.TestCase):
    def test_frozen_manifest_accepts_only_transport_newlines(self):
        import hashlib
        lf = b'{"seed": 1}\n'
        crlf = lf.replace(b'\n', b'\r\n')
        self.assertTrue(validate_frozen_case_bytes(lf, hashlib.sha256(crlf).hexdigest()))
        self.assertTrue(validate_frozen_case_bytes(crlf, hashlib.sha256(lf).hexdigest()))
        self.assertFalse(validate_frozen_case_bytes(b'{"seed": 2}\n', hashlib.sha256(lf).hexdigest()))

    def test_legacy_default_case_is_unchanged(self):
        case = default_demo_case()
        self.assertEqual(case.shape, "cube")
        self.assertEqual(case.dimensions_m, (0.036, 0.036, 0.036))
        self.assertEqual(case.position_m, (0.324, -0.198, 1.068))
        self.assertEqual(case.mass_kg, 0.035)

    def test_default_suite_is_deterministic_and_covers_required_geometry(self):
        first = default_unseen_cases(17)
        second = default_unseen_cases(17)
        self.assertEqual(first, second)
        self.assertEqual(
            {case.shape for case in first},
            {
                "cube",
                "sphere",
                "cylinder",
                "thin",
                "hammer",
                "wrench",
                "screwdriver",
                "key",
                "mug",
            },
        )
        self.assertTrue(any(case.reflective for case in first))
        self.assertTrue(any(case.thin for case in first))
        self.assertTrue(all(not case.expected_feasible for case in first if case.thin))
        self.assertTrue(any(case.fragile for case in first))
        self.assertTrue(
            any(case.metadata.get("scenario") == "industrial_tool" for case in first)
        )
        self.assertGreaterEqual(
            sum(case.metadata.get("scenario") == "industrial_tool" for case in first),
            4,
        )
        self.assertTrue(any(case.metadata.get("has_handle") for case in first))
        self.assertNotEqual(default_unseen_cases(18), first)

    def test_case_json_round_trip(self):
        case = default_unseen_cases(3)[-1]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.json"
            write_case(path, case)
            loaded = load_case(path)
        self.assertEqual(loaded, case)

    def test_invalid_case_is_rejected(self):
        with self.assertRaises(ValueError):
            BenchmarkCase(
                name="bad",
                shape="mesh",
                dimensions_m=(0.02, 0.02, 0.02),
                position_m=(0.0, 0.0, 0.0),
            )


class BenchmarkReportTests(unittest.TestCase):
    def test_recovery_is_not_necessarily_a_regrasp(self):
        report = self._report()
        report['result']['metrics']['grasp_attempts'] = 1
        report['attempt_results'] = [{'success': False}, {'success': True}]
        report['attempt_physics'] = [{'actual_target_lift_m': 0.0}, {'actual_target_lift_m': 0.08}]
        row = classify_report(report)
        self.assertFalse(row['first_attempt_success'])
        self.assertTrue(row['recovered_success'])
        self.assertFalse(row['successful_regrasp'])
        report['result']['metrics']['grasp_attempts'] = 2
        self.assertTrue(classify_report(report)['successful_regrasp'])

    def test_attempted_trials_use_actual_close_commands(self):
        report = self._report(success=False, lift=0.0, failure='grasp_not_detected')
        unknown = classify_report(report)
        report['result']['metrics']['close_commanded'] = 0
        no_close = classify_report(report)
        report['result']['metrics']['close_commanded'] = 1
        closed = classify_report(report)
        summary = aggregate_runs([unknown, no_close, closed])
        self.assertEqual(summary['attempted_trials'], 1)
        self.assertEqual(summary['attempt_count_unknown_trials'], 1)

    @staticmethod
    def _report(*, success=True, lift=0.08, failure=None, selected_backend=None, fallback=None):
        selected = None
        if selected_backend is not None:
            selected = {
                "backend": selected_backend,
                "metadata": {"fallback_reason": fallback} if fallback else {},
            }
        return {
            "result": {
                "success": success,
                "failure": failure,
                "selected_grasp": selected,
                "metrics": {
                    "final_translation_error_m": 0.003,
                    "model_latency_ms": 12.5,
                    "elapsed_s": 4.0,
                    "observations": 8,
                },
            },
            "actual_target_lift_m": lift,
        }

    def test_selected_backend_and_fallback_are_attributed(self):
        row = classify_report(
            self._report(
                selected_backend="graspgenx+geometric_antipodal_fallback",
                fallback="model_timeout",
            )
        )
        self.assertEqual(row["backend"], "graspgenx+geometric_antipodal_fallback")
        self.assertEqual(row["fallback_reason"], "model_timeout")

    def test_success_requires_node_and_ground_truth_lift(self):
        physical = classify_report(self._report())
        false_positive = classify_report(self._report(lift=0.001))
        self.assertTrue(physical["success"])
        self.assertFalse(false_positive["success"])
        self.assertEqual(
            false_positive["failure_category"], "false_positive_no_physical_lift"
        )

    def test_failure_and_process_errors_are_classified(self):
        failed = classify_report(
            self._report(success=False, lift=0.0, failure="ik_unreachable")
        )
        process = classify_report(None, process_returncode=1, process_error="boom")
        self.assertEqual(failed["failure_category"], "ik_unreachable")
        self.assertEqual(process["failure_category"], "process_error")

    def test_physical_infeasibility_rejection_is_a_correct_task_outcome(self):
        row = classify_report(
            self._report(success=False, lift=0.0, failure="table_clearance"),
            expected_feasible=False,
        )
        self.assertFalse(row["success"])
        self.assertTrue(row["correct_infeasibility_rejection"])
        self.assertTrue(row["task_success"])

    def test_aggregate_and_markdown_include_metrics_and_failures(self):
        good = {
            **classify_report(self._report()),
            "case": "cube",
            "shape": "cube",
            "seed": 1,
            "material": "matte",
        }
        bad = {
            **classify_report(
                self._report(success=False, lift=0.0, failure="servo_timeout")
            ),
            "case": "sphere",
            "shape": "sphere",
            "seed": 1,
            "material": "matte",
        }
        summary = aggregate_runs([good, bad])
        markdown = render_markdown(summary)
        self.assertEqual(summary["trials"], 2)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["failure_counts"], {"servo_timeout": 1})
        self.assertEqual(summary["by_material"]["matte"]["trials"], 2)
        self.assertEqual(summary["metrics"]["alignment_error_mm"]["mean"], 3.0)
        self.assertIn("1/2 (50.0%)", markdown)
        self.assertIn("servo_timeout", markdown)
        # Summary remains a JSON artifact, including rows and grouped stats.
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
