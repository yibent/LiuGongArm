from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
from benchmark_franka_pick_place import default_matrix, validate_matrix, outcome
from mr_liu.place.fixtures import fixture_boxes
from mr_liu.place.contracts import PlaceRequest, PlaceError
from mr_liu.place.geometry import support_evidence
from test_fine_place import Rig


class ComparisonTests(unittest.TestCase):
    def test_cross_view_handoff_margin_is_narrow_and_explicit(self):
        from mr_liu.perception.semantic_target import SemanticFlowTarget
        self.assertEqual(SemanticFlowTarget.HANDOFF_MAX_COLOR_ERROR, .20)
        self.assertLess(SemanticFlowTarget.HANDOFF_MAX_COLOR_ERROR, .25)
    def test_full_success_needs_both_reports_release_verification_and_process(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run/"place_report.json").write_text(json.dumps({"result":dict(success=True,released=True,verified=True)}))
            self.assertFalse(outcome(run,0)["complete_success"])
            (run/"report.json").write_text(json.dumps({"result":{"success":True}}))
            self.assertTrue(outcome(run,0)["complete_success"])
            self.assertFalse(outcome(run,2)["complete_success"])
            self.assertFalse(outcome(run,0,timed_out=True)["complete_success"])
            (run/"place_report.json").write_text(json.dumps({"result":dict(success=True,released=True,verified=False)}))
            self.assertFalse(outcome(run,0)["complete_success"])

    def test_manifest_is_reproducible_and_cannot_reuse_output_names(self):
        matrix=default_matrix()
        self.assertEqual(matrix,default_matrix())
        validate_matrix(matrix)
        matrix["cases"].append(matrix["cases"][0])
        with self.assertRaises(ValueError):
            validate_matrix(matrix)

    def test_complex_fixture_relations_are_validated_as_distinct_tasks(self):
        matrix=default_matrix()
        base=matrix["cases"][0].copy()
        for fixture,relation in (("socket","insert"),("rack","hang")):
            case=base.copy();case.update(name="complex_"+relation,fixture=fixture,relation=relation,
                                         destination="blue "+fixture)
            matrix["cases"].append(case)
        validate_matrix(matrix)
        bad=base.copy();bad.update(name="bad_complex",fixture="socket",relation="inside")
        matrix["cases"].append(bad)
        with self.assertRaises(ValueError):
            validate_matrix(matrix)

    def test_complex_requests_cannot_use_horizontal_release(self):
        for relation in ("insert","hang"):
            rig=Rig()
            result=rig.node().execute(PlaceRequest("blue fixture",relation),rig.held)
            self.assertEqual(result.failure,"placement_relation_not_executable")
            self.assertFalse(result.success)
            self.assertFalse(rig.opened)
            self.assertEqual(rig.moves,0)
            with self.assertRaises(PlaceError):
                support_evidence(None,None,relation=relation)

    def test_socket_has_real_empty_channel_and_rack_has_hook(self):
        parts=fixture_boxes("socket")
        walls=[b for b in parts if b.name.startswith("Wall")]
        self.assertEqual(len(walls),4)
        # The centreline above the floor must not intersect any wall.
        for b in walls:
            self.assertFalse(all(abs(p-c) < s/2 for p,c,s in zip((.25,-.198,1.075),b.position,b.size)))
        self.assertEqual({b.name for b in fixture_boxes("rack")},{"Floor","Post","Hook","HookTip"})
        self.assertEqual(len(fixture_boxes("tray",clutter=2)),7)
        with self.assertRaises(ValueError):
            fixture_boxes("unknown")


if __name__ == "__main__":
    unittest.main()
