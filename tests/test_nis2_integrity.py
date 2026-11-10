"""NIS2 score integrity regression: never award credit for missing/non-finite evidence."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from neral_intelligence_score_v2 import (
    WEIGHTS, score_from_components, generation_health, nis2_evidence_status,
)


class NIS2IntegrityTests(unittest.TestCase):
    def test_nonfinite_component_cannot_silently_become_perfect_score(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            metrics = {k: 50.0 for k in WEIGHTS}
            metrics["math_generalization"] = value
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                score_from_components(metrics)

    def test_invalid_component_types_are_rejected(self):
        metrics = {k: 50.0 for k in WEIGHTS}
        for value in (None, True, "nan", "inf", {}, []):
            metrics["language_knowledge"] = value
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                score_from_components(metrics)

    def test_healthy_inputs_preserve_original_formula(self):
        p = {k: 63.25 for k in WEIGHTS}
        report = score_from_components(p)
        self.assertAlmostEqual(report["score"], 63.25)
        self.assertEqual(report["coverage_pct"], 100.0)
        self.assertEqual(report["version"], "NIS-2.0")

    def test_generation_health_rejects_nonfinite_signals(self):
        baseline = {"free_eos_pct":60.0,"free_repeated_trigram_pct":0.0,"utf8_pct":100.0}
        for signal in baseline:
            for invalid in (float("nan"), float("inf"), float("-inf")):
                row = dict(baseline)
                row[signal] = invalid
                with self.subTest(signal=signal,invalid=repr(invalid)), self.assertRaises(ValueError):
                    generation_health(row)

    def test_incomplete_evidence_is_explicitly_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            status = nis2_evidence_status(directory)
            self.assertEqual(status["status"], "not_measured")
            self.assertIsNone(status["nis2_score"])
            self.assertGreater(len(status["missing_files"]),0)
            self.assertIn("nis2_chat.json", status["missing_files"])
            self.assertIn("rawmath.json", status["missing_files"])
            self.assertFalse(status["can_report_nis2"])

    def test_complete_file_names_without_valid_benchmark_provenance_stay_unscored(self):
        names = ("free.json","v3.json","v4.json","rawmath.json","targeted.json","v5.json","nis2_chat.json")
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                (Path(directory)/name).write_text("{}")
            status = nis2_evidence_status(directory)
            self.assertFalse(status["can_report_nis2"])
            self.assertIsNone(status["nis2_score"])
            self.assertIn("nis2_chat", status["invalid_evidence"])
            (Path(directory)/"nis2_chat.json").write_text(json.dumps({
                "benchmark":"nis2-chat-core-v1", "foreign_model_or_tool_assistance": True,
                "checkpoint_sha256":"f"*64, "scores":{"instruction_following_pct":100}
            }))
            status = nis2_evidence_status(directory)
            self.assertFalse(status["can_report_nis2"])
            self.assertIn("nis2_chat", status["invalid_evidence"])


    def test_mixed_checkpoint_evidence_cannot_produce_nis2(self):
        from neral_intelligence_score_v2 import components_from_reports
        with tempfile.TemporaryDirectory() as directory:
            for name in ("free.json","v3.json","v4.json","rawmath.json","targeted.json","v5.json"):
                value={"checkpoint_sha256":"a"*64}
                if name=="v5.json":
                    value["checkpoint_sha256"]="b"*64
                (Path(directory)/name).write_text(json.dumps(value))
            (Path(directory)/"nis2_chat.json").write_text(json.dumps({
                "benchmark":"nis2-chat-core-v1",
                "checkpoint_sha256":"a"*64,
                "suite_sha256":"c"*64,
                "foreign_model_or_tool_assistance":False,
                "used_for_checkpoint_selection":False,
                "scores":{
                    "instruction_following_pct":50.0,
                    "dialogue_state_pct":50.0,
                    "truthfulness_pct":50.0,
                    "robustness_pct":50.0,
                    "compositional_reasoning_pct":50.0,
                },
            }))
            with self.assertRaises(ValueError):
                components_from_reports(directory)
            status=nis2_evidence_status(directory)
            self.assertFalse(status["can_report_nis2"])
            self.assertIsNone(status["nis2_score"])
            self.assertIn("checkpoint_lineage",status["invalid_evidence"])


if __name__=="__main__":
    unittest.main()
