"""READ-ONLY regression audit: test non-leaking v6/v5 response diagnosis."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_v6_fresh_dev_eval import build_cases,score_three_way
from scripts.neral_v6_dev72_regression_audit import (
    exploratory_terminal, classify_ungraded, diagnose_saved, audit_private_files,
)


def fixtures():
    cases=build_cases()
    variants={}
    for variant in ("qwen_fewshot","neral_v5","neral_v6"):
        variants[variant]=[{"id":c["id"],"text":"FINAL_ANSWER: "+c["answer"],
                            "complete":True,"new_tokens":7} for c in cases]
    # One v6 valid-to-invalid regression recoverable with strict final punctuation.
    variants["neral_v6"][0]["text"]="CHECK: calculation here\nFINAL_ANSWER: "+cases[0]["answer"]+"."
    # One v6 real numerical regression remains wrong, not a formatting artifact.
    variants["neral_v6"][1]["text"]="FINAL_ANSWER: 987654"
    # Two ambiguous outputs remain ambiguous no matter where numbers appear.
    variants["neral_v6"][2]["text"]="First 10 then 11\nPerhaps "+cases[2]["answer"]
    variants["neral_v6"][3]["text"]="FINAL_ANSWER: "+cases[3]["answer"]+"\nFINAL_ANSWER: 99"
    return cases,variants


class RegressionAuditTests(unittest.TestCase):
    def test_conservative_recovery_only_final_unambiguous(self):
        for raw in (
            "Work: 7 + 5\nFINAL_ANSWER: 12.",
            "After considering possibilities:\n\\boxed{12}",
            "Work: ...\nDas Ergebnis ist **12**.",
            "FINAL: 12!",
            "Reasoning\nAnswer: 12.",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(exploratory_terminal(raw),12)
        for raw in (
            "Step 1: 12\nStep 2: 17",
            "Work ends in 12",
            "FINAL_ANSWER: 12\nFINAL_ANSWER: 13",
            "FINAL_ANSWER: 12\nsecond guess",
            "Choice 12 or 13",
            "\\boxed{12} and \\boxed{13}",
            "12",
            "",
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(exploratory_terminal(raw))

    def test_regression_partition_and_v6_format_vs_semantic(self):
        cases, variants=fixtures()
        original=score_three_way(cases,variants)
        aggregate, samples=diagnose_saved(cases,variants,original,review_limit=20)
        self.assertEqual(aggregate["original_score"]["neral_v6"]["correct"],68)
        self.assertEqual(aggregate["original_score"]["neral_v6"]["ungraded_completed"],3)
        self.assertEqual(aggregate["exploratory_terminal_recovery"]["neral_v6"]["correct"],1)
        self.assertEqual(aggregate["exploratory_terminal_recovery"]["neral_v6"]["wrong"],0)
        self.assertEqual(aggregate["paired_v6_v5"]["format_recoverable_regressions"],1)
        self.assertEqual(aggregate["paired_v6_v5"]["true_observed_wrong_regressions"],1)
        self.assertEqual(aggregate["paired_v6_v5"]["unresolved_ungraded_regressions"],2)
        self.assertEqual(aggregate["ungraded_categories"]["neral_v6"]["multiple_final_markers"],1)
        self.assertEqual(aggregate["ungraded_categories"]["neral_v6"]["no_unambiguous_terminal"],1)
        self.assertEqual(len([s for s in samples if s["variant"]=="neral_v6"]),20)
        self.assertFalse(aggregate["release_validated"])
        self.assertFalse(aggregate["actual_generalization_improved"])
        self.assertNotIn(cases[0]["prompt"],json.dumps(aggregate))
        self.assertIn(cases[0]["prompt"],json.dumps(samples,ensure_ascii=False))

    def test_never_rewrite_original_or_upload_private_raw(self):
        cases, variants=fixtures()
        original=score_three_way(cases,variants)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/"cases.jsonl").write_text("".join(json.dumps(c,ensure_ascii=False)+"\n" for c in cases),encoding="utf8")
            manifest={
                "schema":1,"suite_id":"NERAL-v6-structural-new-dev-72-20261107-v1",
                "case_sha256":original["suite_sha256"],
                "v5_adapter_sha256":"2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5",
                "v6_adapter_sha256":"8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd",
                "count":72,
            }
            (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf8")
            original.update({
                "v5_adapter_sha256":manifest["v5_adapter_sha256"],
                "v6_adapter_sha256":manifest["v6_adapter_sha256"],
            })
            (root/"summary.json").write_text(json.dumps(original),encoding="utf8")
            for name, rows in variants.items():
                (root/(name+".jsonl")).write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf8")
            aggregate, samples=audit_private_files(root,root/"audit-agg.json",root/"audit-private.jsonl")
            self.assertEqual(aggregate["status"],"NERAL_V6_DEV72_CPU_POSTHOC_AUDIT")
            self.assertEqual(len(samples),60)
            self.assertFalse(aggregate["release_validated"])
            self.assertEqual(aggregate["original_score"]["neral_v5"]["correct"],72)
            self.assertEqual(aggregate["original_score"]["neral_v6"]["correct"],68)
            self.assertNotIn(cases[0]["prompt"],(root/"audit-agg.json").read_text(encoding="utf8"))
            self.assertIn(cases[0]["prompt"],(root/"audit-private.jsonl").read_text(encoding="utf8"))
            with self.assertRaises(ValueError):
                audit_private_files(root,root/"audit-agg.json",root/"audit-private.jsonl")

    def test_saved_report_without_new_partitions_keeps_original_counts(self):
        cases, variants=fixtures()
        fresh=score_three_way(cases,variants)
        locked=json.loads(json.dumps(fresh))
        for row in locked["scores"].values():
            for field in ("strict_correct","wrong_readable_number","unreadable"):
                row.pop(field)
        aggregate,_=diagnose_saved(cases,variants,locked,review_limit=0)
        self.assertEqual(aggregate["original_score"]["neral_v6"]["correct"],68)
        locked["scores"]["neral_v6"]["correct"]=0
        with self.assertRaises(ValueError):
            diagnose_saved(cases,variants,locked,review_limit=0)


if __name__=="__main__":
    unittest.main()
