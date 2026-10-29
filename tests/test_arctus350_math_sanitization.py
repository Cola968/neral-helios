"""RED/GREEN: derive a clean copy of v1, preserve TEST, do not modify originals."""
import copy
import json
import unittest

from frontier350.arctus350_exact_math_curriculum import make_splits as make_v1
from frontier350.arctus350_balanced_arithmetic_v2 import make_splits as make_v2
from frontier350.arctus350_carry_curriculum_v3 import make_splits as make_v3
from frontier350.arctus350_math_semantic_audit import (
    audit_math_splits, sanitize_math_splits, require_disjoint_math_pairs,
)


class MathSanitizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1=make_v1()
        cls.v2=make_v2()
        cls.v3=make_v3()

    def test_v1_semantic_leaks_are_repaired_without_editing_source(self):
        original=copy.deepcopy(self.v1)
        clean,stats=sanitize_math_splits(self.v1)
        self.assertEqual(self.v1,original,"input is immutable")
        self.assertIsNot(clean["train"],self.v1["train"])
        self.assertIsNot(clean["test"][0],self.v1["test"][0])
        self.assertEqual(clean["test"],self.v1["test"],"TEST must remain frozen")
        self.assertEqual(stats["dropped"]["test"],0)
        self.assertGreater(stats["dropped"]["val"]+stats["dropped"]["train"],0)
        self.assertEqual(stats["retained_counts"]["test"],len(self.v1["test"]))
        self.assertEqual(stats["original_counts"]["train"],len(self.v1["train"]))
        self.assertEqual(stats["immutable_test_rows"],True)
        report=audit_math_splits(clean)
        self.assertEqual(report["overlap"]["unique_unordered_pairs_any_split"],0)
        self.assertTrue(report["gate"]["safe_for_controlled_sft"])
        require_disjoint_math_pairs(report)

    def test_newer_curricula_are_not_needlessly_pruned(self):
        for source in (self.v2,self.v3):
            with self.subTest(rows=len(source["train"])):
                clean,stats=sanitize_math_splits(source)
                self.assertEqual(stats["dropped"],{"train":0,"val":0,"test":0})
                self.assertEqual(clean,source)
                require_disjoint_math_pairs(audit_math_splits(clean))

    def test_never_silently_repair_unreviewed_math(self):
        external={"id":"SECRET_ID","language":"en","domain":"math","source":"external",
                  "license":"unknown","messages":[{"role":"user","content":"CONFIDENTIAL question"},
                                                {"role":"assistant","content":"SECRET_ANSWER"}]}
        with self.assertRaisesRegex(ValueError,"opaque math"):
            sanitize_math_splits({"train":[external],"val":[external],"test":[external]})

    def test_stats_contain_no_raw_prompts_or_ids(self):
        _,stats=sanitize_math_splits(self.v1)
        dumped=json.dumps(stats)
        first=self.v1["train"][0]
        self.assertNotIn(first["id"],dumped)
        self.assertNotIn(first["messages"][0]["content"],dumped)
        self.assertIn("TEST",stats["priority"])
        self.assertFalse(stats["training_started"])
        self.assertFalse(stats["release_allowed"])

    def test_invalid_split_or_all_removed_fails_closed(self):
        with self.assertRaises(ValueError):
            sanitize_math_splits({"train":[],"val":[],"test":[]})
        x=next(r for r in self.v3["train"] if r["task_type"]=="direct")
        with self.assertRaisesRegex(ValueError,"empty"):
            sanitize_math_splits({"train":[x],"val":[x],"test":[x]})


if __name__=="__main__":
    unittest.main()
