"""RED/GREEN: numeric *meaning* can leak across data splits without text duplicates."""
import json
import unittest

from frontier350.arctus350_exact_math_curriculum import make_splits as make_v1
from frontier350.arctus350_balanced_arithmetic_v2 import make_splits as make_v2
from frontier350.arctus350_carry_curriculum_v3 import make_splits as make_v3
from frontier350.arctus350_math_semantic_audit import (
    audit_math_splits, require_disjoint_math_pairs,
)


class ArithmeticSemanticLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1=make_v1()
        cls.v2=make_v2()
        cls.v3=make_v3()

    def test_detects_real_v1_commutative_split_leakage_beyond_exact_prompt_gate(self):
        from frontier350.chat350_data import validate_prompt_holdouts
        # Existing lexical checker does NOT detect the underlying unordered-pair overlap.
        validate_prompt_holdouts(self.v1)
        result=audit_math_splits(self.v1)
        self.assertGreater(result["overlap"]["unique_unordered_pairs_any_split"],0)
        self.assertGreater(result["overlap"]["train_with_holdout"],0)
        self.assertFalse(result["gate"]["disjoint_math_pairs"])
        with self.assertRaisesRegex(ValueError,"semantic arithmetic"):
            require_disjoint_math_pairs(result)

    def test_v2_balanced_and_v3_carry_have_no_unordered_operand_leakage(self):
        for rows in (self.v2,self.v3):
            with self.subTest(rows=sum(map(len,rows.values()))):
                report=audit_math_splits(rows)
                self.assertTrue(report["gate"]["disjoint_math_pairs"])
                self.assertEqual(report["overlap"]["unique_unordered_pairs_any_split"],0)
                self.assertEqual(report["gate"]["unverified_generated_math_rows"],0)
                require_disjoint_math_pairs(report)

    def test_rephrased_bilingual_pair_is_detected_without_identical_prompt(self):
        source=self.v3["train"]
        first=next(x for x in source if x["task_type"]=="trace" and x["language"]=="de")
        family=first["id"].rsplit("-",2)[0]
        second=next(x for x in source if x["task_type"]=="direct" and x["language"]=="en"
                    and x["id"].rsplit("-",2)[0]==family)
        assert first["messages"][0]["content"]!=second["messages"][0]["content"]
        # Different template and language, same arithmetic fact.
        report=audit_math_splits({"train":[first],"val":[second],"test":[self.v3["test"][0]]})
        self.assertEqual(report["overlap"]["train_val"],1)
        self.assertEqual(report["overlap"]["train_with_holdout"],1)
        self.assertEqual(report["overlap"]["unique_unordered_pairs_any_split"],1)
        self.assertFalse(report["gate"]["disjoint_math_pairs"])

    def test_invalid_or_poisoned_generated_row_fails_closed(self):
        original=next(x for x in self.v3["train"] if x["task_type"]=="direct")
        poisoned={**original,"messages":[dict(x) for x in original["messages"]]}
        poisoned["messages"][1]["content"]="999"
        with self.assertRaisesRegex(ValueError,"unverifiable generated math"):
            audit_math_splits({"train":[poisoned],"val":[self.v3["val"][0]],"test":[self.v3["test"][0]]})
        poisoned={**original,"license":"unknown"}
        with self.assertRaisesRegex(ValueError,"unverifiable generated math"):
            audit_math_splits({"train":[poisoned],"val":[self.v3["val"][0]],"test":[self.v3["test"][0]]})

    def test_non_generated_math_is_explicitly_unmeasured_not_misreported_clean(self):
        external={"id":"SECRET-EXTERNAL", "language":"en","domain":"math","source":"unreviewed",
                  "license":"not reviewed",
                  "messages":[{"role":"user","content":"PRIVATE_PROMPT 2 + 3"},
                              {"role":"assistant","content":"PRIVATE_ANSWER"}]}
        report=audit_math_splits({"train":[external],"val":[external],"test":[external]})
        self.assertEqual(report["gate"]["unverified_generated_math_rows"],0)
        self.assertEqual(report["gate"]["opaque_math_rows"],3)
        self.assertFalse(report["gate"]["ready_for_release"])
        self.assertNotIn("SECRET-EXTERNAL",json.dumps(report))
        self.assertNotIn("PRIVATE_PROMPT",json.dumps(report))
        self.assertNotIn("PRIVATE_ANSWER",json.dumps(report))

    def test_invalid_splits_and_schema_fail_closed(self):
        for bad in (
            {"train":[],"val":[]},
            {"train":[],"val":[],"test":[]},
            {"train":(),"val":[],"test":[]},
            {"train":[42],"val":[{}],"test":[{}]},
        ):
            with self.subTest(bad=repr(bad)[:30]),self.assertRaises(ValueError):
                audit_math_splits(bad)

    def test_report_contains_only_aggregates_not_raw_prompts_or_ids(self):
        r=audit_math_splits(self.v3)
        rendered=json.dumps(r,ensure_ascii=False)
        first=self.v3["train"][0]
        self.assertNotIn(first["id"],rendered)
        self.assertNotIn(first["messages"][0]["content"],rendered)
        self.assertNotIn(first["messages"][1]["content"],rendered)
        self.assertIn("not_an_independent_frozen_benchmark",rendered)
        self.assertFalse(r["gate"]["ready_for_release"])


if __name__=="__main__":
    unittest.main()
