"""CPU contract for real 24-case evaluation harness, no Ollama or CUDA."""
from __future__ import annotations
from unittest import TestCase
from unittest.mock import patch
import copy
import json
from pathlib import Path

from scripts.neral_qlora_holdout_live import (
    AUTH, BASE_MODEL, run_case, summarize, extract_count, request,
)
from scripts.neral_qlora_holdout_eval import load_holdout

HOLDOUT=Path(__file__).resolve().parents[1]/"datasets/neral_qwen35_holdout_v1.jsonl"


class HoldoutLiveCpu(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=load_holdout(HOLDOUT)

    def test_oracle_schema_scores_exactly_24_correct(self):
        def answer(turns,seed,repair=False):
            q=turns[-1]["content"]
            match=next(v for v in self.rows if v["question"]==q)
            return json.dumps({
                "kind":"formal","complete":True,
                "schema":match["schema"],"evidence":match["evidence"],
            },separators=(",",":"))
        with patch("scripts.neral_qlora_holdout_live.query_structured",
                   side_effect=lambda model,turns,seed,repair=False:
                   answer(turns,seed,repair)):
            observed=[run_case(case,BASE_MODEL,50+i)
                      for i,case in enumerate(self.rows)]
        summary=summarize(self.rows,observed,BASE_MODEL,"fake_test_digest")
        self.assertEqual(summary["raw_correct"],24)
        self.assertEqual(summary["controller_final_correct"],24)
        self.assertEqual(summary["controller_formal_correct"],24)
        self.assertEqual(summary["repair_attempted"],0)
        self.assertEqual(summary["repair_correct"],0)

    def test_repaired_schema_counts_only_if_final_value_correct(self):
        case=self.rows[0]
        good=json.dumps({"kind":"formal","complete":True,
            "schema":case["schema"],"evidence":case["evidence"]},
            separators=(",",":"))
        bad=json.loads(good)
        bad["schema"]["rules"].append({
            "left":bad["schema"]["variables"][0]["name"],
            "op":"==","value":1,
        })
        sequence=[json.dumps(bad,separators=(",",":")),good]
        with patch("scripts.neral_qlora_holdout_live.query_structured",
                   side_effect=lambda *a,**kw:sequence.pop(0)):
            r=run_case(case,BASE_MODEL,18)
        self.assertFalse(r["raw_correct"])
        self.assertTrue(r["repair_attempted"])
        self.assertTrue(r["repair_accepted"])
        self.assertTrue(r["repair_correct"])
        self.assertTrue(r["controller_formal_correct"])

    def test_accepted_incomplete_semantics_is_not_counted_as_correct(self):
        # Structurally valid and grounded but one condition omitted.
        case=self.rows[0]
        partial=copy.deepcopy(case["schema"])
        partial["bool_rules"]=partial["bool_rules"][:1]
        candidate=json.dumps({"kind":"formal","complete":True,
            "schema":partial,"evidence":case["evidence"]},
            separators=(",",":"))
        count,reason=extract_count(case["question"],candidate)
        if count is not None:
            self.assertNotEqual(count,case["expected_count"])
            self.assertEqual(reason,"accepted")

    def test_base_digest_format_is_normalized_without_weakening_pin(self):
        from scripts.neral_qlora_holdout_live import require_base_digest
        from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
        expected=ORIGINAL_DIGESTS[BASE_MODEL]
        for remote in (expected,expected.removeprefix("sha256:")):
            with self.subTest(style=remote.startswith("sha256:")), patch(
                    "scripts.neral_qlora_holdout_live.request",
                    return_value={"models":[{"name":BASE_MODEL,"digest":remote}]}):
                self.assertEqual(require_base_digest(),expected)
        with patch("scripts.neral_qlora_holdout_live.request",
                   return_value={"models":[{"name":BASE_MODEL,"digest":"0"*64}]}):
            with self.assertRaises(RuntimeError):
                require_base_digest()

    def test_local_api_rejects_other_hosts_and_unlisted_paths(self):
        with self.assertRaises(ValueError):
            request("POST","/api/generate")
        with self.assertRaises(ValueError):
            request("DELETE","/api/chat")
