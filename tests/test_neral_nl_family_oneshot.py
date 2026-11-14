"""RED: family-bound 1-shot prompts use no foreign family or case labels.

Only historical development examples; no prospective 100 or Ollama inference.
"""
import json
import unittest
from unittest.mock import patch

from neral_nl_schema_bridge import (
    extraction_prompt_for_question, assess_dual_extraction,
    V2_FEW_SHOT_EXAMPLES,
)
from neral_nl_verified_route import strict_source_certificate
from scripts.neral_nl_schema_probe import old_case_question
from scripts.neral_nl_schema_diagnostics import collect_live_diagnostics


class FamilyPromptContracts(unittest.TestCase):
    def test_only_one_family_example_both_models(self):
        prompts = {
            "graph": old_case_question("logic-02"),
            "binary": old_case_question("logic-06"),
            "pin": old_case_question("21453"),
        }
        samples = V2_FEW_SHOT_EXAMPLES
        for case, question in prompts.items():
            expected = {"graph": 0, "binary": 1, "pin": 2}[case]
            for is_base in (False, True):
                with self.subTest(case=case, base=is_base):
                    prompt = extraction_prompt_for_question(question, is_base=is_base)
                    self.assertEqual(prompt.count("BEISPIEL"), 1)
                    self.assertLess(len(prompt), 2800)
                    self.assertIn(samples[expected]["task"], prompt)
                    self.assertIn(json.dumps(samples[expected]["schema"],
                                      ensure_ascii=False, separators=(",", ":")), prompt)
                    for i, example in enumerate(samples):
                        if i != expected:
                            self.assertNotIn(example["task"], prompt)
                            self.assertNotIn(json.dumps(example["schema"],
                                              ensure_ascii=False, separators=(",", ":")), prompt)
                    self.assertIn('"variables"', prompt)
                    self.assertIn('"constraints"', prompt)
        self.assertNotEqual(extraction_prompt_for_question(prompts["pin"], is_base=False),
                            extraction_prompt_for_question(prompts["pin"], is_base=True))

    def test_pin_exact_position_idioms_are_explicit_and_not_gold_answer(self):
        prompt = extraction_prompt_for_question(old_case_question("21453"))
        for s in ("p1..p5", "all_different", "pY - pX >= 2",
                  "pX - pY == 2", "pY - pX == 2",
                  "pX == pY + 1", "pX > 1", "pX < 5"):
            self.assertIn(s, prompt)
        self.assertNotIn("21453", prompt)

    def test_new_family_sizes_use_same_prompt_no_case_specific_branch(self):
        for question, expected in (
            ("Color vertices 0..3 with red, green or blue. "
             "Vertex 0 is red, and endpoints of edges [(0, 1)] must differ. "
             "How many valid colorings?", "graph"),
            ("Count binary strings of length 7 with exactly 2 ones, "
             "first two bits opposite and no two ones exactly two positions apart.", "binary"),
            ("Ein dreistelliger Code verwendet 1, 2 und 3 genau einmal. "
             "Die 1 steht unmittelbar rechts von der 2.", "pin"),
        ):
            with self.subTest(expected=expected):
                result = extraction_prompt_for_question(question)
                self.assertEqual(result.count("BEISPIEL"), 1)
                self.assertIn({"graph": "Color vertices",
                               "binary": "Count binary strings",
                               "pin": "Ein fünfstelliger Code"}[expected], result)

    def test_unknown_source_has_no_oneshot_no_auto_promotion(self):
        question = "Zähle beliebige Kombinationen auf unbekannter Domäne."
        prompt = extraction_prompt_for_question(question)
        self.assertEqual(prompt.count("BEISPIEL"), 0)
        self.assertNotIn("21453", prompt)

    def test_position_source_specific_clause_obligations_are_generalized(self):
        source = ("Ein fünfstelliger Code verwendet 1, 2, 3, 4 und 5 genau einmal. "
                  "Die 1 steht vor der 3, nicht direkt davor. "
                  "Zwischen der 2 und 4 steht genau eine Ziffer. "
                  "Die 5 steht unmittelbar rechts von der 1. "
                  "Die 2 steht nicht am Rand. Bestimme alle gültigen Codes.")
        prompt = extraction_prompt_for_question(source)
        self.assertIn("p3 >= p1 + 2", prompt)
        self.assertIn("p2 == p4 + 2 OR p4 == p2 + 2", prompt)
        self.assertIn("p5 == p1 + 1", prompt)
        self.assertIn("p2 > 1 UND p2 < 5", prompt)
        self.assertIn("all_different", prompt)
        self.assertEqual(prompt.count("BEISPIEL"), 1)
        self.assertNotIn("21453", prompt)

    def test_binary_source_specific_pairs_are_complete_for_length_six(self):
        question = ("Count binary strings of length 6 with exactly 3 ones, "
                    "first two bits opposite and no two ones exactly two positions apart.")
        prompt = extraction_prompt_for_question(question)
        self.assertIn("B0/B2", prompt)
        self.assertIn("B1/B3", prompt)
        self.assertIn("B2/B4", prompt)
        self.assertIn("B3/B5", prompt)
        self.assertNotIn("B4/B6", prompt)
        self.assertEqual(prompt.count("BEISPIEL"), 1)

    def test_dual_extraction_uses_selected_prompts(self):
        q = old_case_question("logic-06")
        doc = strict_source_certificate(q)
        seen = []
        def extractor(turns):
            seen.append(turns)
            return json.dumps(doc)
        result = assess_dual_extraction(q, extractor, extractor_copy(extractor))
        self.assertEqual(result["status"], "consensus_unverified")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0]["content"],
                         extraction_prompt_for_question(q, is_base=False))
        self.assertEqual(seen[1][0]["content"],
                         extraction_prompt_for_question(q, is_base=True))

    def test_local_diagnostics_use_identical_family_routing(self):
        seen = []
        q = old_case_question("21453")
        canonical = strict_source_certificate(q)
        def fake_transport(payload):
            seen.append(payload)
            return {"done_reason": "stop",
                    "message": {"content": json.dumps(canonical)}}
        with patch("scripts.neral_nl_schema_diagnostics._local_transport",
                   side_effect=fake_transport):
            result = collect_live_diagnostics("21453", "adapter:test", "base:test")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["messages"][0]["content"],
                         extraction_prompt_for_question(q, is_base=False))
        self.assertEqual(seen[1]["messages"][0]["content"],
                         extraction_prompt_for_question(q, is_base=True))
        self.assertEqual(result["models"]["base"]["candidate_count"], 1)
        self.assertFalse(result["safe_to_promote"])


def extractor_copy(fn):
    def other(turns):
        return fn(turns)
    return other


if __name__ == "__main__":
    unittest.main()
