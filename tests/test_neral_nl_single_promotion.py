"""CPU-only source-proven sequential selection and bounded model ASTs."""
import copy
import json
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError
from neral_nl_schema_format import get_v2_ollama_format
from neral_nl_schema_bridge import coverage_gate, _parse_candidate, extraction_prompt_for_question
from neral_nl_verified_route import strict_source_certificate, run_guarded_nl_bridge
from scripts.neral_nl_schema_probe import old_case_question


def turns(q):
    return [{"role":"system","content":"NERAL"},{"role":"user","content":q}]


class SingleSourcePromotion(unittest.TestCase):
    def test_correct_qlora_never_invokes_wrong_base(self):
        q=old_case_question("21453")
        good=strict_source_certificate(q)
        def forbidden(_): self.fail("No base model call after QLoRA source proof")
        result=run_guarded_nl_bridge(turns(q),lambda _:json.dumps(good),forbidden,forbidden)
        self.assertTrue(result["safe_to_promote"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["text"],"1")
        self.assertEqual(result["calls_used"],1)
        self.assertEqual(result["selected_extractor"],"qlora")
        self.assertEqual(result["proof_backend"],"z3_unsat")

    def test_base_only_after_failed_qlora_proof(self):
        q=old_case_question("logic-06")
        good=strict_source_certificate(q)
        bad=copy.deepcopy(good); bad["constraints"].pop()
        calls=[]
        def first(_): calls.append("qlora"); return json.dumps(bad)
        def second(_): calls.append("base"); return json.dumps(good)
        result=run_guarded_nl_bridge(turns(q),first,second,lambda _:self.fail("No raw fallback"))
        self.assertEqual(calls,["qlora","base"])
        self.assertEqual(result["selected_extractor"],"base")
        self.assertEqual(result["text"],"4")
        self.assertEqual(result["calls_used"],2)
        self.assertTrue(result["safe_to_promote"])

    def test_two_failures_preserve_original_raw_fallback(self):
        q=old_case_question("logic-06")
        bad=strict_source_certificate(q); bad["constraints"].pop()
        seen=[]
        def raw(original): seen.append(original); return "ORIGINAL_BASE"
        result=run_guarded_nl_bridge(turns(q),lambda _:json.dumps(bad),lambda _:json.dumps(bad),raw)
        self.assertEqual(seen,[turns(q)])
        self.assertEqual(result["text"],"ORIGINAL_BASE")
        self.assertEqual(result["calls_used"],3)
        self.assertFalse(result["safe_to_promote"])

    def test_heuristic_coverage_is_not_full_source_proof(self):
        q=old_case_question("logic-02")
        bad=strict_source_certificate(q)
        taut={"compare":{"left":{"var":"C0"},"op":"==","right":{"var":"C0"}}}
        bad["constraints"]=[{"or":[{"and":bad["constraints"]},taut]}]
        self.assertTrue(coverage_gate(q,bad)["passed"])
        result=run_guarded_nl_bridge(turns(q),lambda _:json.dumps(bad),lambda _:json.dumps(bad),lambda _:"BASE")
        self.assertFalse(result["verified"])
        self.assertFalse(result["safe_to_promote"])
        self.assertEqual(result["text"],"BASE")
        self.assertEqual(result["extraction_attempts"][0]["proof_backend"],"z3_counterexample")

    def test_unknown_extra_source_clause_cannot_promote(self):
        q=old_case_question("21453")+" Außerdem steht die Ziffer 1 ganz links."
        def forbidden(_): self.fail("Do not translate unsupported full source")
        result=run_guarded_nl_bridge(turns(q),forbidden,forbidden,lambda _:"BASE")
        self.assertFalse(result["safe_to_promote"])
        self.assertEqual(result["calls_used"],1)

    def test_no_unknown_or_enumeration_proof_is_promoted(self):
        q=old_case_question("logic-06"); good=strict_source_certificate(q)
        for proof in ((False,"z3_unknown"),(True,"enumeration_exhaustive"),(False,"z3_error")):
            with self.subTest(proof=proof),patch("neral_nl_verified_route._equivalence",return_value=proof):
                result=run_guarded_nl_bridge(turns(q),lambda _:json.dumps(good),lambda _:json.dumps(good),lambda _:"BASE")
                self.assertFalse(result["safe_to_promote"])
                self.assertEqual(result["text"],"BASE")

    def test_malformed_qlora_may_use_independently_verified_base(self):
        q=old_case_question("21453"); good=strict_source_certificate(q)
        result=run_guarded_nl_bridge(turns(q),lambda _:"{",lambda _:json.dumps(good),lambda _:self.fail("No fallback"))
        self.assertEqual(result["selected_extractor"],"base")
        self.assertTrue(result["safe_to_promote"])


class ModelArrayBudget(unittest.TestCase):
    def test_generic_array_is_bounded_and_unique(self):
        spec=get_v2_ollama_format()["properties"]["constraints"]
        self.assertEqual(spec["maxItems"],12)
        self.assertTrue(spec["uniqueItems"])

    def test_binary_requires_only_real_source_clauses(self):
        for n in (4,6,7,10):
            q=f"Count binary strings of length {n} with exactly 2 ones, first two bits opposite and no two ones exactly two positions apart."
            spec=get_v2_ollama_format(q)["properties"]["constraints"]
            self.assertEqual((spec["minItems"],spec["maxItems"]),(n,n))
        q=old_case_question("logic-06")
        Draft202012Validator(get_v2_ollama_format(q)).validate(strict_source_certificate(q))

    def test_full_pin_has_six_real_source_slots(self):
        q=old_case_question("21453")
        spec=get_v2_ollama_format(q)["properties"]["constraints"]
        self.assertEqual((spec["minItems"],spec["maxItems"]),(6,6))
        Draft202012Validator(get_v2_ollama_format(q)).validate(strict_source_certificate(q))
        partial="Ein fünfstelliger Code verwendet 1, 2, 3, 4 und 5 genau einmal."
        self.assertEqual(get_v2_ollama_format(partial)["properties"]["constraints"]["maxItems"],12)

    def test_runtime_rejects_source_budget_padding_even_if_equivalent(self):
        q=old_case_question("logic-06"); bad=strict_source_certificate(q)
        bad["constraints"].append({"compare":{"left":{"var":"B0"},"op":"==","right":{"var":"B0"}}})
        result=run_guarded_nl_bridge(turns(q),lambda _:json.dumps(bad),lambda _:json.dumps(bad),lambda _:"BASE")
        self.assertFalse(result["safe_to_promote"])
        self.assertEqual(result["extraction_attempts"][0]["status"],"source_clause_budget_rejected")

    def test_binary_rejects_padding_and_missing_clauses(self):
        q=old_case_question("logic-06"); good=strict_source_certificate(q)
        validator=Draft202012Validator(get_v2_ollama_format(q))
        for delta in (-1,1):
            bad=copy.deepcopy(good)
            if delta<0: bad["constraints"].pop()
            else: bad["constraints"].append(copy.deepcopy(bad["constraints"][0]))
            with self.assertRaises(ValidationError): validator.validate(bad)

    def test_runtime_rejects_duplicate_and_excess_constraints(self):
        good=strict_source_certificate(old_case_question("logic-06"))
        for clauses in (good["constraints"]+[good["constraints"][0]],good["constraints"]*3):
            bad=copy.deepcopy(good); bad["constraints"]=clauses
            with self.assertRaises(ValueError): _parse_candidate(json.dumps(bad))

    def test_prompt_has_every_pair_and_explicit_stop(self):
        prompt=extraction_prompt_for_question(old_case_question("logic-06"))
        for text in ("(B0,B2)","(B1,B3)","(B2,B4)","(B3,B5)","Genau 6 Constraints","JSON sofort schließen"):
            self.assertIn(text,prompt)
        self.assertNotIn("B6",prompt)


if __name__=="__main__": unittest.main()
