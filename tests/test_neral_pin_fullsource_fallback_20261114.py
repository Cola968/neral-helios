"""RED contracts: five-position semantics and explicit no-503 guarded fallback.

All model calls are CPU mocks; no Ollama, GPU or sealed benchmark.
"""
import copy
import json
import os
import unittest
from unittest.mock import patch

from flask import Flask

from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend
from neral_local_chat_api import register_neral_local_chat_api
from neral_nl_schema_bridge import (
    EXTRACT_BASE_PROMPT, EXTRACT_QLORA_PROMPT, V2_FEW_SHOT_EXAMPLES,
    coverage_gate,
)
from neral_nl_verified_route import run_guarded_nl_bridge, strict_source_certificate
from neral_typed_constraints_v2 import PREFIX, solve_typed_constraints
from scripts.neral_nl_schema_probe import old_case_question


def turns(q):
    return [{"role": "system", "content": "NERAL"},
            {"role": "user", "content": q}]


class PinFullSourceWitnessTests(unittest.TestCase):
    def test_complete_five_digit_example_is_synthetic_and_solver_valid(self):
        pin = V2_FEW_SHOT_EXAMPLES[2]
        self.assertIn("fünfstelliger", pin["task"])
        self.assertNotEqual(pin["task"], old_case_question("21453"))
        doc = pin["schema"]
        self.assertEqual(set(doc["variables"]), {f"p{i}" for i in range(1,6)})
        for item in doc["variables"].values():
            self.assertEqual(item, {"type":"int","domain":[1,2,3,4,5]})
        self.assertEqual(doc["constraints"][0],
                         {"all_different":[f"p{i}" for i in range(1,6)]})
        self.assertEqual(solve_typed_constraints(
            PREFIX+json.dumps(doc,ensure_ascii=False)),2)
        self.assertEqual(doc, strict_source_certificate(pin["task"]))

    def test_both_prompts_explicitly_explain_all_position_idioms(self):
        for prompt in (EXTRACT_BASE_PROMPT,EXTRACT_QLORA_PROMPT):
            for literal in ("p1..p5", "pY - pX >= 2", "pX - pY == 2",
                            "pY - pX == 2", "pX == pY + 1",
                            "pX > 1", "pX < 5", "all_different"):
                with self.subTest(literal=literal):
                    self.assertIn(literal, prompt)

    def test_coverage_rejects_missing_wrong_and_logically_unpaired_rules(self):
        question=old_case_question("21453")
        doc=strict_source_certificate(question)
        self.assertTrue(coverage_gate(question,doc)["passed"])
        for pos in range(len(doc["constraints"])):
            damaged=copy.deepcopy(doc)
            damaged["constraints"].pop(pos)
            with self.subTest(pos=pos):
                self.assertFalse(coverage_gate(question,damaged)["passed"])
        # Existing lexical witness can be tricked by detached OR + AND.
        mistaken=copy.deepcopy(doc)
        between=mistaken["constraints"][2].pop("or")
        mistaken["constraints"][2]["and"]=between
        taut={"compare":{"left":{"var":"p1"},"op":"==","right":{"var":"p1"}}}
        mistaken["constraints"].append({"or":[copy.deepcopy(taut),taut]})
        self.assertFalse(coverage_gate(question,mistaken)["passed"])
        extra=copy.deepcopy(doc)
        extra["constraints"].append({"compare":{"left":{"var":"p4"},
            "op":"==","right":{"const":2}}})
        self.assertFalse(coverage_gate(question,extra)["passed"])

    def test_not_on_edge_accepts_exact_equivalent_inequalities(self):
        doc=copy.deepcopy(strict_source_certificate(old_case_question("21453")))
        doc["constraints"][-2]={"compare":{"left":{"var":"p4"},
            "op":">","right":{"const":1}}}
        doc["constraints"][-1]={"compare":{"left":{"var":"p4"},
            "op":"<","right":{"const":5}}}
        self.assertTrue(coverage_gate(old_case_question("21453"),doc)["passed"])


class No503FallbackContracts(unittest.TestCase):
    def test_token_limit_returns_regular_explicitly_unverified_refusal(self):
        q=old_case_question("21453")
        bad=copy.deepcopy(strict_source_certificate(q))
        bad["constraints"].pop(2)
        calls=[]
        def limit(original):
            calls.append(original)
            raise OllamaCompletionError("token_limit")
        result=run_guarded_nl_bridge(
            turns(q),lambda _:json.dumps(bad),lambda _:json.dumps(bad),limit)
        self.assertEqual(calls,[turns(q)])
        self.assertFalse(result["safe_to_promote"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["response_source"],"deterministic_safety_refusal")
        self.assertEqual(result["verification_status"],"base_fallback_token_limit_unverified")
        self.assertEqual(result["fallback_error_code"],"token_limit")
        self.assertEqual(result["selection_policy"],"explicit_safe_refusal")
        self.assertNotIn("1",result["text"])
        self.assertFalse(result["model_generated"])

    def test_http_pin_and_new_unsafe_condition_are_both_200_not_503(self):
        app=Flask("neral-token-limit-guard-cpu")
        backend=OllamaLocalBackend("qwen3.5:9b",
                  inference_profile="measured_nonthinking_2048")
        observed=[]
        def raw(original):
            observed.append(original)
            raise OllamaCompletionError("token_limit")
        backend.respond_chat=raw
        register_neral_local_chat_api(app,backend)
        q=old_case_question("21453")
        bad=copy.deepcopy(strict_source_certificate(q))
        bad["constraints"].pop(2)
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_NL_SCHEMA":"1"}), \
             patch("neral_nl_local_ollama.make_local_extractor",
                   return_value=lambda _:json.dumps(bad)):
            for question in (q,q+" Außerdem steht die Ziffer 1 ganz links."):
                with self.subTest(question=question[-38:]):
                    response=app.test_client().post("/api/neral/chat/think",
                        json={"prompt":question,"nl_schema_consensus":True})
                    self.assertEqual(response.status_code,200,response.get_json())
                    body=response.get_json()
                    self.assertFalse(body["safe_to_promote"])
                    self.assertFalse(body["verified"])
                    self.assertEqual(body["verification_status"],
                                     "base_fallback_token_limit_unverified")
                    self.assertEqual(body["response_source"],
                                     "deterministic_safety_refusal")
            self.assertEqual(len(observed),2)

    def test_successful_base_remains_original_raw_fallback(self):
        q=old_case_question("21453")
        result=run_guarded_nl_bridge(turns(q),
            lambda _:"{}",lambda _:"{}",lambda _:"BASIS_UNVERIFIED")
        self.assertEqual(result["text"],"BASIS_UNVERIFIED")
        self.assertEqual(result["response_source"],"raw_qwen")
        self.assertFalse(result["safe_to_promote"])


if __name__=="__main__":
    unittest.main()
