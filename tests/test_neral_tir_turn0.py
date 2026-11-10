"""RED: Turn-0 constrained JSON extraction, deterministic fail-closed fallback.

CPU-only; no Ollama GPU, network, cloud, generated code or personal data.
"""
from __future__ import annotations
import json
import unittest
from unittest.mock import patch

from neral_turn0_extractor import EXTRACT_PROMPT, parse_extracted_constraints
from neral_tir_experimental import run_experimental_tir

QUESTION = ("A and B each choose a number from 1 to 3. "
            "A must be smaller than B. How many assignments are possible?")
SUFFIX = "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."
SCHEMA = {
    "variables": {"A": [1, 2, 3], "B": [1, 2, 3]},
    "rules": [{"left": "A", "op": "<", "right": "B"}],
    "query": "count",
}
EVIDENCE = ["each choose a number from 1 to 3", "A must be smaller than B",
            "How many assignments are possible?"]

def packet(*, kind="formal", complete=True, schema=SCHEMA, evidence=EVIDENCE):
    return json.dumps({
        "kind": kind, "complete": complete, "schema": schema,
        "evidence": evidence,
    }, separators=(",", ":"))

def messages(question):
    return [{"role": "system", "content": "Du bist NERAL."},
            {"role": "user", "content": question}]


class ExtractionSchemaContracts(unittest.TestCase):
    def test_single_few_shot_and_strict_json_without_xml(self):
        self.assertEqual(EXTRACT_PROMPT.count("Example input:"), 1)
        self.assertIn("JSON", EXTRACT_PROMPT)
        self.assertNotIn("<tool_call", EXTRACT_PROMPT)

    def test_valid_model_extraction_is_a_computed_but_unverified_answer(self):
        self.assertEqual(parse_extracted_constraints(QUESTION, packet()), 3)

    def test_rejects_unproven_incomplete_or_oversized_proposals(self):
        from neral_formal_constraints import MAX_ASSIGNMENTS
        self.assertEqual(MAX_ASSIGNMENTS, 100000)
        variants = [
            "", "not JSON", "{bad", " "+packet(), packet()+"\nhello",
            '{"kind":"none"}'+packet(),
            packet(complete=False), packet(evidence=[]),
            packet(evidence=["not in source"]*3),
            packet(evidence=EVIDENCE[:1]), packet(kind="unknown"),
            json.dumps({"kind":"formal","complete":True,"schema":SCHEMA,
                        "evidence":EVIDENCE,"code":"print(1)"}),
            packet(schema={**SCHEMA,"variables":{
                "A":list(range(12)),"B":list(range(12)),
                "C":list(range(12)),"D":list(range(12)),
                "E":list(range(12))}}),
            '{"kind":"formal","kind":"none"}',
            packet(schema={**SCHEMA,"rules":[]}),
        ]
        for value in variants:
            with self.subTest(value=str(value)[:55]):
                self.assertIsNone(parse_extracted_constraints(QUESTION,value))
        self.assertIsNone(parse_extracted_constraints(QUESTION,'{"kind":"none"}'))

    def test_natural_language_is_never_verified_by_model_self_certification(self):
        replies=iter(["Fallback untouched."])
        observed=[]
        def respond(turns):
            observed.append([dict(row) for row in turns])
            return next(replies)
        extract_calls=[]
        def extract(turns):
            extract_calls.append(turns)
            return packet()
        report=run_experimental_tir(
            messages(QUESTION+SUFFIX),respond,extract_constraints=extract,
            first_turn_diagnostic=lambda:{
                "eval_count":94,"prompt_eval_count":122,"done_reason":"stop"})
        self.assertEqual(len(extract_calls),1)
        self.assertEqual(len(observed),0)
        self.assertEqual(report["text"],"FINAL_ANSWER: 3")
        self.assertEqual(report["verification_status"],"model_extracted_schema_unverified")
        self.assertFalse(report["verified"])
        self.assertTrue(report["model_generated"])
        self.assertEqual(report["verification_scope"],"model_extracted_finite_domain")
        self.assertEqual(report["tool_used"],"neral_logic_exact")
        self.assertEqual(report["calls_used"],1)
        self.assertEqual(report["tool_calls_succeeded"],1)
        self.assertEqual(report["first_turn_telemetry"]["eval_count"],94)
        self.assertEqual(report["first_turn_telemetry"]["decision"],"extracted_formal")

    def test_nonformal_refusal_uses_untouched_qwen_not_old_tir_prompt(self):
        original=messages("How does rain form?")
        observed=[]
        def respond(turns):
            observed.append([dict(row) for row in turns])
            return "Rain forms after condensation."
        report=run_experimental_tir(original,respond,
                extract_constraints=lambda _: '{"kind":"none"}')
        self.assertEqual(observed,[original])
        self.assertEqual(report["text"],"Rain forms after condensation.")
        self.assertEqual(report["calls_used"],2)
        self.assertEqual(report["tool_calls_succeeded"],0)
        self.assertEqual(report["verification_status"],"base_fallback_unverified")
        self.assertEqual(report["first_turn_telemetry"]["decision"],"extraction_fallback")

    def test_malformed_or_semantically_unsupported_falls_back_without_executing_code(self):
        for proposed in ("not JSON", packet(complete=False),
                         packet(evidence=["invented"]*3)):
            with self.subTest(proposed=proposed[:20]):
                original=messages(QUESTION)
                seen=[]
                result=run_experimental_tir(original,
                    lambda m:(seen.append(m) or "Raw safe answer"),
                    extract_constraints=lambda _:proposed,
                    tool_runner=lambda *_:self.fail("no generated Python"))
                self.assertEqual(seen,[original])
                self.assertEqual(result["text"],"Raw safe answer")
                self.assertFalse(result["verified"])
                self.assertEqual(result["tool_calls_triggered"],0)

    def test_extractor_timeout_or_length_causes_direct_fallback(self):
        original=messages(QUESTION)
        def fail(_):
            raise RuntimeError("local model truncated")
        report=run_experimental_tir(original, lambda _: "Raw fallback",
                                    extract_constraints=fail)
        self.assertEqual(report["text"],"Raw fallback")
        self.assertEqual(report["calls_used"],2)

    def test_legacy_tir_without_extractor_is_unchanged(self):
        seen=[]
        def respond(turns):
            seen.append([dict(row) for row in turns])
            return "NERAL_FINAL_TEXT: ordinary answer"
        result=run_experimental_tir(messages("Explain a nebula"), respond)
        self.assertEqual(result["text"],"ordinary answer")
        self.assertEqual(len(seen),1)
        self.assertNotIn("extraction_status",result)


class OllamaExtractWireContracts(unittest.TestCase):
    def test_only_extractor_uses_json_384_temp0_and_raw_remains_2048(self):
        from neral_agent_autonomy import OllamaLocalBackend
        requests=[]
        class Local:
            def open(self,request,timeout):
                requests.append(json.loads(request.data))
                class Response:
                    def __enter__(self): return self
                    def __exit__(self,*args): return False
                    def read(self,*args):
                        return json.dumps({"done":True,"done_reason":"stop",
                            "eval_count":64,"message":{"content":"{\"kind\":\"none\"}"}}).encode()
                return Response()
        b=OllamaLocalBackend("qwen3.5:9b", timeout=90,
                            inference_profile="measured_nonthinking_2048")
        turns=messages("Explain rain.")
        with patch("neral_agent_autonomy.local_only_opener",return_value=Local()):
            b.respond_formal_extract(turns)
            b.respond_chat(turns)
        self.assertEqual(len(requests),2)
        self.assertEqual(requests[0]["format"],"json")
        self.assertFalse(requests[0]["think"])
        self.assertEqual(requests[0]["options"]["num_predict"],384)
        self.assertEqual(requests[0]["options"]["temperature"],0)
        self.assertEqual(requests[1]["options"]["num_predict"],2048)
        self.assertNotIn("format",requests[1])

if __name__ == "__main__":
    unittest.main()
