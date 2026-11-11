"""RED CPU contracts: opt-in bounded formal solving and private first-turn telemetry."""
from __future__ import annotations
import json
import unittest

from neral_formal_constraints import InvalidFormalConstraints, try_solve_formal_constraints
from neral_tir_experimental import run_experimental_tir

PREFIX = "NERAL_CONSTRAINTS_JSON: "

def formal(**kw):
    return PREFIX + json.dumps(kw, separators=(",", ":"))

def messages(question):
    return [{"role": "system", "content": "Du bist NERAL."},
            {"role": "user", "content": question}]


class FormalConstraintContracts(unittest.TestCase):
    def test_bounded_integer_order_count(self):
        q = formal(variables={"A":[1,2,3], "B":[1,2,3]},
                   rules=[{"left":"A", "op":"<", "right":"B"}], query="count")
        self.assertEqual(try_solve_formal_constraints(q), 3)

    def test_boolean_exclusion_and_satisfiability(self):
        base = {"variables":{"A":[False,True],"B":[False,True]},
                "rules":[{"left":"A","op":"!=","right":"B"}]}
        self.assertEqual(try_solve_formal_constraints(formal(**base,query="count")), 2)
        self.assertEqual(try_solve_formal_constraints(formal(**base,query="satisfiable")), 1)
        base["rules"].append({"left":"A","op":"==","right":"B"})
        self.assertEqual(try_solve_formal_constraints(formal(**base,query="satisfiable")), 0)

    def test_fixed_value_and_all_different(self):
        q = formal(variables={"A":[1,2,3],"B":[1,2,3],"C":[1,2,3]},
                   rules=[{"left":"A","op":"==","value":2}],
                   all_different=[["A","B","C"]],query="count")
        self.assertEqual(try_solve_formal_constraints(q), 2)

    def test_nonformal_requests_are_never_classified_by_words(self):
        self.assertIsNone(try_solve_formal_constraints(
            "There are three people A, B, C. Count valid permutations."))
        self.assertIsNone(try_solve_formal_constraints("Explain a galaxy."))

    def test_rejects_unsafe_schema_and_excessive_search(self):
        base = {"variables":{"A":[1,2,3],"B":[1,2,3]},
                "rules":[{"left":"A","op":"<","right":"B"}],"query":"count"}
        variants=[
            {**base,"extra":"ignore me"},
            {**base,"rules":[{"left":"A","op":"__import__","right":"B"}]},
            {**base,"rules":[{"left":"A","op":"<","right":"DOES_NOT_EXIST"}]},
            {**base,"rules":[{"left":"A","op":"<","right":"B","value":0}]},
            {**base,"variables":{"A":["__import__('os').system('echo X')"]}},
            {**base,"variables":{"A":list(range(101)), "B":list(range(101)),
                                 "C":list(range(101))}},
            {**base,"variables":{"A":[1,1],"B":[1,2]}},
        ]
        for variant in variants:
            with self.subTest(variant=str(variant)[:90]):
                with self.assertRaises(InvalidFormalConstraints):
                    try_solve_formal_constraints(formal(**variant))
        with self.assertRaises(InvalidFormalConstraints):
            try_solve_formal_constraints(PREFIX + '{"variables":')

    def test_formal_route_does_not_invoke_model_or_generated_python(self):
        q = formal(variables={"A":[1,2,3],"B":[1,2,3]},
                   rules=[{"left":"A","op":"<","right":"B"}],query="count")
        output = run_experimental_tir(
            messages(q+"\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."),
            lambda _: self.fail("formal solver must bypass model"),
            tool_runner=lambda *_: self.fail("must never execute generated code"))
        self.assertEqual(output["text"],"FINAL_ANSWER: 3")
        self.assertEqual(output["calls_used"],0)
        self.assertTrue(output["verified"])
        self.assertFalse(output["model_generated"])
        self.assertEqual(output["tool_used"],"neral_logic_exact")

    def test_invalid_formal_schema_fails_closed(self):
        out = run_experimental_tir(
            messages(PREFIX+"{bad JSON"),
            lambda _: self.fail("malformed structured request must not prompt-inject"))
        self.assertEqual(out["verification_status"],"formal_schema_rejected")
        self.assertFalse(out["verified"])
        self.assertEqual(out["calls_used"],0)


class FirstTurnTelemetryContracts(unittest.TestCase):
    def test_only_scalar_first_turn_evidence_is_emitted(self):
        seen=[]
        replies=iter([
            'NERAL_EXECUTE_JSON: {"tool":"solve","code":"6*7"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 42",
        ])
        def respond(turns):
            seen.append([dict(x) for x in turns])
            return next(replies)
        result = run_experimental_tir(
            messages("Compute 6*7."),respond,
            tool_runner=lambda *_:{"status":"PASSED","output":"42"},
            first_turn_diagnostic=lambda:{
                "eval_count":17, "prompt_eval_count":65,"done_reason":"stop",
                "content":"SECRET_MUST_NOT_LEAK","thinking":"SECRET_MUST_NOT_LEAK"})
        self.assertEqual(result["first_turn_telemetry"],
                         {"eval_count":17,"prompt_eval_count":65,
                          "done_reason":"stop","decision":"tool"})
        self.assertNotIn("SECRET_MUST_NOT_LEAK",str(result))

    def test_free_text_fallback_classified_as_invalid(self):
        answers=iter(["lengthy reasoning without any packet", "Fallback answer"])
        result=run_experimental_tir(messages("Count all arrangements"),
                                    lambda _:next(answers),
                                    first_turn_diagnostic=lambda:{
                                        "eval_count":2048,"done_reason":"stop"})
        self.assertEqual(result["first_turn_telemetry"]["decision"],"unstructured")
        self.assertEqual(result["first_turn_telemetry"]["eval_count"],2048)
        self.assertEqual(result["verification_status"],"base_fallback_unverified")



class FirstTurnFailureHttpContracts(unittest.TestCase):
    def test_token_limit_still_reports_first_turn_without_private_text(self):
        from unittest.mock import patch
        from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend
        from neral_preview_server import create_preview_app

        class Fake(OllamaLocalBackend):
            def unload(self):
                return None
            def respond_chat(self, turns):
                self.last_inference_diagnostic = {
                    "eval_count":2048, "prompt_eval_count":83,
                    "done_reason":"length", "content":"PRIVATE_NOT_FOR_CLIENT"}
                raise OllamaCompletionError("token_limit")
            def respond_think(self, turns):
                return self.respond_chat(turns)
        low=Fake("qwen3:8b", inference_profile="legacy")
        high=Fake("qwen3.5:9b", inference_profile="measured_nonthinking_2048")
        client=create_preview_app(low, experimental_high_backend=high).test_client()
        client.get("/",base_url="http://127.0.0.1:9690")
        session=client.get("/api/neral/preview/session",
                           base_url="http://127.0.0.1:9690")
        with patch.dict("os.environ",{"NERAL_EXPERIMENTAL_TIR":"1"}):
            response=client.post("/api/neral/chat/think",
                json={"prompt":"Find solutions satisfying several constraints.",
                      "experimental_tir":True},
                headers={"X-Neral-CSRF":session.json["csrf"]},
                base_url="http://127.0.0.1:9690")
        self.assertEqual(response.status_code,503,response.get_json())
        stats=response.json["first_turn_telemetry"]
        self.assertEqual(stats["eval_count"],2048)
        self.assertEqual(stats["done_reason"],"length")
        self.assertEqual(stats["decision"],"unavailable_on_failure")
        self.assertNotIn("PRIVATE_NOT_FOR_CLIENT",str(response.json))



class ScalarRejectionOnLaterTokenLimit(unittest.TestCase):
    def test_503_retains_first_extractor_reason_without_model_reply(self):
        from unittest.mock import patch
        from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend
        from neral_preview_server import create_preview_app
        class Fake(OllamaLocalBackend):
            def unload(self):
                return None
            def respond_formal_extract(self, turns):
                self.last_inference_diagnostic = {
                    "eval_count": 287, "prompt_eval_count": 800,
                    "done_reason": "stop",
                    "content": "PRIVATE_MODEL_JSON_NOT_AUDITED"}
                return '{"kind":"formal","complete":true}'
            def respond_chat(self, turns):
                raise OllamaCompletionError("token_limit")
            def respond_think(self, turns):
                return self.respond_chat(turns)
        low = Fake("qwen3:8b")
        high = Fake("qwen3.5:9b",
                    inference_profile="measured_nonthinking_2048")
        client = create_preview_app(low, experimental_high_backend=high).test_client()
        client.get("/", base_url="http://127.0.0.1:9690")
        session = client.get("/api/neral/preview/session",
                             base_url="http://127.0.0.1:9690")
        with patch.dict("os.environ",{"NERAL_EXPERIMENTAL_TIR":"1"}):
            response = client.post("/api/neral/chat/think",
                json={"prompt": "Count complete arrangements subject to constraints.",
                      "experimental_tir": True},
                headers={"X-Neral-CSRF": session.json["csrf"]},
                base_url="http://127.0.0.1:9690")
        self.assertEqual(response.status_code, 503, response.get_json())
        self.assertEqual(response.json["error_code"], "token_limit")
        event = response.json["first_turn_telemetry"]
        self.assertEqual(event["eval_count"], 287)
        self.assertEqual(event["done_reason"], "stop")
        self.assertEqual(event["decision"], "unavailable_on_failure")
        self.assertEqual(event["extraction_rejection_reason"], "invalid_envelope")
        self.assertNotIn("PRIVATE_MODEL_JSON_NOT_AUDITED", str(response.json))


class BenchmarkTelemetrySafety(unittest.TestCase):
    def test_scalar_filter_never_leaks_arbitrary_payload(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        event = {"eval_count":2048,"prompt_eval_count":110,
                 "done_reason":"stop","decision":"unstructured",
                 "prompt":"TOP_SECRET","content":"TOP_SECRET"}
        filtered=safe_first_turn_event(event)
        self.assertEqual(filtered, {"eval_count":2048,"prompt_eval_count":110,
                         "done_reason":"stop","decision":"unstructured"})
        self.assertNotIn("TOP_SECRET",str(filtered))
        for decision in ("extracted_formal", "extraction_fallback", "extraction_none"):
            with self.subTest(decision=decision):
                candidate = dict(event, decision=decision)
                self.assertEqual(
                    safe_first_turn_event(candidate)["decision"], decision)
                self.assertNotIn("TOP_SECRET", str(safe_first_turn_event(candidate)))

        safe = safe_first_turn_event(dict(event,
            decision="unavailable_on_failure",
            extraction_rejection_reason="invalid_envelope",
            content="UNTRUSTED_MODEL_REPLY"))
        self.assertEqual(safe["extraction_rejection_reason"], "invalid_envelope")
        self.assertNotIn("UNTRUSTED_MODEL_REPLY",str(safe))
        self.assertIsNone(safe_first_turn_event(dict(event,
            extraction_rejection_reason="UNTRUSTED_MODEL_REPLY")))

        self.assertIsNone(safe_first_turn_event(
            {"eval_count":True,"prompt_eval_count":0,"done_reason":"stop",
             "decision":"tool"}))
        self.assertIsNone(safe_first_turn_event(
            {"eval_count":100,"prompt_eval_count":0,"done_reason":"unexpected",
             "decision":"tool"}))

if __name__=="__main__":
    unittest.main()
