"""Regression contracts: truthful failure provenance and PR537 proof boundary.

No benchmark-specific routing, no regex over the 25 old logic prompts, no
model-generate-code => verified shortcut and no increase in token budget.
"""
import unittest
from unittest.mock import patch
from neral_dual_routing import (
    BASE_TAG,TUNED_TAG,OllamaTransportFailure,run_dual_route,_ollama_chat,
)
from neral_tir_regression_guard import choose_guarded_tir_reply

MESSAGE=[{"role":"system","content":"You are local NERAL"},
         {"role":"user","content":"An arbitrary unseen finite-domain problem"}]

class LogicFailureProvenance(unittest.TestCase):
    def test_rejected_schema_then_base_length_failure_keeps_only_safe_scalar_stage(self):
        sent=[]
        def fake_chat(tag,turns,seed,*,structured=False,repair=False):
            sent.append((tag,structured,repair))
            if structured:
                return '{"kind":"none"}',{"model":tag}
            if tag!=BASE_TAG:
                raise AssertionError("No tool model may replace base fallback")
            raise OllamaTransportFailure({"reason":"token_limit","done_reason":"length",
                "http_status":200,"attempts":1})
        def fake_tir(messages,respond,*,extract_constraints):
            extract_constraints([{"role":"system","content":"schema"},
                                  {"role":"user","content":"source"}])
            respond([dict(x) for x in messages])
            raise AssertionError("Unreachable")
        with patch("neral_dual_routing.require_exact_models",return_value={}), \
             patch("neral_dual_routing.require_exact_guard",return_value="pinned"), \
             patch("neral_dual_routing.sample_residency",return_value={}), \
             patch("neral_dual_routing._ollama_chat",side_effect=fake_chat), \
             patch("neral_dual_routing.run_experimental_tir",side_effect=fake_tir):
            with self.assertRaises(OllamaTransportFailure) as observed:
                run_dual_route(MESSAGE,sample_gpu=False)
        packet=observed.exception.diagnostic
        self.assertEqual(packet["reason"],"token_limit")
        self.assertEqual(packet["done_reason"],"length")
        self.assertEqual(packet["http_status"],200)
        self.assertEqual(packet["inference_phase"],"base_controller_fallback")
        self.assertEqual(packet["schema_attempts"],1)
        self.assertEqual(packet["model_proposal_calls"],0)
        self.assertEqual(packet["base_fallback_attempts"],1)
        self.assertNotIn("question",str(packet).casefold())
        self.assertNotIn("arbitrary unseen",str(packet))
        self.assertEqual(sent,[(TUNED_TAG,True,False),(BASE_TAG,False,False)])

    def test_qlora_proposal_length_error_tagged_separately(self):
        def fake_chat(tag,turns,seed,*,structured=False,repair=False):
            if structured:
                raise OllamaTransportFailure({"reason":"token_limit",
                    "done_reason":"length","http_status":200})
            raise AssertionError("Not reached")
        def fake_tir(messages,respond,*,extract_constraints):
            extract_constraints([{"role":"system","content":"schema"},
                                  {"role":"user","content":"source"}])
            raise AssertionError("Not reached")
        with patch("neral_dual_routing.require_exact_models",return_value={}), \
             patch("neral_dual_routing.require_exact_guard",return_value="pinned"), \
             patch("neral_dual_routing.sample_residency",return_value={}), \
             patch("neral_dual_routing._ollama_chat",side_effect=fake_chat), \
             patch("neral_dual_routing.run_experimental_tir",side_effect=fake_tir):
            with self.assertRaises(OllamaTransportFailure) as observed:
                run_dual_route(MESSAGE,sample_gpu=False)
        self.assertEqual(observed.exception.diagnostic["inference_phase"],"qlora_schema_extraction")
        self.assertEqual(observed.exception.diagnostic["schema_attempts"],1)
        self.assertEqual(observed.exception.diagnostic["base_fallback_attempts"],0)

    def test_current_4096_context_limit_and_2048_output_budget(self):
        complete={"done":True,"done_reason":"stop","message":{"content":"42"},
                  "load_duration":1000000}
        with patch("neral_dual_routing._post_chat_with_status",return_value=(complete,200)) as send, \
             patch("neral_dual_routing.sample_residency",return_value={}):
            result,detail=_ollama_chat(BASE_TAG,MESSAGE,0)
        packet=send.call_args.args[0]
        self.assertEqual(packet["options"]["num_ctx"],4096)
        self.assertEqual(packet["options"]["num_predict"],2048)
        self.assertEqual(packet["options"]["temperature"],0)
        self.assertFalse(packet["think"])
        self.assertEqual(result,"42")

class GuardRejectsModelAuthoredLogicCode(unittest.TestCase):
    def test_successful_model_python_execution_is_not_a_semantic_proof(self):
        report={"text":"FINAL_ANSWER: 123","mode":"HIGH",
           "verified":True,"model_generated":True,
           "verification_status":"verified",
           "verification_scope":"exact_user_python_execution",
           "review_method":"exact_user_ast_match",
           "tool_used":"neral_python_sandbox",
           "tool_calls_succeeded":1,"tool_calls_triggered":1,
           "calls_used":2,"seconds":.5}
        calls=[]
        def raw(turns):
            calls.append(turns)
            return "FINAL_ANSWER: 7"
        selected=choose_guarded_tir_reply(report,MESSAGE,raw)
        self.assertEqual(selected["selection_policy"],"raw_qwen_due_to_unverified_tir")
        self.assertFalse(selected["verified"])
        self.assertEqual(selected["text"],"FINAL_ANSWER: 7")
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0],MESSAGE)

    def test_independently_identical_user_python_remains_eligible(self):
        report={"text":"FINAL_ANSWER: 123","mode":"HIGH",
           "verified":True,"model_generated":False,
           "verification_status":"verified",
           "verification_scope":"exact_user_python_execution",
           "review_method":"exact_user_ast_match",
           "tool_used":"neral_python_sandbox",
           "tool_calls_succeeded":1,"tool_calls_triggered":1,
           "calls_used":0,"seconds":.5}
        result=choose_guarded_tir_reply(report,MESSAGE,
                      lambda _:self.fail("verified result should not invoke raw"))
        self.assertTrue(result["verified"])
        self.assertEqual(result["selection_policy"],"independently_verified_tir")

if __name__=="__main__":
    unittest.main()
