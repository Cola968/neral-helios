"""NERAL 1.1 adaptive experimental HTTP contracts. All offline CPU fakes."""
import json
import os
import unittest
from unittest.mock import patch
from flask import Flask

from neral_agent_autonomy import OllamaLocalBackend
from neral_local_chat_api import register_neral_local_chat_api

QUESTION = ("Count permutations of numbers 1 through 4. "
            "Number 1 is before number 2.")
UNSUPPORTED = QUESTION + " Number 4 must be lucky."
CHECKED_CODE = "answer=12\nassert answer == (3*4)\nprint(answer)"


def rig(model="qwen3.5:9b", profile="measured_nonthinking_2048"):
    app = Flask("adaptive-planner-fake-only")
    high = OllamaLocalBackend(model, inference_profile=profile)
    seen = []
    def raw(turns):
        seen.append([dict(x) for x in turns])
        return "UNTOUCHED_BASE_QWEN"
    high.respond_chat = raw
    register_neral_local_chat_api(app, high)
    return app.test_client(), seen


class AdaptiveThinkHTTP(unittest.TestCase):
    def test_disabled_by_default_even_if_request_opted_in(self):
        client, calls = rig()
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "0"}):
            resp = client.post("/api/neral/chat/think", json={
                "prompt": QUESTION, "adaptive_planner": True})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(calls, [])

    def test_mode_is_mutually_exclusive_and_boolean(self):
        client, calls = rig()
        for conflict in (
            {"adaptive_planner": "1"},
            {"adaptive_planner": True, "nl_schema_consensus": True},
            {"adaptive_planner": True, "dual_routing": True},
            {"adaptive_planner": True, "experimental_tir": True},
            {"adaptive_planner": True, "evaluation_final_answer": True},
            {"adaptive_planner": True, "conservative_tir": True},
        ):
            with self.subTest(conflict=conflict):
                response = client.post("/api/neral/chat/think", json={
                    "prompt": QUESTION, **conflict})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(calls, [])

    def test_noneligible_model_or_profile_rejected(self):
        for model, profile in (("some-model:1", "measured_nonthinking_2048"),
                               ("qwen3.5:9b", "legacy")):
            client, calls = rig(model, profile)
            with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "1"}):
                r = client.post("/api/neral/chat/think", json={
                    "prompt": QUESTION, "adaptive_planner": True})
            self.assertEqual(r.status_code, 409)
            self.assertEqual(calls, [])

    def test_full_source_python_proof_end_to_end_with_one_local_model(self):
        client, calls = rig()
        models, signals, programs = [], [], []
        def maker(model):
            models.append(model)
            def proposal(turns, signal):
                signals.append(signal)
                return {"strategy": "PYTHON_SANDBOX", "code": CHECKED_CODE}
            return proposal
        def docker(code):
            programs.append(code)
            return {"status": "PASSED", "output": "12",
                    "verification_scope": "execution_only"}
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "1"}), \
             patch("neral_adaptive_ollama.make_local_adaptive_proposer", side_effect=maker), \
             patch("neral_adaptive_planner.run_python_in_docker", side_effect=docker):
            response = client.post("/api/neral/chat/think", json={
                "prompt": QUESTION, "adaptive_planner": True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(models, ["neral-qwen35-qlora-v1:latest"])
        self.assertEqual(len(signals), 1)
        self.assertEqual(programs, [CHECKED_CODE])
        self.assertEqual(calls, [])
        self.assertTrue(response.json["verified"])
        self.assertEqual(response.json["text"], "12")
        self.assertFalse(response.json["model_generated"])
        self.assertTrue(response.json["safe_to_promote"])
        self.assertEqual(response.json["response_source"], "independent_source_python")

    def test_unsound_source_proposal_cannot_promote_without_proof(self):
        client, calls = rig()
        signals = []
        def maker(model):
            def propose(turns, signal):
                signals.append(signal)
                return {"strategy": "PYTHON_SANDBOX", "code": CHECKED_CODE}
            return propose
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "1"}), \
             patch("neral_adaptive_ollama.make_local_adaptive_proposer", side_effect=maker), \
             patch("neral_adaptive_planner.run_python_in_docker",
                   return_value={"status": "PASSED", "output": "12"}):
            response = client.post("/api/neral/chat/think", json={
                "prompt": UNSUPPORTED, "adaptive_planner": True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json["text"], "UNTOUCHED_BASE_QWEN")
        self.assertFalse(response.json["verified"])
        self.assertFalse(response.json["safe_to_promote"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-1]["content"], UNSUPPORTED)
        # Post-verification deliberately permits one tool proposal on unknown
        # computational text, but never promotes it without an independent proof.
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["preferred_strategy"], "PYTHON_SANDBOX")
        self.assertEqual(response.json["selection_policy"],
                         "raw_qwen_due_to_unverified_adaptive")

    def test_normal_direct_request_uses_only_unchanged_base(self):
        client, calls = rig()
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "1"}), \
             patch("neral_adaptive_ollama.make_local_adaptive_proposer",
                   side_effect=AssertionError("No call to proposal")):
            response = client.post("/api/neral/chat/think", json={
                "prompt": "Beschreibe ein Teleskop.", "adaptive_planner": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["text"], "UNTOUCHED_BASE_QWEN")
        self.assertFalse(response.json["verified"])
        self.assertEqual(len(calls), 1)

    def test_multi_turn_never_uses_trusted_full_source_proof(self):
        client, calls = rig()
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "1"}), \
             patch("neral_adaptive_ollama.make_local_adaptive_proposer",
                   side_effect=AssertionError("No multi-turn proposal")):
            response = client.post("/api/neral/chat/think", json={
                "prompt": QUESTION,
                "history": [{"role": "user", "content": "Earlier"},
                            {"role": "assistant", "content": "Earlier answer"}],
                "adaptive_planner": True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertFalse(response.json["verified"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)

    def test_sandbox_unavailable_uses_unmodified_fallback(self):
        client, calls = rig()
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "1"}), \
             patch("neral_adaptive_ollama.make_local_adaptive_proposer",
                   return_value=lambda turns, signal: {
                       "strategy": "PYTHON_SANDBOX", "code": CHECKED_CODE}), \
             patch("neral_adaptive_planner.run_python_in_docker",
                   return_value={"status": "UNAVAILABLE"}):
            response = client.post("/api/neral/chat/think", json={
                "prompt": QUESTION, "adaptive_planner": True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(len(calls), 1)
        self.assertEqual(response.json["text"], "UNTOUCHED_BASE_QWEN")
        self.assertFalse(response.json["verified"])

    def test_failed_raw_model_never_returns_partial_candidate(self):
        app = Flask("adaptive-fallback-refusal")
        high = OllamaLocalBackend("qwen3.5:9b",
                                  inference_profile="measured_nonthinking_2048")
        high.respond_chat = lambda turns: (_ for _ in ()).throw(
            RuntimeError("SECRET_MODEL_OUTPUT"))
        register_neral_local_chat_api(app, high)
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER": "1"}), \
             patch("neral_adaptive_ollama.make_local_adaptive_proposer",
                   return_value=lambda turns, signal: {
                       "strategy": "PYTHON_SANDBOX", "code": CHECKED_CODE}), \
             patch("neral_adaptive_planner.run_python_in_docker",
                   return_value={"status": "FAILED", "traceback": "PRIVATE"}):
            response = app.test_client().post("/api/neral/chat/think", json={
                "prompt": QUESTION, "adaptive_planner": True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertFalse(response.json["verified"])
        self.assertEqual(response.json["selection_policy"], "explicit_safe_refusal")
        self.assertNotIn("SECRET_MODEL_OUTPUT", json.dumps(response.json))
        self.assertNotIn("PRIVATE", json.dumps(response.json))


if __name__ == "__main__":
    unittest.main()
