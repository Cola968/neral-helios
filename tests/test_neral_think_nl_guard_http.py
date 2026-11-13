"""RED: HTTP THINK opt-in bridge cannot bypass full-source PR537 evidence.

Uses Flask test client, fake local extractors; NO Ollama/RTX, no sealed100.
"""
import json
import os
import unittest
from unittest.mock import patch

from flask import Flask
from neral_agent_autonomy import OllamaLocalBackend
from neral_local_chat_api import register_neral_local_chat_api
from neral_nl_verified_route import strict_source_certificate
from scripts.neral_nl_schema_probe import old_case_question


def fixture():
    app = Flask("strict-source-route-cpu-only")
    backend = OllamaLocalBackend(
        "qwen3.5:9b", inference_profile="measured_nonthinking_2048")
    base_calls = []
    def fallback(turns):
        base_calls.append(turns)
        return "UNTOUCHED_BASE"
    backend.respond_chat = fallback
    register_neral_local_chat_api(app, backend)
    return app.test_client(), base_calls


class ThinkNLGuardIntegrationCPU(unittest.TestCase):
    def test_route_disabled_without_both_explicit_flag_and_env(self):
        client, raw = fixture()
        question = old_case_question("21453")
        r = client.post("/api/neral/chat/think", json={
            "prompt": question, "nl_schema_consensus": True})
        self.assertEqual(r.status_code, 403)
        self.assertFalse(raw)

        # Without the explicit opt-in flag, the ordinary THINK route is
        # unchanged. It is intentionally not invoked by this CPU test.

    def test_missing_flag_type_and_mixed_modes_rejected(self):
        client, raw = fixture()
        q = old_case_question("21453")
        for p in (
            {"prompt": q, "nl_schema_consensus": "true"},
            {"prompt": q, "nl_schema_consensus": True, "dual_routing": True},
            {"prompt": q, "nl_schema_consensus": True, "experimental_tir": True},
            {"prompt": q, "nl_schema_consensus": True, "evaluation_final_answer": True},
        ):
            r = client.post("/api/neral/chat/think", json=p)
            self.assertEqual(r.status_code, 400, (p, r.json))
        self.assertFalse(raw)

    def test_real_http_handler_selects_only_three_way_proven_source(self):
        client, raw_calls = fixture()
        question = old_case_question("21453")
        canonical = strict_source_certificate(question)
        models = []
        def fake_maker(model):
            models.append(model)
            return lambda turns: json.dumps(canonical)
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_NL_SCHEMA":"1"}), \
             patch("scripts.neral_nl_schema_probe.make_local_extractor",
                   side_effect=fake_maker):
            resp = client.post("/api/neral/chat/think", json={
                "prompt": question, "nl_schema_consensus": True})
        self.assertEqual(resp.status_code, 200, resp.json)
        self.assertEqual(resp.json["text"], "1")
        self.assertTrue(resp.json["safe_to_promote"])
        self.assertTrue(resp.json["verified"])
        self.assertFalse(resp.json["model_generated"])
        self.assertEqual(models,
                         ["neral-qwen35-qlora-v1:latest", "qwen3.5:9b"])
        self.assertFalse(raw_calls)

    def test_http_wrong_agreement_keeps_untouched_base_fallback(self):
        client, raw_calls = fixture()
        q = old_case_question("logic-06")
        fake = strict_source_certificate(q)
        fake["constraints"].pop()
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_NL_SCHEMA":"1"}), \
             patch("scripts.neral_nl_schema_probe.make_local_extractor",
                   return_value=lambda turns: json.dumps(fake)):
            resp = client.post("/api/neral/chat/think", json={
                "prompt": q, "nl_schema_consensus": True})
        self.assertEqual(resp.status_code, 200, resp.json)
        self.assertEqual(resp.json["text"], "UNTOUCHED_BASE")
        self.assertFalse(resp.json["verified"])
        self.assertFalse(resp.json["safe_to_promote"])
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(raw_calls[0][-1]["content"], q)

    def test_http_unknown_source_calls_base_only(self):
        client, raw_calls = fixture()
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_NL_SCHEMA":"1"}), \
             patch("scripts.neral_nl_schema_probe.make_local_extractor",
                   return_value=lambda turns: self.fail("Never infer for unknown source")):
            resp = client.post("/api/neral/chat/think", json={
                "prompt": "Hallo NERAL", "nl_schema_consensus": True})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["text"], "UNTOUCHED_BASE")
        self.assertEqual(len(raw_calls), 1)
        self.assertFalse(resp.json["safe_to_promote"])

    def test_noneligible_backend_rejected_for_experimental_route(self):
        app = Flask("noneligible-neral")
        class Fake:
            model = "generic-adapter"
            def respond(self, turns):
                return "test"
        register_neral_local_chat_api(app, Fake())
        with patch.dict(os.environ, {"NERAL_EXPERIMENTAL_NL_SCHEMA":"1"}):
            reply=app.test_client().post("/api/neral/chat/think",json={
                "prompt":old_case_question("21453"),"nl_schema_consensus":True})
        self.assertEqual(reply.status_code, 409)


if __name__ == "__main__":
    unittest.main()
