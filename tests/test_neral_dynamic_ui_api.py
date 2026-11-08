"""NERAL Dynamic UI must never bypass the existing local chat guardrails."""
from __future__ import annotations
import unittest

from neral_local_chat_api import _messages, _ui_instruction
from neral_preview_server import create_preview_app


class FakeLocalModel:
    model="qwen3:8b"
    def __init__(self):
        self.calls=[]
    def respond_chat(self,turns):
        self.calls.append(turns)
        return "Hier ist der Rechner."
    def respond_think(self,turns):
        self.calls.append(turns)
        return "Kurze Antwort."
    def respond(self,turns):
        self.calls.append(turns)
        return "{}"


class DynamicUIAPITests(unittest.TestCase):
    def setUp(self):
        self.backend=FakeLocalModel()
        self.app=create_preview_app(self.backend,secret_key=b"D"*32)
        self.client=self.app.test_client()
        base="http://127.0.0.1:9690"
        self.client.get("/",base_url=base)
        session=self.client.get("/api/neral/preview/session",base_url=base)
        self.headers={"X-Neral-CSRF":session.json["csrf"]}
        self.base=base

    def test_ordinary_chat_has_no_ui_system_instructions(self):
        request={"prompt":"Hallo"}
        reply=self.client.post("/api/neral/chat/generate",
                               json=request,headers=self.headers,base_url=self.base)
        self.assertEqual(reply.status_code,200)
        self.assertEqual(self.backend.calls[-1][-1]["content"],"Hallo")
        self.assertNotIn("OPTIONALE INTERAKTIVE DARSTELLUNG",
                         self.backend.calls[-1][0]["content"])

    def test_ui_hint_goes_only_to_system_turn_not_user_text(self):
        prompt="Erstelle ein interaktives Quiz über Elektrizität"
        reply=self.client.post("/api/neral/chat/generate",
            json={"prompt":prompt,"ui_intent":"quiz"},headers=self.headers,base_url=self.base)
        self.assertEqual(reply.status_code,200)
        self.assertEqual(reply.json["text"],"Hier ist der Rechner.")
        turns=self.backend.calls[-1]
        self.assertEqual(turns[-1]["content"],prompt)
        self.assertIn("neral-ui",turns[0]["content"])
        self.assertIn("OPTIONALE INTERAKTIVE DARSTELLUNG",turns[0]["content"])
        self.assertEqual(reply.json["model_origin"],"local_foundation_not_neral_adapter")

    def test_think_still_accepts_declarative_intent(self):
        reply=self.client.post("/api/neral/chat/think",
            json={"prompt":"Quiz über Europa","ui_intent":"quiz"},
            headers=self.headers,base_url=self.base)
        self.assertEqual(reply.status_code,200)
        self.assertEqual(self.backend.calls[-1][-1]["content"],"Quiz über Europa")

    def test_unknown_ui_intent_is_rejected_before_model_inference(self):
        for kind in ["javascript","html","../../../secrets",123,{"role":"system"}]:
            result=self.client.post("/api/neral/chat/generate",
                json={"prompt":"Test","ui_intent":kind},
                headers=self.headers,base_url=self.base)
            self.assertEqual(result.status_code,400)
        self.assertEqual(self.backend.calls,[])

    def test_unknown_payload_properties_and_missing_csrf_remain_blocked(self):
        for payload in [
            {"prompt":"Frage","ui_intent":"quiz","javascript":"alert(1)"},
            {"prompt":"Frage","ui_intent":None}
        ]:
            result=self.client.post("/api/neral/chat/generate",
                json=payload,headers=self.headers,base_url=self.base)
            self.assertEqual(result.status_code,400)
        no_csrf=self.client.post("/api/neral/chat/generate",
            json={"prompt":"Frage","ui_intent":"calculator"},base_url=self.base)
        self.assertEqual(no_csrf.status_code,403)
        self.assertEqual(self.backend.calls,[])


if __name__=="__main__":
    unittest.main()
