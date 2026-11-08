"""Opt-in Qwen3.5 HIGH must route real local requests without breaking default LOW."""
import unittest
from unittest.mock import patch

from neral_agent_autonomy import OllamaLocalBackend, AgentModelError
from neral_preview_server import create_preview_app

HOST="http://127.0.0.1:9690"

class Backend(OllamaLocalBackend):
    def __init__(self, model, *, profile="legacy", events=None, fail_unload=False):
        super().__init__(model,inference_profile=profile,timeout=90)
        self.events=events if events is not None else []
        self.fail_unload=fail_unload
    def respond_chat(self, turns):
        self.events.append(("generate",self.model))
        return f"Antwort von {self.model}."
    def respond_think(self, turns):
        self.events.append(("think",self.model))
        return f"Ergebnis von {self.model}. FINAL_ANSWER: 7"
    def unload(self):
        self.events.append(("unload",self.model))
        if self.fail_unload:
            raise AgentModelError("Local unload refused")

def web(backend, high=None):
    app=create_preview_app(backend,experimental_high_backend=high)
    cli=app.test_client()
    cli.get("/",base_url=HOST)
    csrf=cli.get("/api/neral/preview/session",base_url=HOST).json["csrf"]
    return cli,{"X-Neral-CSRF":csrf}

class DualModelHighContracts(unittest.TestCase):
    def test_default_remains_one_model_and_no_extra_unloads(self):
        events=[]
        low=Backend("qwen3:8b",events=events)
        cli,hdr=web(low)
        stat=cli.get("/api/neral/chat/status",base_url=HOST).json
        self.assertEqual(stat["model"],"qwen3:8b")
        self.assertEqual(stat["experimental_high_model"],None)
        self.assertFalse(stat["experimental_high_enabled"])
        r=cli.post("/api/neral/chat/think",json={"prompt":"Erkläre Sternentwicklung."},
                    headers=hdr,base_url=HOST)
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json["model"],"qwen3:8b")
        self.assertFalse(any(a=="unload" for a,m in events))

    def test_real_low_vs_high_routing_and_honest_provenance(self):
        events=[]
        low=Backend("qwen3:8b",events=events)
        high=Backend("qwen3.5:9b",profile="measured_nonthinking_2048",events=events)
        cli,hdr=web(low,high)
        stat=cli.get("/api/neral/chat/status",base_url=HOST).json
        self.assertTrue(stat["experimental_high_enabled"])
        self.assertEqual(stat["experimental_high_model"],"qwen3.5:9b")
        sess=cli.get("/api/neral/preview/session",base_url=HOST).json
        self.assertEqual(sess["model"],"qwen3:8b")
        self.assertEqual(sess["experimental_high_model"],"qwen3.5:9b")
        first=cli.post("/api/neral/chat/generate",json={"prompt":"Erkläre Sterne."},
                       headers=hdr,base_url=HOST)
        self.assertEqual(first.status_code,200)
        self.assertEqual(first.json["model"],"qwen3:8b")
        self.assertEqual(first.json["mode"],"LOW")
        thinking=cli.post("/api/neral/chat/think",json={"prompt":"Erkläre Sternentwicklung."},
                          headers=hdr,base_url=HOST)
        self.assertEqual(thinking.status_code,200)
        self.assertEqual(thinking.json["mode"],"HIGH")
        self.assertEqual(thinking.json["model"],"qwen3.5:9b")
        self.assertEqual(thinking.json["inference_profile"],"measured_nonthinking_2048")
        self.assertFalse(thinking.json["adapter_loaded"])
        self.assertEqual(thinking.json["model_origin"],"local_foundation_not_neral_adapter")
        again=cli.post("/api/neral/chat/generate",json={"prompt":"Erkläre Sterne."},
                       headers=hdr,base_url=HOST)
        self.assertEqual(again.status_code,200)
        self.assertEqual(again.json["model"],"qwen3:8b")
        self.assertEqual([m for a,m in events if a=="unload"],["qwen3.5:9b","qwen3:8b","qwen3.5:9b"])
        self.assertTrue(any(a=="think" and m=="qwen3.5:9b" for a,m in events))

    def test_fail_closed_when_model_cannot_unload(self):
        events=[]
        low=Backend("qwen3:8b",events=events,fail_unload=True)
        high=Backend("qwen3.5:9b",profile="measured_nonthinking_2048",events=events)
        cli,hdr=web(low,high)
        first=cli.post("/api/neral/chat/generate",json={"prompt":"Erkläre Sterne."},
                       headers=hdr,base_url=HOST)
        self.assertEqual(first.status_code,200)
        blocked=cli.post("/api/neral/chat/think",json={"prompt":"Erkläre Sternentwicklung."},
                         headers=hdr,base_url=HOST)
        self.assertEqual(blocked.status_code,503)
        self.assertFalse(any(a=="think" and m=="qwen3.5:9b" for a,m in events))
        self.assertNotIn("Local unload refused",str(blocked.json))

    def test_experimental_high_requires_exact_models_and_profile(self):
        bad=[
            (Backend("qwen3.5:9b"),Backend("qwen3.5:9b",profile="measured_nonthinking_2048")),
            (Backend("qwen3:8b"),Backend("qwen3:8b",profile="measured_nonthinking_2048")),
            (Backend("qwen3:8b"),Backend("qwen3.5:9b")),
            (Backend("qwen3:8b"),object())
        ]
        for low,high in bad:
            with self.subTest(model=getattr(high,"model",None)),self.assertRaises(ValueError):
                create_preview_app(low,experimental_high_backend=high)

    def test_exact_high_tools_need_no_model_or_unload(self):
        events=[]
        low=Backend("qwen3:8b",events=events)
        high=Backend("qwen3.5:9b",profile="measured_nonthinking_2048",events=events)
        cli,hdr=web(low,high)
        r=cli.post("/api/neral/chat/think",json={"prompt":"Calculate 17*13"},
                   headers=hdr,base_url=HOST)
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json["text"],"221")
        self.assertFalse(r.json["model_generated"])
        self.assertTrue(r.json["verified"])
        self.assertEqual(r.json["model"],"qwen3.5:9b")
        self.assertFalse(events)

    def test_cli_start_fails_closed_for_missing_secondary_model(self):
        from neral_preview_server import build_experimental_high
        self.assertIsNone(build_experimental_high("qwen3:8b",False))
        with patch("neral_preview_server._installed_ollama_models",
                   return_value={"qwen3:8b":"sha256:"+"a"*64}):
            with self.assertRaises(ValueError):
                build_experimental_high("qwen3:8b",True)
        with patch("neral_preview_server._installed_ollama_models",
                   return_value={
                       "qwen3:8b":"sha256:"+"a"*64,
                       "qwen3.5:9b":"sha256:"+"b"*64}):
            high=build_experimental_high("qwen3:8b",True)
            self.assertEqual(high.model,"qwen3.5:9b")
            self.assertEqual(high.inference_profile,"measured_nonthinking_2048")
        with self.assertRaises(ValueError):
            build_experimental_high("qwen3.5:9b",True)

if __name__=="__main__":unittest.main()
