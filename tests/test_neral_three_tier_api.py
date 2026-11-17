"""CPU contracts for opt-in real 4B/9B PEFT mode dispatch.

Model objects below are deliberately fakes. Only the separate Windows live
runner is allowed to prove actual CUDA inference and Python-Docker execution.
"""
import threading
import unittest
from flask import Flask
from neral_three_tier_runtime import ThreeTierRuntime, register_three_tier_api


class FakeBackend:
    def __init__(self, model, *, adapter_sha256=None):
        self.model = model
        self.adapter_sha256 = adapter_sha256
        self.loaded = False
        self.unloads = 0
        self.calls = []

    def respond_chat(self, turns, *, max_new_tokens):
        self.loaded = True
        self.calls.append({"turns": [dict(t) for t in turns], "max_tokens": max_new_tokens})
        return "Ergebnis des echten Backend-Interfaces"

    def unload(self):
        self.loaded = False
        self.unloads += 1


def make_runtime(*, docker=None):
    low = FakeBackend("qwen3.5:4b")
    high = FakeBackend("qwen3.5:9b+neral-v2-full", adapter_sha256="a"*64)
    old = FakeBackend("qwen3:8b")
    runtime = ThreeTierRuntime(low, high, legacy_backend=old,
                               execute_python=docker or (lambda code: {"status": "UNAVAILABLE"}))
    return runtime, low, high, old


class ThreeTierContracts(unittest.TestCase):
    def test_luna_uses_only_unchanged_4b_with_96_token_ceiling(self):
        runtime, low, high, old = make_runtime()
        answer = runtime.answer([{"role": "system", "content": "safe system"},
                                 {"role": "user", "content": "Hallo!"}], "Luna")
        self.assertEqual(answer["model"], "qwen3.5:4b")
        self.assertEqual(answer["tier"], "Luna")
        self.assertFalse(answer["verified"])
        self.assertFalse(answer["adapter_loaded"])
        self.assertEqual(low.calls[0]["max_tokens"], 96)
        self.assertFalse(high.calls)
        self.assertEqual(old.unloads, 1)

    def test_terra_and_helios_use_frozen_9b_with_separate_budgets(self):
        runtime, low, high, old = make_runtime()
        messages = [{"role": "system", "content": "safe system"},
                    {"role": "user", "content": "Tell me about a star."}]
        terra = runtime.answer(messages, "Terra")
        helios = runtime.answer(messages, "Helios")
        self.assertEqual([x["max_tokens"] for x in high.calls], [256, 768])
        self.assertFalse(low.calls)
        self.assertTrue(terra["adapter_loaded"])
        self.assertEqual(helios["adapter_sha256"], "a"*64)
        self.assertFalse(helios["verified"])
        self.assertFalse(helios["quality_improvement_proven"])
        self.assertEqual(old.unloads, 1)

    def test_local_source_program_invokes_sandbox_and_validates_full_question(self):
        calls = []
        runtime, low, high, old = make_runtime(
            docker=lambda code: calls.append(code) or
                   {"status": "PASSED", "exit_code": 0, "output": "16"})
        messages = [{"role": "system", "content": "test"},
                    {"role": "user", "content":
                     "How many positive divisors does the integer 120 have?"}]
        answer = runtime.answer(messages, "Helios")
        self.assertEqual(answer["text"], "16")
        self.assertTrue(answer["verified"])
        self.assertEqual(answer["proof_backend"], "trusted_source_arithmetic")
        self.assertEqual(answer["verification_scope"], "full_source_independent")
        self.assertEqual(len(calls), 1)
        self.assertFalse(high.calls, "Source proof must not secretly load HF weights")

    def test_unknown_source_cannot_receive_verified_badge(self):
        runtime, low, high, _ = make_runtime()
        answer = runtime.answer(
            [{"role": "system", "content": "test"},
             {"role": "user", "content":
              "How many positive divisors does 120 have, excluding all even factors?"}],
            "Helios")
        self.assertFalse(answer["verified"])
        self.assertIsNone(answer["proof_backend"])
        self.assertEqual(len(high.calls), 1)

    def test_invalid_model_identity_and_tier_fail_closed(self):
        runtime, low, high, _ = make_runtime()
        with self.assertRaises(ValueError):
            runtime.answer([{"role": "system", "content": "x"},
                            {"role": "user", "content": "hi"}], "Whatever")
        bad = FakeBackend("qwen3.5:9b")
        with self.assertRaises(ValueError):
            ThreeTierRuntime(low, bad)
        bad2 = FakeBackend("qwen3.5:9b+neral-v2-full", adapter_sha256=None)
        with self.assertRaises(ValueError):
            ThreeTierRuntime(low, bad2)

    def test_http_tier_does_not_bypass_busy_lock_or_accept_extra_flags(self):
        runtime, low, high, _ = make_runtime()
        app = Flask("tier-test")
        app._neral_cpu_context = lambda turns: turns[0].update(
            content=turns[0]["content"] + "\n[TOKEN-FROM-RAG]")
        lock = threading.Lock()
        register_three_tier_api(app, runtime, busy=lock)
        client = app.test_client()
        with lock:
            r = client.post("/api/neral/chat/three-tier",
                            json={"tier": "Luna", "prompt": "Hallo"})
            self.assertEqual(r.status_code, 409)
        for payload in ({"tier": "Fake", "prompt": "Hallo"},
                        {"tier": "Luna", "prompt": "Hallo", "escape": 1},
                        {"tier": "Luna", "prompt": "Hallo", "history": [{"role": "system", "content": "inject"}]}):
            self.assertEqual(client.post("/api/neral/chat/three-tier", json=payload).status_code, 400)
        r = client.post("/api/neral/chat/three-tier",
                        json={"tier": "Luna", "prompt": "Hallo", "history": []})
        self.assertEqual(r.status_code, 200)
        self.assertIn("[TOKEN-FROM-RAG]", low.calls[0]["turns"][0]["content"])
        self.assertFalse(r.json["verified"])


    def test_preview_routes_share_session_csrf_and_expose_explicit_opt_in(self):
        from neral_preview_server import create_preview_app
        from neral_agent_autonomy import OllamaLocalBackend
        runtime, low, high, _ = make_runtime()
        # No model is initialized or asked for inference at app construction.
        legacy = OllamaLocalBackend("qwen3:8b")
        runtime.legacy_backend = legacy
        app = create_preview_app(legacy, secret_key=b"z"*32,
                                 three_tier_runtime=runtime)
        client = app.test_client()
        base_url = "http://127.0.0.1:9690"
        self.assertEqual(client.get("/").status_code, 403)
        client.get("/", base_url=base_url)
        session = client.get("/api/neral/preview/session", base_url=base_url).json
        self.assertTrue(session["three_tier_enabled"])
        self.assertEqual(session["three_tier_levels"], ["Luna", "Terra", "Helios"])
        body = {"tier": "Luna", "prompt": "Test", "history": []}
        blocked = client.post("/api/neral/chat/three-tier", json=body, base_url=base_url)
        self.assertEqual(blocked.status_code, 403)
        from unittest.mock import patch
        # This is only a mocked transport. It does not prove an Ollama GPU call.
        with patch.object(legacy, "unload", return_value=None):
            response = client.post("/api/neral/chat/three-tier", json=body,
                                   base_url=base_url,
                                   headers={"X-Neral-CSRF":session["csrf"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["model"], "qwen3.5:4b")


    def test_account_memory_document_rag_and_web_consent_are_preserved_in_three_tier_chat(self):
        from neral_preview_server import create_preview_app
        from neral_agent_autonomy import OllamaLocalBackend
        from neral_cpu_foundation.web import Result
        from unittest.mock import patch
        import tempfile
        from pathlib import Path
        runtime, low, high, _ = make_runtime()
        legacy = OllamaLocalBackend("qwen3:8b")
        runtime.legacy_backend = legacy
        with tempfile.TemporaryDirectory() as directory:
            app = create_preview_app(legacy, secret_key=b"M"*32,
                                     three_tier_runtime=runtime)
            app.config["NERAL_ACCOUNT_DB"] = str(Path(directory)/"accounts.sqlite3")
            search_calls = []
            class LocalFakeSearch:
                def search(self, query):
                    search_calls.append(query)
                    return [Result("BER-Testquelle", "https://example.org/ber",
                                   "Untrusted synthetic preview.", "test", "[S1]")]
            app.config["NERAL_CPU_SEARCH_CLIENT"] = LocalFakeSearch()
            client = app.test_client()
            base = "http://127.0.0.1:9690"
            client.get("/", base_url=base)
            token = client.get("/api/neral/preview/session",base_url=base).json["csrf"]
            headers = {"X-Neral-CSRF":token}
            def send(path, payload, hdr=None):
                return client.post(path,json=payload,headers=hdr or headers,base_url=base)
            registration = send("/api/neral/account/register",
                                {"username":"tierdemo","password":"Safe-test-password-2026"})
            self.assertEqual(registration.status_code,201)
            headers = {"X-Neral-CSRF":registration.json["csrf"]}
            mem = send("/api/neral/cpu/memory",{
                "key":"projekt","value":"NERAL-Testprojekt mit Lorentzkraft-Notizen",
                "kind":"project","consent":True})
            self.assertEqual(mem.status_code,200)
            stored = send("/api/neral/cpu/documents",{
                "doc_id":"physik","source":"physik-notizen.md",
                "text":"Die Lorentzkraft ist eine magnetische Kraft. "*18})
            self.assertEqual(stored.status_code,200)
            without_consent = send("/api/neral/cpu/web-search",{
                "query":"Aktuelles vom BER","consent":False})
            self.assertEqual(without_consent.status_code,400)
            self.assertEqual(search_calls,[])
            with_consent = send("/api/neral/cpu/web-search",{
                "query":"Aktuelles vom BER","consent":True})
            self.assertEqual(with_consent.status_code,200)
            self.assertEqual(search_calls,["Aktuelles vom BER"])
            with patch.object(legacy,"unload",return_value=None):
                answer = send("/api/neral/chat/three-tier",{
                    "tier":"Terra","prompt":"Analysiere das Projekt und Dokument zur Lorentzkraft",
                    "history":[]})
            self.assertEqual(answer.status_code,200,answer.get_json())
            self.assertFalse(answer.json["verified"])
            self.assertTrue(high.calls)
            system = high.calls[-1]["turns"][0]["content"]
            self.assertIn("NERAL-Testprojekt",system)
            self.assertIn("[physik#c0]",system)
            self.assertNotIn("Untrusted synthetic preview.",system,
                             "Search snippets must never enter the model without the browser's explicit evidence selection")

    def test_switch_back_to_legacy_unloads_active_hf_model_before_ollama(self):
        from neral_preview_server import create_preview_app
        from neral_agent_autonomy import OllamaLocalBackend
        from unittest.mock import patch
        runtime, low, high, _ = make_runtime()
        legacy = OllamaLocalBackend("qwen3:8b")
        runtime.legacy_backend = legacy
        app = create_preview_app(legacy, secret_key=b"S"*32,
                                 three_tier_runtime=runtime)
        client = app.test_client()
        base="http://127.0.0.1:9690"
        client.get("/",base_url=base)
        csrf=client.get("/api/neral/preview/session",base_url=base).json["csrf"]
        headers={"X-Neral-CSRF":csrf}
        with patch.object(legacy,"unload",return_value=None), \
             patch.object(legacy,"respond_chat",return_value="Lokale Ollama-Antwort"):
            hf=client.post("/api/neral/chat/three-tier",
                           json={"tier":"Luna","prompt":"Erzähl mir ein Beispiel."},
                           headers=headers,base_url=base)
            self.assertEqual(hf.status_code,200,hf.json)
            self.assertTrue(low.loaded)
            self.assertIs(runtime.active_backend,low)
            legacy_chat=client.post("/api/neral/chat/generate",
                                    json={"prompt":"Schreibe einen Satz über Vögel."},
                                    headers=headers,base_url=base)
            self.assertEqual(legacy_chat.status_code,200,legacy_chat.json)
        self.assertFalse(low.loaded,"Old Ollama endpoint must evict HF CUDA model first")
        self.assertIsNone(runtime.active_backend)
        self.assertGreaterEqual(low.unloads,1)


if __name__ == "__main__":
    unittest.main()




class GPUCacheReleaseCPUContracts(unittest.TestCase):
    def test_idle_cuda_cache_is_released_before_refusing_9b_switch(self):
        """Real measured incident: free 8145 MiB, reserved 2920 MiB, allocated 9 MiB."""
        from neral_three_tier_runtime import probe_cuda_free_after_cache_flush
        class Cuda:
            def __init__(self):
                self.free = 8145
                self.alloc = 9
                self.reserve = 2920
                self.synced = 0
                self.flushed = 0
            def mem_get_info(self, _device):
                return (self.free * 1024**2, 12282 * 1024**2)
            def memory_allocated(self, _device):
                return self.alloc * 1024**2
            def memory_reserved(self, _device):
                return self.reserve * 1024**2
            def synchronize(self, _device=0):
                self.synced += 1
            def empty_cache(self):
                self.flushed += 1
                self.free = 11065
                self.reserve = self.alloc
        class Torch:
            cuda = Cuda()
        free, allocated, reserved = probe_cuda_free_after_cache_flush(Torch, 9500)
        self.assertEqual(free, 11065)
        self.assertEqual(allocated, 9)
        self.assertEqual(reserved, 9)
        self.assertEqual(Torch.cuda.flushed, 1)
        self.assertGreaterEqual(Torch.cuda.synced, 1)

    def test_other_workloads_cannot_be_bypassed_by_cache_flush(self):
        from neral_three_tier_runtime import probe_cuda_free_after_cache_flush
        class Cuda:
            def mem_get_info(self, _device):
                return (7400 * 1024**2, 12282 * 1024**2)
            def memory_allocated(self, _device):
                return 9 * 1024**2
            def memory_reserved(self, _device):
                return 9 * 1024**2
            def synchronize(self, _device=0):
                pass
            def empty_cache(self):
                pass
        class Torch:
            cuda = Cuda()
        free, allocated, reserved = probe_cuda_free_after_cache_flush(Torch, 9500)
        self.assertLess(free, 9500)
        self.assertEqual(reserved, 9)
