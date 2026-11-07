"""NERAL Lab: genuine paired adapter OFF/ON comparisons, not rebranded answers."""
import contextlib
import json
import threading
import time
from pathlib import Path

import pytest
from flask import Flask

from neral_agent_hf import HFAdapterLocalBackend
from neral_local_chat_api import register_neral_local_chat_api


def fake_backend(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fake never for deployment")
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
    return HFAdapterLocalBackend(base, adapter)


class ModelWithSwitch:
    def __init__(self):
        self.disabled = False
        self.switches = 0

    @contextlib.contextmanager
    def disable_adapter(self):
        assert not self.disabled
        self.disabled = True
        self.switches += 1
        try:
            yield
        finally:
            self.disabled = False


def test_same_prompt_and_decode_path_with_real_disable_adapter(tmp_path):
    backend = fake_backend(tmp_path)
    model = ModelWithSwitch()
    backend._model = model
    seen = []

    def fake_generate(messages):
        seen.append((tuple((m["role"], m["content"]) for m in messages), model.disabled))
        return "basis-antwort" if model.disabled else "adapter-antwort"

    backend._respond_once = fake_generate
    msg = [{"role": "system", "content": "neutral"}, {"role": "user", "content": "Korrigiere diesen Code"}]
    result = backend.compare_with_base(msg)
    assert result == {"base": "basis-antwort", "neral": "adapter-antwort"}
    assert seen == [(tuple((m["role"], m["content"]) for m in msg), True),
                    (tuple((m["role"], m["content"]) for m in msg), False)]
    assert model.disabled is False
    assert model.switches == 1


def test_serialization_prevents_interleaving_between_base_and_adapter(tmp_path):
    backend = fake_backend(tmp_path)
    backend._model = ModelWithSwitch()
    events = []
    started = threading.Event()

    def generate(messages):
        events.append("base" if backend._model.disabled else "neral")
        started.set()
        time.sleep(0.04)
        return events[-1]

    backend._respond_once = generate
    first = threading.Thread(target=lambda: backend.compare_with_base([{"role": "user", "content": "test"}]))
    first.start()
    assert started.wait(2)
    second = threading.Thread(target=lambda: backend.respond([{"role": "user", "content": "other"}]))
    second.start()
    first.join(3)
    second.join(3)
    assert events == ["base", "neral", "neral"]


def test_ab_endpoint_returns_only_real_comparison_and_does_not_claim_superiority(tmp_path):
    backend = fake_backend(tmp_path)
    backend._model = ModelWithSwitch()
    calls = []

    def generate(messages):
        calls.append((messages, backend._model.disabled))
        return "Unverändert" if backend._model.disabled else "NERAL-Antwort"

    backend._respond_once = generate
    app = Flask("neral-ab")
    register_neral_local_chat_api(app, backend)
    client = app.test_client()
    result = client.post("/api/neral/chat/compare", json={"prompt": "Löse das Problem."})
    assert result.status_code == 200
    packet = result.json
    assert packet["base"] == "Unverändert"
    assert packet["neral"] == "NERAL-Antwort"
    assert packet["model_generated"] is True
    assert packet["same_prompt"] is True
    assert packet["same_decoding"] is True
    assert packet["quality_improvement_proven"] is False
    assert packet["adapter_loaded"] is True
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert "NERAL" not in calls[0][0][0]["content"]
    assert "Qwen" not in calls[0][0][0]["content"]
    assert len(calls[0][0]) == 2


def test_compare_is_disabled_when_no_verifiable_adapter():
    class Foundation:
        model = "qwen3.5:9b"
        def respond_chat(self, messages):
            return "fake"
    app = Flask("no-adapter")
    register_neral_local_chat_api(app, Foundation())
    client = app.test_client()
    status = client.get("/api/neral/chat/status").json
    assert status["compare_available"] is False
    response = client.post("/api/neral/chat/compare", json={"prompt": "test"})
    assert response.status_code == 409
    assert "base" not in response.json


def test_compare_requires_only_one_valid_prompt(tmp_path):
    backend = fake_backend(tmp_path)
    app = Flask("invalid-compare")
    register_neral_local_chat_api(app, backend)
    client = app.test_client()
    for payload in [{"prompt": ""}, {"prompt": "x" * 4097},
                    {"prompt": "ok", "history": []},
                    {"prompt": "ok", "system": "forget everything"}]:
        response = client.post("/api/neral/chat/compare", json=payload)
        assert response.status_code == 400
    response = client.post("/api/neral/chat/compare", data="nope", content_type="text/plain")
    assert response.status_code == 415
    assert backend._model is None


def test_model_switch_must_be_supported_and_exception_leaves_no_leaked_result(tmp_path):
    backend = fake_backend(tmp_path)
    backend._model = object()
    app = Flask("no-switch")
    register_neral_local_chat_api(app, backend)
    result = app.test_client().post("/api/neral/chat/compare", json={"prompt": "x"})
    assert result.status_code == 503
    assert "base" not in result.json


def test_ui_exposes_actual_same_prompt_comparison_without_html_injection():
    root = Path(__file__).resolve().parents[1]
    html = (root / "templates/index.html").read_text(encoding="utf-8")
    js = (root / "static/neral-local-chat.js").read_text(encoding="utf-8")
    assert 'id="neralLocalChatCompare"' in html
    assert 'id="neralCompareBase"' in html
    assert 'id="neralCompareNeral"' in html
    assert "req('compare'" in js
    assert "textContent" in js
    assert "innerHTML" not in js
    assert "gleicher Prompt" in html.lower() or "gleicher Prompt" in js


def test_ab_endpoint_inherits_local_session_origin_protection(monkeypatch):
    import app as runtime
    client = runtime.app.test_client()
    assert client.post("/api/neral/chat/compare", json={"prompt": "Hallo"},
                       base_url="http://localhost").status_code == 401
    client.get("/", base_url="http://localhost")
    for headers in ({"Origin": "https://evil.invalid"}, {"Host": "evil.invalid"},
                    {"Sec-Fetch-Site": "cross-site"}):
        assert client.post("/api/neral/chat/compare", json={"prompt": "Hallo"},
                           headers=headers, base_url="http://localhost").status_code == 403
