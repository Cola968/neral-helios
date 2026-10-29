"""Issue #195: offline-only integration tests; no Ollama, downloads or GPU."""
import pytest
import app as web
from neral_assistant import AssistantUnavailable


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "PREPARATION_MODE", False)
    monkeypatch.setattr(web, "ASSISTANT_ENABLED", True)
    monkeypatch.setattr(web, "ASSISTANT_MODEL", "qwen3.5:9b")
    monkeypatch.setattr(web, "checkpoint_loaded", False)
    monkeypatch.setattr(web, "arctus", None)
    monkeypatch.setattr(web, "chat_store", web.ChatStore(tmp_path / "sync.db"))
    web._hits.clear()
    c = web.app.test_client()
    c.get("/", base_url="http://localhost")
    return c


def post(client, payload):
    return client.post("/api/generate", json=payload, base_url="http://localhost")


def test_local_assistant_keeps_its_provenance_without_native_checkpoint(client, monkeypatch):
    observed = []
    def fake(prompt, history, **kwargs):
        observed.append((prompt, history, kwargs))
        return dict(mode="neral-assistant", source="external-model",
                    model_generated=True, text="Hallo von Qwen",
                    base_model="qwen3.5:9b", model_family="qwen3.5",
                    generated_tokens=6, seconds=1.0, tokens_per_second=6.0,
                    tools_used=[])
    monkeypatch.setattr(web, "assistant_turn", fake)
    response = post(client, {"mode": "assistant", "prompt": "Hallo", "history": [],
                             "stream": False, "runtime_id": "stale-native-runtime"})
    assert response.status_code == 200
    result = response.json
    assert result["text"] == "Hallo von Qwen"
    assert result["source"] == "external-model"
    assert result["base_model"] == "qwen3.5:9b"
    assert result["mode"] == "neral-assistant"
    assert result["model_family"] == "qwen3.5"
    assert observed[0][2]["model"] == "qwen3.5:9b"
    assert not web._generation_lock.locked()


def test_assistant_is_explicitly_opt_in_and_offline_returns_503(client, monkeypatch):
    monkeypatch.setattr(web, "ASSISTANT_ENABLED", False)
    assert post(client, {"mode": "assistant", "prompt": "Hallo"}).status_code == 503
    assert web.info_payload()["assistant"]["enabled"] is False
    monkeypatch.setattr(web, "ASSISTANT_ENABLED", True)
    monkeypatch.setattr(web, "assistant_turn", lambda *a, **k: (_ for _ in ()).throw(AssistantUnavailable("offline")))
    response = post(client, {"mode": "assistant", "prompt": "Hallo"})
    assert response.status_code == 503 and "Ollama" in response.json["error"]
    assert not web._generation_lock.locked()


def test_assistant_refuses_stream_illusion_and_foreign_history(client, monkeypatch):
    monkeypatch.setattr(web, "assistant_turn", lambda *a, **k: pytest.fail("must not invoke Ollama"))
    assert post(client, {"mode": "assistant", "prompt": "Hallo", "stream": True}).status_code == 400
    response = post(client, {"mode": "assistant", "prompt": "Weiter", "history": [
        {"role": "user", "content": "Frage"},
        {"role": "assistant", "content": "Arctus-Ausgabe", "model_family": "neral-arctus350"},
    ]})
    assert response.status_code == 400


def test_assistant_results_remain_separate_in_chat_sync(client):
    chat = {"title": "Qwen", "mode": "assistant", "messages": [
        {"role": "user", "content": "Frage", "mode": "assistant"},
        {"role": "assistant", "content": "Qwen-Antwort", "mode": "neral-assistant",
         "source": "external-model", "base_model": "qwen3.5:9b", "model_family": "qwen3.5",
         "imported": False, "interrupted": False},
    ]}
    response = client.post("/api/chats/abcdefghijklmnop",
                           json={"revision": 0, "chat": chat}, base_url="http://localhost")
    assert response.status_code == 200
    saved = client.get("/api/chats", base_url="http://localhost").json["chats"][0]["chat"]
    assert saved["mode"] == "assistant"
    assert saved["messages"][1]["base_model"] == "qwen3.5:9b"
    assert saved["messages"][1]["source"] == "external-model"
