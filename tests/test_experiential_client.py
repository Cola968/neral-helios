import json

import pytest

import experiential_client as client


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return json.dumps(self.payload).encode("utf-8")


def test_missing_key_is_safe(monkeypatch):
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    monkeypatch.delenv("EXPERIENTIAL_API_KEY", raising=False)
    with pytest.raises(client.ExperientialError, match="EXPLABS_API_KEY"):
        client.chat([{"role": "user", "content": "Hallo"}])


def test_opus_request_keeps_key_server_side_and_omits_pinned_sampling(monkeypatch):
    monkeypatch.setenv("EXPLABS_API_KEY", "xpl_test_secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({
            "model": "claude-opus-5",
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    result = client.chat([{"role": "user", "content": "Hallo"}], max_tokens=2048)

    assert captured["url"] == "https://api.experientiallabs.ai/v1/chat/completions"
    assert captured["authorization"] == "Bearer xpl_test_secret"
    assert captured["body"]["model"] == "claude-opus-5"
    assert captured["body"]["max_tokens"] == 2048
    assert "temperature" not in captured["body"]
    assert "top_p" not in captured["body"]
    assert result["text"] == "OK"
