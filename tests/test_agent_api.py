"""HTTP boundary: localhost app session middleware is inherited by agent routes."""
import hashlib

import pytest
from flask import Flask

from agent_api import register_agent_api


@pytest.fixture
def client(tmp_path):
    (tmp_path / "entry.py").write_text("x=1\n", encoding="utf-8")
    app = Flask("agent-api-smoke")
    register_agent_api(app, tmp_path)
    return app.test_client()


def test_capabilities_and_plan_http(client):
    result = client.get("/api/agent/capabilities")
    assert result.status_code == 200
    assert result.json["external_model"] is False
    started = client.post("/api/agent/start", json={"goal": "Prüfe", "strategy": "audit"})
    assert started.status_code == 201
    ident = started.json["id"]
    assert client.get(f"/api/agent/{ident}").status_code == 200
    for _ in range(6):
        progress = client.post(f"/api/agent/{ident}/next", json={})
    assert progress.json["status"] == "completed"
    assert client.post(f"/api/agent/{ident}/next", json={}).json["status"] == "completed"


def test_bad_json_and_invalid_id_never_500(client):
    assert client.post("/api/agent/start", data="hello").status_code == 415
    assert client.post("/api/agent/start", json=[]).status_code == 400
    assert client.post("/api/agent/start", json={"goal": ""}).status_code == 400
    assert client.post("/api/agent/wrong/next", json={}).status_code == 400


def test_http_patch_approval_and_undo(client, tmp_path):
    file = tmp_path / "entry.py"
    started = client.post("/api/agent/start", json={"goal": "Aktualisiere"})
    ident = started.json["id"]
    info = client.post("/api/agent/check", json={"path": "entry.py"})
    assert info.status_code == 200
    assert info.json["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    preview = client.post(f"/api/agent/{ident}/propose", json={
        "path": "entry.py", "content": "x=2\n", "expected_sha": info.json["sha256"]
    })
    assert preview.status_code == 200
    assert file.read_text() == "x=1\n"
    assert client.post(f"/api/agent/{ident}/approve", json={"token": "forged"}).status_code == 400
    token = preview.json["proposal"]["token"]
    assert client.post(f"/api/agent/{ident}/approve", json={"token": token}).status_code == 200
    assert file.read_text() == "x=2\n"
    assert client.post(f"/api/agent/{ident}/undo", json={}).status_code == 200
    assert file.read_text() == "x=1\n"


def test_http_cancel_and_resume(client):
    task = client.post("/api/agent/start", json={"goal": "Check"}).json["id"]
    assert client.post(f"/api/agent/{task}/pause", json={}).json["status"] == "paused"
    assert client.post(f"/api/agent/{task}/resume", json={}).json["status"] == "running"
    assert client.post(f"/api/agent/{task}/cancel", json={}).json["status"] == "cancelled"
    assert client.post(f"/api/agent/{task}/next", json={}).json["status"] == "cancelled"


def test_malicious_paths_not_disclosed(client):
    assert client.post("/api/agent/check", json={"path": "../test.py"}).status_code == 400
    assert client.post("/api/agent/check", json={"path": ".env"}).status_code == 400

def test_http_reviewed_quick_fix_after_audit(client, tmp_path):
    target = tmp_path / "broken.py"
    target.write_text("def answer()\n    return 42\n")
    started = client.post("/api/agent/start", json={"goal": "Prüfe Python", "strategy": "quality"})
    task = started.json["id"]
    for _ in range(8):
        finished = client.post(f"/api/agent/{task}/next", json={})
        if finished.json["status"] == "completed":
            break
    assert finished.json["status"] == "completed"
    preview = client.post(f"/api/agent/{task}/quickfix", json={
        "path": "broken.py", "kind": "python_colon"
    })
    assert preview.status_code == 200
    assert preview.json["status"] == "awaiting_approval"
    assert "def answer():" in preview.json["proposal"]["diff"]
    assert target.read_text() == "def answer()\n    return 42\n"
    assert client.post(f"/api/agent/{task}/approve", json={"token": "invalid"}).status_code == 400
    approve = client.post(f"/api/agent/{task}/approve", json={
        "token": preview.json["proposal"]["token"]
    })
    assert approve.status_code == 200
    assert approve.json["status"] == "completed"
    assert target.read_text() == "def answer():\n    return 42\n"
    assert client.post(f"/api/agent/{task}/undo", json={}).status_code == 200
    assert target.read_text() == "def answer()\n    return 42\n"


def test_http_quickfix_invalid_input_never_writes(client, tmp_path):
    original = (tmp_path / "entry.py").read_bytes()
    task = client.post("/api/agent/start", json={"goal": "Review"}).json["id"]
    for payload in [
        {"path": "../outside.py", "kind": "python_colon"},
        {"path": "entry.py", "kind": "execute_script"},
        {"path": "entry.py", "kind": "python_colon"},
        {"path": "entry.py", "kind": None},
    ]:
        response = client.post(f"/api/agent/{task}/quickfix", json=payload)
        assert response.status_code == 400
        assert (tmp_path / "entry.py").read_bytes() == original
