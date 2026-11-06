"""Computer Use HTTP security boundary: loopback only, no paired-phone gateway."""
from flask import Flask
import pytest

from computer_api import register_computer_api


class FakeDesktop:
    def __init__(self):
        self.calls=[]
    def screenshot(self):
        return {"image_base64":"aW1hZ2U=", "media_type":"image/png",
                "width":800, "height":600}
    def execute(self,action):
        self.calls.append(action)


@pytest.fixture
def client():
    backend=FakeDesktop()
    app=Flask("test-computer-use")
    register_computer_api(app, enabled=True, backend=backend, clock=lambda:1000)
    return app.test_client(),backend


def test_opt_in_http_preview_approve_and_screenshot(client):
    http,backend=client
    assert http.post("/api/computer/screenshot",json={}).status_code==400
    assert http.post("/api/computer/start",json={}).json["enabled"] is True
    frame=http.post("/api/computer/screenshot",json={})
    assert frame.status_code==200
    assert frame.json["width"]==800
    p=http.post("/api/computer/preview",json={"action":"click","x":12,"y":20})
    assert p.status_code==200
    assert backend.calls==[]
    assert http.post("/api/computer/execute",json={"approval_token":"fake"}).status_code==400
    ok=http.post("/api/computer/execute",json={"approval_token":p.json["approval_token"]})
    assert ok.status_code==200 and ok.json["executed"] is True
    assert backend.calls == [{"action":"click","x":12,"y":20}]
    assert http.post("/api/computer/stop",json={}).json["enabled"] is False
    assert http.post("/api/computer/preview",json={"action":"click","x":12,"y":20}).status_code==400


def test_gateway_is_denied_even_when_pair_token_valid(client):
    http,_=client
    for path in ("/api/computer/status","/api/computer/start","/api/computer/screenshot"):
        assert http.post(path,json={},headers={"X-Helios-Gateway":"gateway-token"}).status_code==403
    assert http.post("/api/computer/start",json={},environ_overrides={"REMOTE_ADDR":"192.168.1.5"}).status_code==403


def test_cross_site_and_invalid_json_blocked(client):
    http,_=client
    assert http.post("/api/computer/start",json={},headers={"Origin":"https://evil.invalid"}).status_code==403
    assert http.post("/api/computer/start",data="invalid").status_code==415
    assert http.post("/api/computer/start",json=[]).status_code==400


def test_disabled_api_cannot_take_screenshot_or_start():
    app=Flask("test-disabled-computer-use")
    register_computer_api(app,enabled=False)
    c=app.test_client()
    assert c.post("/api/computer/status",json={}).json["enabled"] is False
    for path in ("/api/computer/start","/api/computer/screenshot",
                 "/api/computer/preview","/api/computer/execute"):
        assert c.post(path,json={}).status_code==403
