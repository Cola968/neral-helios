"""Launch readiness API respects local session and does not certify compliance."""
import app


def test_launch_check_read_only_and_session_protected():
    client = app.app.test_client()
    anonymous = client.get("/api/launch-readiness", base_url="http://localhost")
    assert anonymous.status_code == 401
    assert client.get("/", base_url="http://localhost").status_code == 200
    response = client.get("/api/launch-readiness", base_url="http://localhost")
    assert response.status_code == 200
    result = response.get_json()
    assert result["scope"] == "static-local-source-scan"
    assert result["launch_ready"] is False
    assert result["is_legal_certification"] is False
    assert len(result["checks"]) == 6
    assert {check["id"] for check in result["checks"]} == {
        "minors", "remote_fonts", "session_replay", "marketing_email",
        "subscriptions", "user_uploads"
    }
    assert response.headers["Cache-Control"] == "no-store"
    assert "'self'" in response.headers["Content-Security-Policy"]
    assert client.get("/api/launch-readiness", base_url="http://localhost",
                      headers={"Origin": "https://attacker.invalid"}).status_code == 403
