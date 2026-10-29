"""Prelaunch legal/privacy regression guard (offline, no network, no model)."""
from pathlib import Path
import shutil
import pytest

from launch_check import audit


@pytest.fixture
def surface(tmp_path):
    (tmp_path / "templates").mkdir()
    (tmp_path / "static").mkdir()
    (tmp_path / "billing").mkdir()
    (tmp_path / "templates" / "index.html").write_text(
        '<html><head><link rel="stylesheet" href="/static/neral.css">'
        '<script defer src="/static/neral.js"></script></head><body></body></html>',
        encoding="utf8")
    (tmp_path / "static" / "neral.css").write_text(
        "body{font-family:system-ui,sans-serif}", encoding="utf8")
    (tmp_path / "static" / "chat-experience.css").write_text(
        "body{color:#fff}", encoding="utf8")
    (tmp_path / "static" / "neral.js").write_text(
        "const local = true;", encoding="utf8")
    return tmp_path


def by_id(report):
    return {check["id"]: check for check in report["checks"]}


def test_no_claim_of_full_legal_compliance_for_local_app(surface):
    report = audit(surface)
    assert len(report["checks"]) == 6
    assert report["scope"] == "static-local-source-scan"
    assert report["launch_ready"] is False
    assert report["is_legal_certification"] is False
    assert by_id(report)["remote_fonts"]["status"] == "pass"
    assert by_id(report)["session_replay"]["status"] == "pass"
    assert all(by_id(report)[key]["status"] == "review" for key in (
        "minors", "marketing_email", "subscriptions", "user_uploads"))


@pytest.mark.parametrize("html", [
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter">',
    '<script src="https://cdn.example.com/analytics.js"></script>',
    '<link href="https://example.com/remote.css" rel="stylesheet">',
])
def test_remote_html_dependencies_are_blockers(surface, html):
    template = surface / "templates" / "index.html"
    template.write_text(template.read_text() + html)
    report = audit(surface)
    assert by_id(report)["remote_fonts"]["status"] == "blocker"
    assert report["blocker_count"] > 0


@pytest.mark.parametrize("css", [
    '@import url("https://fonts.googleapis.com/css2?family=Inter");',
    "body{background-image:url(https://tracker.invalid/pixel)}",
    "@font-face{src:url('https://fonts.gstatic.com/file.woff2')}",
])
def test_external_css_dependencies_are_blockers(surface, css):
    (surface / "static" / "neral.css").write_text(css)
    assert by_id(audit(surface))["remote_fonts"]["status"] == "blocker"


@pytest.mark.parametrize("text", [
    'window.clarity("set", "x", "y")',
    'import {record} from "rrweb"',
    'LogRocket.init("fake-project")',
    'const audit = "https://static.hotjar.com/script.js"',
])
def test_session_replay_signatures_blocked(surface, text):
    (surface / "static" / "neral.js").write_text(text)
    assert by_id(audit(surface))["session_replay"]["status"] == "blocker"


def test_missing_frontend_file_must_not_be_reported_safe(surface):
    (surface / "templates" / "index.html").unlink()
    r = audit(surface)
    assert by_id(r)["remote_fonts"]["status"] == "unverified"
    assert r["launch_ready"] is False


def test_repository_current_surface_scans_no_external_fonts_or_known_replay():
    report = audit(Path(__file__).resolve().parents[1])
    checks = by_id(report)
    assert checks["remote_fonts"]["status"] == "pass"
    assert checks["session_replay"]["status"] == "pass"
    assert report["launch_ready"] is False
