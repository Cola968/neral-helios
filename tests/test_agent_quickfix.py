"""Safe deterministic repairs must be previewed, explicitly approved and reversible."""
import hashlib
import json

import pytest

from agent_quickfix import build_quick_fix, QuickFixError
from agent_mode import AgentEngine, AgentError


@pytest.mark.parametrize("kind,filename,before,expected", [
    ("json_pretty", "settings.json", '{"hi":true,"count":2}', '{\n  "hi": true,\n  "count": 2\n}\n'),
    ("python_whitespace", "main.py", "x = 1  \ny = x\t", "x = 1\ny = x\n"),
    ("python_colon", "main.py", "def get_number()\n    return 42\n", "def get_number():\n    return 42\n"),
])
def test_deterministic_quick_fixes(kind, filename, before, expected):
    assert build_quick_fix(filename, before, kind) == expected


def test_python_whitespace_does_not_modify_multiline_string_literals():
    original = "text = '''a  \nb'''\n"
    with pytest.raises(QuickFixError, match="Semantik"):
        build_quick_fix("code.py", original, "python_whitespace")


@pytest.mark.parametrize("text", [
    '{"a":1,"a":2}',
    '{"value":NaN}',
    '{"value":Infinity}',
    '{"truncated":',
])
def test_json_quick_fix_never_silently_changes_dangerous_or_invalid_data(text):
    with pytest.raises(QuickFixError):
        build_quick_fix("data.json", text, "json_pretty")


@pytest.mark.parametrize("filename,kind,text", [
    (".env", "python_whitespace", "a  \n"),
    ("code.js", "python_whitespace", "const x=2;  \n"),
    ("data.json", "python_colon", "a:\n"),
    ("run.py", "json_pretty", '{"a":1}'),
    ("code.py", "python_colon", "value = 2\n"),
    ("code.py", "python_colon", "def x():\n    return 42\n"),
    ("code.py", "python_colon", "def x():  # comment\n    return 42\n"),
])
def test_quick_fix_does_not_rewrite_unsupported_or_unnecessary_code(filename, kind, text):
    with pytest.raises(QuickFixError):
        build_quick_fix(filename, text, kind)


def test_reviewed_quick_fix_after_completed_audit_has_undo(tmp_path):
    target = tmp_path / "broken.py"
    original = "def answer()\n    return 42\n"
    target.write_text(original, encoding="utf-8")
    engine = AgentEngine(tmp_path)
    task = engine.create("Analysiere Fehler", strategy="quality")
    for _ in range(10):
        final = engine.next(task["id"])
        if final["status"] == "completed":
            break
    assert final["status"] == "completed"
    preview = engine.propose_fix(task["id"], path="broken.py", kind="python_colon")
    assert preview["status"] == "awaiting_approval"
    assert target.read_text() == original
    assert "def answer():" in preview["proposal"]["diff"]
    assert "token" in preview["proposal"]
    assert engine.approve(task["id"], preview["proposal"]["token"])["status"] == "completed"
    assert target.read_text() == "def answer():\n    return 42\n"
    assert engine.undo(task["id"])["status"] == "completed"
    assert target.read_text() == original


def test_quick_fix_rejects_concurrent_file_changes_and_keeps_preview(tmp_path):
    file = tmp_path / "data.json"
    file.write_text('{"v":1}', encoding="utf-8")
    engine = AgentEngine(tmp_path)
    task = engine.create("Prüfung")
    preview = engine.propose_fix(task["id"], path="data.json", kind="json_pretty")
    file.write_text('{"v":99}', encoding="utf-8")
    with pytest.raises(AgentError, match="verändert"):
        engine.approve(task["id"], preview["proposal"]["token"])
    assert json.loads(file.read_text()) == {"v":99}


def test_quick_fix_requires_permission_and_refuses_secret_paths(tmp_path):
    (tmp_path / ".env").write_text("SECRET=private\n")
    engine = AgentEngine(tmp_path)
    task = engine.create("Schütze Dateien")
    with pytest.raises(AgentError):
        engine.propose_fix(task["id"], path=".env", kind="json_pretty")
    with pytest.raises(AgentError):
        engine.propose_fix(task["id"], path="../secret.py", kind="python_colon")


def test_quick_fix_rejects_aborted_and_paused_jobs(tmp_path):
    (tmp_path / "data.json").write_text('{"v":1}')
    engine = AgentEngine(tmp_path)
    task = engine.create("Review")
    engine.pause(task["id"])
    with pytest.raises(AgentError):
        engine.propose_fix(task["id"], path="data.json", kind="json_pretty")
    engine.cancel(task["id"])
    with pytest.raises(AgentError):
        engine.propose_fix(task["id"], path="data.json", kind="json_pretty")
