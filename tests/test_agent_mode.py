"""Behavioral contract for NERAL's local, bounded agent mode."""
import hashlib
import json
from pathlib import Path

import pytest

from agent_mode import AgentEngine, AgentError


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "good.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def nope(:\n pass\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"ok":true}', encoding="utf-8")
    (tmp_path / "note.md").write_text("# Note\nSee [local](good.py)\n", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=sensitive", encoding="utf-8")
    return tmp_path


@pytest.fixture
def engine(workspace):
    return AgentEngine(workspace)


def drain(engine, task_id):
    for _ in range(30):
        result = engine.next(task_id)
        if result["status"] in {"completed", "cancelled", "failed", "awaiting_approval"}:
            return result
    raise AssertionError("Agent did not terminate")


@pytest.mark.parametrize("strategy", ["audit", "quality", "search", "inspect", "docs", "security"])
def test_strategies_finish_in_bounded_steps(engine, strategy):
    run = engine.create("Untersuche das Projekt", strategy=strategy, query="answer")
    result = drain(engine, run["id"])
    assert result["status"] == "completed"
    assert result["progress"] == 1
    assert result["steps_completed"] <= 12
    assert result["events"]


def test_python_syntax_error_is_reported(engine):
    task = engine.create("Python überprüfen", strategy="quality")
    finished = drain(engine, task["id"])
    evidence = str(finished["results"])
    assert "broken.py" in evidence
    assert "SyntaxError" in evidence


def test_search_is_limited_to_workspace_and_hides_secrets(engine):
    task = engine.create("suche", strategy="search", query="sensitive")
    result = drain(engine, task["id"])
    assert "sensitive" not in str(result["results"])
    assert ".env" not in str(result["results"])


@pytest.mark.parametrize("path", [
    "../escape.py", "/etc/passwd", r"C:\Windows\win.ini", ".env", ".git/config",
    "./.ssh/id_rsa", "node_modules/package.json", "a/../../b.py", "foo\\bar.txt",
])
def test_disallowed_path_never_read(engine, path):
    with pytest.raises(AgentError):
        engine.safe_path(path)


def test_symlink_escape_rejected(engine, workspace, tmp_path):
    outside = workspace.parent / (workspace.name + "_private")
    outside.mkdir(exist_ok=True)
    (outside / "leak.py").write_text("do not read")
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AgentError):
        engine.safe_path("link/leak.py")


def test_invalid_request_rejected(engine):
    for kwargs in [{"goal": ""}, {"goal": "x" * 2000},
                   {"goal": "ok", "strategy": "run_shell"},
                   {"goal": "ok", "query": "a" * 1000}]:
        with pytest.raises(AgentError):
            engine.create(**kwargs)


def test_cancel_stops_future_actions(engine):
    task = engine.create("audit", strategy="audit")
    assert engine.cancel(task["id"])["status"] == "cancelled"
    assert engine.next(task["id"])["status"] == "cancelled"


def test_pause_then_resume(engine):
    task = engine.create("audit")
    assert engine.pause(task["id"])["status"] == "paused"
    assert engine.next(task["id"])["status"] == "paused"
    assert engine.resume(task["id"])["status"] == "running"
    assert drain(engine, task["id"])["status"] == "completed"


def test_edit_requires_approval_and_preview_cas(engine, workspace):
    original = (workspace / "good.py").read_text()
    sha = hashlib.sha256(original.encode()).hexdigest()
    task = engine.create("update", strategy="inspect", path="good.py")
    proposal = engine.propose(task["id"], path="good.py", content="print('updated')\n", expected_sha=sha)
    assert (workspace / "good.py").read_text() == original
    assert proposal["status"] == "awaiting_approval"
    assert "---" in proposal["proposal"]["diff"]
    with pytest.raises(AgentError):
        engine.approve(task["id"], "wrong")
    assert (workspace / "good.py").read_text() == original
    done = engine.approve(task["id"], proposal["proposal"]["token"])
    assert done["status"] == "running"
    assert (workspace / "good.py").read_text() == "print('updated')\n"
    assert engine.undo(task["id"])["status"] == "running"
    assert (workspace / "good.py").read_text() == original


def test_concurrent_change_fails_closed(engine, workspace):
    original = (workspace / "good.py").read_bytes()
    task = engine.create("update")
    proposal = engine.propose(task["id"], path="good.py", content="print(1)\n",
                              expected_sha=hashlib.sha256(original).hexdigest())
    (workspace / "good.py").write_text("changed by another editor", encoding="utf-8")
    with pytest.raises(AgentError):
        engine.approve(task["id"], proposal["proposal"]["token"])
    assert (workspace / "good.py").read_text() == "changed by another editor"


def test_undo_refuses_to_overwrite_new_user_work(engine, workspace):
    original = (workspace / "good.py").read_bytes()
    task = engine.create("update")
    p = engine.propose(task["id"], path="good.py", content="print(1)\n",
                       expected_sha=hashlib.sha256(original).hexdigest())
    engine.approve(task["id"], p["proposal"]["token"])
    (workspace / "good.py").write_text("someone else's work", encoding="utf-8")
    with pytest.raises(AgentError):
        engine.undo(task["id"])


def test_new_file_requires_approval_and_undo(engine, workspace):
    task = engine.create("add")
    p = engine.propose(task["id"], path="new.py", content="print(1)\n", expected_sha=None)
    assert not (workspace / "new.py").exists()
    engine.approve(task["id"], p["proposal"]["token"])
    assert (workspace / "new.py").exists()
    engine.undo(task["id"])
    assert not (workspace / "new.py").exists()


def test_large_edit_and_secret_path_rejected(engine):
    task = engine.create("update")
    for path, content in [("normal.py", "x" * 50000), (".env", "x"), ("x.exe", "binary")]:
        with pytest.raises(AgentError):
            engine.propose(task["id"], path=path, content=content, expected_sha=None)


def test_markdown_link_check_and_json(engine, workspace):
    (workspace / "note.md").write_text("[bad](missing.md)\n", encoding="utf-8")
    (workspace / "config.json").write_text("{bad", encoding="utf-8")
    docs = drain(engine, engine.create("docs", strategy="docs")["id"])
    quality = drain(engine, engine.create("lint", strategy="quality")["id"])
    assert "missing.md" in str(docs["results"])
    assert "config.json" in str(quality["results"])


def test_run_log_is_json_serializable_and_bounded(engine):
    task = engine.create("scan", strategy="audit")
    result = drain(engine, task["id"])
    assert len(json.dumps(result)) < 120_000
    assert all("tool" in event and "status" in event for event in result["events"])


def test_limit_overlapping_jobs(engine):
    for i in range(engine.MAX_TASKS):
        engine.create(f"Task {i}")
    with pytest.raises(AgentError):
        engine.create("overflow")


def test_expired_task_cannot_propose(engine):
    task = engine.create("edit")
    engine._tasks[task["id"]]["deadline"] = 0
    with pytest.raises(AgentError, match="Zeitbudget"):
        engine.propose(task["id"], path="new.py", content="x=1", expected_sha=None)


def test_expired_preview_cannot_be_approved(engine, workspace):
    task = engine.create("edit")
    proposal = engine.propose(task["id"], path="new.py", content="x=1", expected_sha=None)
    engine._tasks[task["id"]]["deadline"] = 0
    with pytest.raises(AgentError, match="Zeitbudget"):
        engine.approve(task["id"], proposal["proposal"]["token"])
    assert not (workspace / "new.py").exists()


def test_noop_edit_cannot_enter_approval(engine, workspace):
    data = (workspace / "good.py").read_bytes()
    task = engine.create("edit")
    with pytest.raises(AgentError, match="unverändert"):
        engine.propose(task["id"], path="good.py", content=data.decode(),
                       expected_sha=hashlib.sha256(data).hexdigest())


def test_cancel_disposes_pending_edits(engine):
    task = engine.create("edit")
    engine.propose(task["id"], path="new.py", content="test", expected_sha=None)
    engine.cancel(task["id"])
    assert "_pending" not in engine._tasks[task["id"]]


def test_failed_atomic_replace_preserves_approval_for_safe_retry(engine, workspace, monkeypatch):
    task = engine.create("edit")
    proposal = engine.propose(task["id"], path="new.py", content="test", expected_sha=None)
    real = engine._atomic_write

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(engine, "_atomic_write", fail)
    with pytest.raises(AgentError, match="Schreiben"):
        engine.approve(task["id"], proposal["proposal"]["token"])
    assert engine.snapshot(task["id"])["status"] == "awaiting_approval"
    monkeypatch.setattr(engine, "_atomic_write", real)
    engine.approve(task["id"], proposal["proposal"]["token"])
    assert (workspace / "new.py").read_text() == "test"


@pytest.mark.parametrize("path", ["CON", "nul.txt", "COM1.py", "lpt9.md"])
def test_windows_reserved_names_rejected(engine, path):
    with pytest.raises(AgentError):
        engine.safe_path(path)


def test_search_goal_too_long_without_query_rejected(engine):
    with pytest.raises(AgentError, match="Suchtext"):
        engine.create("x" * 121, strategy="search")


def test_json_toml_csv_quality_audit(engine, workspace):
    (workspace / "bad.toml").write_text("a = ", encoding="utf-8")
    (workspace / "bad.csv").write_text("a,a\n1,2\n", encoding="utf-8")
    result = drain(engine, engine.create("audit", strategy="quality")["id"])
    assert "bad.toml" in str(result["results"])
    assert "bad.csv" in str(result["results"])


def test_summary_never_claims_autonomous_model_edits(engine):
    result = drain(engine, engine.create("review")["id"])
    assert "kein autonomer Modell-Patch" in result["results"]["summary"]["note"]


def test_empty_or_malformed_agent_id_rejected(engine):
    for ident in ["", "abc", None, "x" * 100]:
        with pytest.raises(AgentError):
            engine.snapshot(ident)


def test_python_analysis_never_runs_code(engine, workspace):
    (workspace / "danger.py").write_text("raise RuntimeError('executed')\n", encoding="utf-8")
    result = drain(engine, engine.create("audit", strategy="quality")["id"])
    assert result["status"] == "completed"
    assert result["results"]["python"]["executed_code"] is False


def test_search_is_case_insensitive_with_line_numbers(engine):
    result = drain(engine, engine.create("find", strategy="search", query="ANSWER")["id"])
    matches = result["results"]["search"]["matches"]
    assert any(m["file"] == "good.py" and m["line"] == 1 for m in matches)


def test_quality_detects_missing_final_newline_and_long_lines(engine, workspace):
    (workspace / "style.py").write_text("x=" + "1"*170, encoding="utf-8")
    result = drain(engine, engine.create("lint", strategy="quality")["id"])
    issues = result["results"]["whitespace"]["issues"]
    assert any(x["type"] == "missing_final_newline" for x in issues)
    assert any(x["type"] == "long_line" for x in issues)


def test_pending_approval_blocks_next_steps(engine):
    task = engine.create("edit")
    proposal = engine.propose(task["id"], path="new.py", content="hello", expected_sha=None)
    result = engine.next(task["id"])
    assert result["status"] == "awaiting_approval"
    assert result["steps_completed"] == 0


def test_completed_slots_can_be_recycled(engine):
    ids = [engine.create(f"Task {i}")["id"] for i in range(engine.MAX_TASKS)]
    drain(engine, ids[0])
    assert engine.create("after completion")["status"] == "running"
