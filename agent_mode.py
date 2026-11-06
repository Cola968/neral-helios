"""NERAL local-first agent: bounded, reviewable file tools, never a shell or cloud model.

The deterministic planner is deliberately not advertised as autonomous LLM coding.
All user content is untrusted; mutating workspace files requires a separate approval.
"""
from __future__ import annotations

import ast
import csv
import difflib
import hashlib
import io
import json
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from agent_quickfix import QuickFixError, build_quick_fix

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    tomllib = None


class AgentError(ValueError):
    """Safe error suitable for user display."""


class AgentEngine:
    MAX_TASKS = 16
    MAX_STEPS = 12
    MAX_FILES = 180
    MAX_READ = 128 * 1024
    MAX_EDIT = 32 * 1024
    MAX_RESULTS = 80
    MAX_SECONDS = 300
    ALLOWED_EDIT = {".py", ".js", ".css", ".html", ".json", ".toml", ".md", ".txt", ".yml", ".yaml"}
    IGNORED = {".git", ".svn", ".hg", ".ssh", ".venv", "venv", "node_modules",
               "__pycache__", ".pytest_cache", ".mypy_cache", ".idea", "local_state",
               "checkpoints", "export", "b2_sources", "sft50_sources", "evidence"}
    SECRET_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
                    "credentials.json", "secrets.json", "gateway-token", "private.key"}
    STRATEGIES = {
        "audit": ("inventory", "python", "markers", "summary"),
        "quality": ("python", "formats", "whitespace", "summary"),
        "search": ("search", "summary"),
        "inspect": ("inventory", "read", "summary"),
        "docs": ("markdown", "summary"),
        "security": ("security", "summary"),
    }

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise AgentError("Agenten-Arbeitsordner ist ungültig.")
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def safe_path(self, name: str) -> Path:
        if not isinstance(name, str) or not name or len(name) > 240 or "\x00" in name:
            raise AgentError("Ungültiger Dateipfad.")
        if "\\" in name or re.match(r"^[A-Za-z]:", name) or "%" in name:
            raise AgentError("Unzulässiger Dateipfad.")
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise AgentError("Pfad verlässt den Arbeitsordner.")
        parts = [part for part in p.parts if part != "."]
        if any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part) for part in parts):
            raise AgentError("Reservierter Windows-Dateiname.")
        if any(part.startswith(".") or part.lower() in self.IGNORED or
               part.lower() in self.SECRET_NAMES or "secret" in part.lower() or
               "token" in part.lower() or "password" in part.lower() for part in parts):
            raise AgentError("Geschützter Pfad.")
        current = self.root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise AgentError("Symbolische Links sind gesperrt.")
        resolved = current.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise AgentError("Pfad verlässt den Arbeitsordner.")
        return current

    def _files(self, suffixes=None):
        found = []
        for dirname, dirs, names in os.walk(self.root, followlinks=False):
            rel_dir = Path(dirname).relative_to(self.root)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and
                             d.lower() not in self.IGNORED and not (Path(dirname) / d).is_symlink())
            for name in sorted(names):
                if name.startswith("."):
                    continue
                relative = (rel_dir / name).as_posix()
                try:
                    p = self.safe_path(relative)
                except AgentError:
                    continue
                if not p.is_file() or (suffixes and p.suffix.lower() not in suffixes):
                    continue
                if p.stat().st_size > self.MAX_READ:
                    continue
                found.append((relative, p))
                if len(found) >= self.MAX_FILES:
                    return found
        return found

    def _read(self, file: Path) -> str:
        try:
            if file.stat().st_size > self.MAX_READ:
                raise AgentError("Datei überschreitet die Lesegrenze.")
            return file.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            raise AgentError("Datei kann nicht sicher als UTF-8 gelesen werden.")

    def _inventory(self, _: dict) -> dict:
        files = self._files()
        types: dict[str, int] = {}
        for name, _ in files:
            ext = Path(name).suffix.lower() or "(ohne Erweiterung)"
            types[ext] = types.get(ext, 0) + 1
        return {"files": len(files), "truncated": len(files) == self.MAX_FILES,
                "types": dict(sorted(types.items())),
                "sample": [name for name, _ in files[:30]]}

    def _python(self, _: dict) -> dict:
        files = self._files({".py"})
        errors, checked, functions, classes, imports = [], 0, 0, 0, 0
        for name, p in files:
            checked += 1
            try:
                tree = ast.parse(self._read(p), filename=name)
                functions += sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
                classes += sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
                imports += sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
            except (SyntaxError, AgentError) as err:
                errors.append({"file": name, "error": "SyntaxError" if isinstance(err, SyntaxError) else "Unreadable",
                               "line": getattr(err, "lineno", None)})
        return {"checked": checked, "syntax_errors": errors[:30], "functions": functions,
                "classes": classes, "imports": imports, "executed_code": False}

    def _formats(self, _: dict) -> dict:
        files = self._files({".json", ".toml", ".csv"})
        invalid = []
        for name, p in files:
            try:
                value = self._read(p)
                if p.suffix == ".json":
                    json.loads(value)
                elif p.suffix == ".toml":
                    if tomllib is None:
                        continue
                    tomllib.loads(value)
                elif p.suffix == ".csv":
                    rows = csv.reader(io.StringIO(value))
                    header = next(rows, [])
                    if not header or len(set(header)) != len(header):
                        raise ValueError("empty/duplicate header")
            except (ValueError, AgentError, csv.Error):
                invalid.append(name)
        return {"checked": len(files), "invalid": invalid[:30]}

    def _markers(self, _: dict) -> dict:
        found = []
        for name, p in self._files({".py", ".js", ".ts", ".md", ".txt", ".html"}):
            try:
                for i, line in enumerate(self._read(p).splitlines(), 1):
                    if re.search(r"\b(TODO|FIXME|XXX)\b", line, re.I):
                        found.append({"file": name, "line": i, "tag": re.search(r"\b(TODO|FIXME|XXX)\b", line, re.I).group().upper()})
                        if len(found) >= 50:
                            return {"markers": found, "truncated": True}
            except AgentError:
                continue
        return {"markers": found, "truncated": False}

    def _whitespace(self, _: dict) -> dict:
        issues = []
        for name, p in self._files({".py", ".js", ".md", ".json", ".css", ".html"}):
            try:
                content = self._read(p)
            except AgentError:
                continue
            if content and not content.endswith("\n"):
                issues.append({"file": name, "type": "missing_final_newline"})
            for i, line in enumerate(content.splitlines(), 1):
                if line.rstrip() != line:
                    issues.append({"file": name, "line": i, "type": "trailing_whitespace"})
                if len(line) > 160:
                    issues.append({"file": name, "line": i, "type": "long_line"})
                if len(issues) >= 50:
                    return {"issues": issues, "truncated": True}
        return {"issues": issues, "truncated": False}

    def _markdown(self, _: dict) -> dict:
        missing = []
        for name, p in self._files({".md"}):
            try:
                content = self._read(p)
            except AgentError:
                continue
            for dest in re.findall(r"\[[^]]+\]\(([^)]+)\)", content):
                link = dest.split("#", 1)[0].strip()
                if not link or link.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                if " " in link:
                    link = link.split(" ", 1)[0]
                rel = (Path(name).parent / link).as_posix()
                try:
                    if not self.safe_path(rel).exists():
                        missing.append({"file": name, "target": link})
                except AgentError:
                    missing.append({"file": name, "target": "(blocked link)"})
                if len(missing) >= 40:
                    return {"broken_links": missing, "truncated": True}
        return {"broken_links": missing, "truncated": False}

    def _security(self, _: dict) -> dict:
        issues = []
        for name, p in self._files({".py", ".js", ".json", ".yml", ".yaml"}):
            try:
                data = self._read(p)
            except AgentError:
                continue
            for lineno, line in enumerate(data.splitlines(), 1):
                if re.search(r"(?:subprocess\..*shell\s*=\s*True|\beval\s*\(|\bexec\s*\()", line):
                    issues.append({"file": name, "line": lineno, "risk": "dynamic_execution_review"})
                if len(issues) >= 30:
                    return {"findings": issues, "truncated": True, "proof_of_exploit": False}
        return {"findings": issues, "truncated": False, "proof_of_exploit": False}

    def _search(self, task: dict) -> dict:
        query = task["query"] or task["goal"]
        if len(query) > 120:
            raise AgentError("Suchtext zu lang.")
        needle = query.casefold()
        matches = []
        for name, p in self._files({".py", ".md", ".txt", ".json", ".js", ".css", ".html", ".toml"}):
            try:
                for i, line in enumerate(self._read(p).splitlines(), 1):
                    if needle in line.casefold():
                        matches.append({"file": name, "line": i, "excerpt": line[:180]})
                        if len(matches) >= 40:
                            return {"matches": matches, "truncated": True}
            except AgentError:
                continue
        return {"matches": matches, "truncated": False}

    def _inspect(self, task: dict) -> dict:
        file = self.safe_path(task["path"])
        if file == self.root or not file.is_file():
            return {"note": "Kein lesbarer Dateipfad gewählt; wähle eine Datei."}
        value = self._read(file)
        return {"file": task["path"], "sha256": self._digest(value.encode()),
                "line_count": len(value.splitlines()), "preview": "\n".join(value.splitlines()[:80])[:8000]}

    def _summary(self, task: dict) -> dict:
        checks = task["results"]
        return {"completed_tools": list(checks), "note": "Statische Prüfung, keine Codeausführung und kein autonomer Modell-Patch."}

    TOOLS = {"inventory": _inventory, "python": _python, "formats": _formats,
             "markers": _markers, "whitespace": _whitespace, "markdown": _markdown,
             "security": _security, "search": _search, "read": _inspect, "summary": _summary}

    def create(self, goal: str, strategy: str = "audit", path: str = ".", query: str = "") -> dict:
        if not isinstance(goal, str) or not 1 <= len(goal.strip()) <= 800:
            raise AgentError("Ziel muss 1 bis 800 Zeichen enthalten.")
        if not isinstance(strategy, str) or strategy not in self.STRATEGIES:
            raise AgentError("Unbekannte Agentenstrategie.")
        if not isinstance(path, str) or len(path) > 240:
            raise AgentError("Ungültiger Dateipfad.")
        if not isinstance(query, str) or len(query) > 120:
            raise AgentError("Suchtext zu lang.")
        if strategy == "search" and not query and len(goal) > 120:
            raise AgentError("Suchtext zu lang.")
        if strategy == "inspect" and path != ".":
            self.safe_path(path)
        with self._lock:
            if len(self._tasks) >= self.MAX_TASKS:
                for key, val in list(self._tasks.items()):
                    if val["status"] in {"completed", "cancelled", "failed"}:
                        del self._tasks[key]
                        break
            if len(self._tasks) >= self.MAX_TASKS:
                raise AgentError("Zu viele aktive Agentenaufträge.")
            ident = secrets.token_hex(12)
            now = time.monotonic()
            self._tasks[ident] = dict(id=ident, goal=goal.strip(), strategy=strategy,
                                     path=path, query=query.strip(), plan=list(self.STRATEGIES[strategy]),
                                     index=0, status="running", created=now, deadline=now + self.MAX_SECONDS,
                                     results={}, events=[], proposal=None, undo=None, error=None)
            return self.snapshot(ident)

    def _task(self, ident: str) -> dict:
        if not isinstance(ident, str) or not re.fullmatch(r"[0-9a-f]{24}", ident):
            raise AgentError("Ungültige Agenten-ID.")
        if ident not in self._tasks:
            raise AgentError("Agentenauftrag nicht gefunden.")
        return self._tasks[ident]

    def snapshot(self, ident: str) -> dict:
        with self._lock:
            task = self._task(ident)
            out = {k: task[k] for k in ("id", "goal", "strategy", "status", "error")}
            out["plan"] = list(task["plan"])
            out["results"] = dict(task["results"])
            out["events"] = list(task["events"])
            out["steps_completed"] = task["index"]
            out["progress"] = round(task["index"] / max(1, len(task["plan"])), 3)
            out["proposal"] = dict(task["proposal"]) if task["proposal"] else None
            out["can_undo"] = task["undo"] is not None
            return out

    def _log(self, task: dict, tool: str, status: str) -> None:
        task["events"].append({"step": len(task["events"]) + 1, "tool": tool, "status": status})
        task["events"] = task["events"][-self.MAX_RESULTS:]

    def next(self, ident: str) -> dict:
        with self._lock:
            task = self._task(ident)
            if task["status"] != "running":
                return self.snapshot(ident)
            if time.monotonic() >= task["deadline"]:
                task["status"] = "failed"
                task["error"] = "Zeitbudget überschritten."
                self._log(task, "deadline", "failed")
                return self.snapshot(ident)
            if task["index"] >= len(task["plan"]) or task["index"] >= self.MAX_STEPS:
                task["status"] = "completed"
                return self.snapshot(ident)
            tool = task["plan"][task["index"]]
            try:
                # A tool is called by fixed dispatch, never by eval, exec or a shell.
                result = self.TOOLS[tool](self, task)
                task["results"][tool] = result
                task["index"] += 1
                self._log(task, tool, "ok")
                if task["index"] >= len(task["plan"]):
                    task["status"] = "completed"
            except (AgentError, OSError, ValueError):
                task["status"] = "failed"
                task["error"] = "Der Schritt konnte sicher nicht abgeschlossen werden."
                self._log(task, tool, "failed")
            return self.snapshot(ident)

    def cancel(self, ident: str) -> dict:
        with self._lock:
            task = self._task(ident)
            if task["status"] in ("running", "paused", "awaiting_approval"):
                task["status"] = "cancelled"
                task["proposal"] = None
                task.pop("_pending", None)
                task.pop("_after_edit_status", None)
                self._log(task, "cancel", "ok")
            return self.snapshot(ident)

    def pause(self, ident: str) -> dict:
        with self._lock:
            task = self._task(ident)
            if task["status"] == "running":
                task["status"] = "paused"
                self._log(task, "pause", "ok")
            return self.snapshot(ident)

    def resume(self, ident: str) -> dict:
        with self._lock:
            task = self._task(ident)
            if task["status"] != "paused":
                raise AgentError("Nur pausierte Aufträge können fortgesetzt werden.")
            if time.monotonic() >= task["deadline"]:
                raise AgentError("Zeitbudget abgelaufen.")
            task["status"] = "running"
            self._log(task, "resume", "ok")
            return self.snapshot(ident)

    def propose(self, ident: str, *, path: str, content: str, expected_sha: str | None) -> dict:
        with self._lock:
            task = self._task(ident)
            if task["status"] != "running":
                raise AgentError("Auftrag ist nicht aktiv.")
            if time.monotonic() >= task["deadline"]:
                raise AgentError("Zeitbudget abgelaufen.")
            file = self.safe_path(path)
            if file.suffix.lower() not in self.ALLOWED_EDIT:
                raise AgentError("Dateityp ist für Änderungen nicht freigegeben.")
            if not isinstance(content, str):
                raise AgentError("Dateiinhalt muss Text sein.")
            try:
                raw = content.encode("utf-8")
            except UnicodeError:
                raise AgentError("Ungültiger Unicode-Text.")
            if len(raw) > self.MAX_EDIT:
                raise AgentError("Dateiinhalt ist zu groß.")
            if not file.parent.is_dir():
                raise AgentError("Zielverzeichnis existiert nicht.")
            if file.exists() and not file.is_file():
                raise AgentError("Ziel ist keine reguläre Datei.")
            existing = file.read_bytes() if file.exists() else None
            if existing is not None and len(existing) > self.MAX_EDIT:
                raise AgentError("Datei ist zu groß für sichere Änderungen.")
            if existing is not None:
                if not isinstance(expected_sha, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha) or self._digest(existing) != expected_sha:
                    raise AgentError("Dateiversion passt nicht mehr: SHA-256-Konflikt.")
                try:
                    old_text = existing.decode("utf-8")
                except UnicodeError:
                    raise AgentError("Bestehende Datei ist kein UTF-8-Text.")
            else:
                if expected_sha is not None:
                    raise AgentError("Datei existiert nicht mehr.")
                old_text = ""
            if existing == raw:
                raise AgentError("Datei ist unverändert.")
            diff_lines = list(difflib.unified_diff(old_text.splitlines(keepends=True),
                            content.splitlines(keepends=True), fromfile=path + ":alt", tofile=path + ":neu"))
            diff = "".join(diff_lines)
            if len(diff_lines) > 400 or len(diff) > 12000:
                raise AgentError("Diff ist für eine vollständige Vorschau zu groß. Teile die Änderung auf.")
            token = secrets.token_hex(16)
            task["proposal"] = {"path": path, "expected_sha": expected_sha, "new_sha": self._digest(raw),
                                "token": token, "diff": diff, "is_new": existing is None}
            task["_pending"] = raw
            task["status"] = "awaiting_approval"
            self._log(task, "preview", "approval_required")
            return self.snapshot(ident)

    def propose_fix(self, ident: str, *, path: str, kind: str) -> dict:
        """Compute a deterministic fix, then use the normal SHA approval gate.

        Supports completed audits so the user can inspect results first.
        No file mutation occurs until a separate approve(token) call.
        """
        with self._lock:
            task = self._task(ident)
            if task["status"] not in {"running", "completed"}:
                raise AgentError("Quick-Fix nur für aktive oder abgeschlossene Prüfungen.")
            if time.monotonic() >= task["deadline"]:
                raise AgentError("Zeitbudget abgelaufen.")
            file = self.safe_path(path)
            if not file.is_file():
                raise AgentError("Quick-Fix benötigt eine vorhandene Textdatei.")
            size = file.stat().st_size
            if size > self.MAX_EDIT:
                raise AgentError("Datei überschreitet Bearbeitungsgrenze.")
            raw = file.read_bytes()
            try:
                original = raw.decode("utf-8")
                updated = build_quick_fix(path, original, kind)
            except (UnicodeError, QuickFixError) as exc:
                raise AgentError(str(exc) if isinstance(exc, QuickFixError)
                                 else "Datei ist kein UTF-8-Text.") from exc
            completed = task["status"] == "completed"
            if completed:
                task["status"] = "running"
            try:
                result = self.propose(ident, path=path, content=updated, expected_sha=self._digest(raw))
                task["_after_edit_status"] = "completed" if completed else "running"
                return result
            except Exception:
                if completed:
                    task["status"] = "completed"
                raise

    @staticmethod
    def _atomic_write(file: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(prefix=".neral-agent-", dir=file.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def approve(self, ident: str, token: str) -> dict:
        with self._lock:
            task = self._task(ident)
            p = task["proposal"]
            if task["status"] != "awaiting_approval" or not p or not isinstance(token, str) or not secrets.compare_digest(p["token"], token):
                raise AgentError("Freigabetoken stimmt nicht.")
            if time.monotonic() >= task["deadline"]:
                raise AgentError("Zeitbudget abgelaufen.")
            file = self.safe_path(p["path"])
            existing = file.read_bytes() if file.exists() and file.is_file() else None
            expected = p["expected_sha"]
            if (existing is None) != (expected is None) or (existing is not None and self._digest(existing) != expected):
                raise AgentError("Datei hat sich seit der Vorschau verändert.")
            new_data = task["_pending"]
            try:
                self._atomic_write(file, new_data)
            except OSError:
                raise AgentError("Schreiben fehlgeschlagen; Vorschau bleibt offen.")
            task.pop("_pending", None)
            task["undo"] = {"path": p["path"], "old": existing, "new_sha": p["new_sha"]}
            task["proposal"] = None
            task["status"] = task.pop("_after_edit_status", "running")
            self._log(task, "apply", "ok")
            return self.snapshot(ident)

    def undo(self, ident: str) -> dict:
        with self._lock:
            task = self._task(ident)
            undo = task["undo"]
            if undo is None:
                raise AgentError("Keine reversible Änderung vorhanden.")
            file = self.safe_path(undo["path"])
            if not file.is_file() or self._digest(file.read_bytes()) != undo["new_sha"]:
                raise AgentError("Rückgängig nicht möglich: Datei wurde extern geändert.")
            if undo["old"] is None:
                file.unlink()
            else:
                self._atomic_write(file, undo["old"])
            task["undo"] = None
            self._log(task, "undo", "ok")
            return self.snapshot(ident)

    def capabilities(self) -> dict:
        return {"strategies": list(self.STRATEGIES), "tools": list(self.TOOLS),
                "max_steps": self.MAX_STEPS, "max_file_bytes": self.MAX_READ,
                "max_edit_bytes": self.MAX_EDIT, "external_model": False,
                "remote_execution": False, "shell_execution": False,
                "autonomous_code_generation": False, "approval_required_for_writes": True,
                "reviewed_quick_fixes": ["json_pretty", "python_whitespace", "python_colon"]}
