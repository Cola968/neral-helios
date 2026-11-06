"""NERAL local model-directed read/inspect/plan/patch agent.

An LLM suggests one JSON tool call at a time. Every call is validated and
executed by the existing local AgentEngine. Generated edits are NEVER
automatically written: an explicit signed-by-session one-time approval is
required. No shell, remote provider or arbitrary network access.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from agent_mode import AgentEngine, AgentError


class AgentModelError(RuntimeError):
    """Untrusted inference failed; raw details must not reach HTTP clients."""


class OllamaLocalBackend:
    """Opt-in localhost Ollama, never a remote provider or outbound proxy."""

    ENDPOINT = "http://127.0.0.1:11434/api/chat"

    def __init__(self, model: str, endpoint: str = ENDPOINT, timeout: float = 55.0):
        if endpoint != self.ENDPOINT:
            raise ValueError("Nur die lokale Ollama-Schnittstelle ist erlaubt.")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}", model):
            raise ValueError("Ungültige lokale Modellkennung.")
        if not 1 <= timeout <= 90:
            raise ValueError("Ungültiges Modell-Zeitlimit.")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def respond(self, messages: list[dict]) -> str:
        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 512},
            "keep_alive": "5m",
        }
        payload = json.dumps(request_data, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
        # The proxy handler is explicitly disabled. No cloud inference fallback.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                raw = response.read(128 * 1024 + 1)
            if len(raw) > 128 * 1024:
                raise AgentModelError("Modellantwort zu groß.")
            packet = json.loads(raw)
            answer = packet["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise AgentModelError("Leere Modellantwort.")
            return answer
        except (OSError, KeyError, ValueError, TypeError, urllib.error.URLError) as exc:
            raise AgentModelError("Lokales Modell nicht erreichbar oder Antwort ungültig.") from exc


SYSTEM = """Du bist der lokale NERAL-Werkzeug-Agent. Du arbeitest nur im bereitgestellten Projekt.
Deine Aufgabe: analysiere das Projekt, finde relevante Dateien, lies sie, prüfe Fehler,
schlage bei Bedarf konkrete Codeaenderungen vor. Du darfst nicht vortäuschen,
dass du Code ausgeführt hast, oder dass Checks bestanden sind.

Antworte immer mit GENAU EINEM gültigen JSON-Objekt (kein Markdown/Prosa).
Feld action ist eine dieser Optionen:
  inventory                 -> Projektdateien auflisten
  read (path)               -> Textdatei lesen (relativer Pfad)
  search (query)            -> Text in Projektdateien suchen
  python                    -> statischer Python-AST-Syntaxcheck
  formats                   -> JSON/TOML/CSV prüfen
  security                  -> verdächtige Muster prüfen
  markdown                  -> lokale Dokumentationslinks prüfen
  markers                   -> TODO/FIXME sammeln
  whitespace                -> Formatierungsmängel finden
  propose_edit (path,content)-> kompletten Dateiinhalt als Änderung vorschlagen;
                                erfordert BENUTZERFREIGABE vor dem Schreiben
  finish (summary)          -> Aufgabe abschließen, ehrlich Ergebnisse zusammenfassen
Optionale Angabe reason beschreibt kurz, warum das Werkzeug gewählt wurde.
Schreibe nur benötigte Felder, z.B. {"action":"read","path":"main.py"}.
Keine Shell, kein Subprocess, kein Browser, keine Downloads, keine API-Schlüssel,
kein Lesen von .env-Dateien und keine Änderungen ohne Benutzerfreigabe.
Lies Dateien vor einem Patch. Du hast ein knappes Schrittbudget.
Wenn du nicht weiterkommst, verwende finish mit einer ehrlichen Erklärung.
"""


def _parse_choice(raw: str) -> dict:
    if not isinstance(raw, str) or len(raw) > 40_000:
        raise AgentModelError("Modell lieferte keine gültige Tool-Anweisung.")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.S | re.I)
        if not match:
            raise AgentModelError("Ungültige Tool-Anweisung.")
        cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError) as exc:
        raise AgentModelError("Modell lieferte kein JSON.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("action"), str):
        raise AgentModelError("Tool-Anweisung ohne gültige Aktion.")
    return data


class AutonomousAgent:
    MAX_JOBS = 6
    TOOL_NAMES = set(AgentEngine.TOOLS) - {"summary"} | {"propose_edit", "finish"}
    READ_EXTENSIONS = {".py", ".js", ".ts", ".md", ".json", ".toml", ".txt",
                       ".yml", ".yaml", ".html", ".css"}

    def __init__(self, root: Path | str, backend, *, max_steps: int = 8, max_seconds: int = 300):
        if not 1 <= max_steps <= 12 or not 30 <= max_seconds <= 900:
            raise ValueError("Ungültige Laufzeitbegrenzung.")
        self.engine = AgentEngine(root)
        self.backend = backend
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()

    def _job(self, id: str) -> dict:
        if not isinstance(id, str) or not re.fullmatch(r"[a-f0-9]{24}", id) or id not in self.jobs:
            raise AgentError("Unbekannter KI-Agentenauftrag.")
        return self.jobs[id]

    def start(self, goal: str) -> dict:
        if not isinstance(goal, str) or not 1 <= len(goal.strip()) <= 800:
            raise AgentError("Ziel muss 1 bis 800 Zeichen lang sein.")
        with self.lock:
            for key, job in list(self.jobs.items()):
                if job["status"] in {"completed", "failed", "limit_reached", "cancelled"}:
                    del self.jobs[key]
                    break
            if len(self.jobs) >= self.MAX_JOBS:
                raise AgentError("Zu viele laufende KI-Aufträge.")
            ident = secrets.token_hex(12)
            self.jobs[ident] = {
                "id": ident, "goal": goal.strip(), "status": "running", "steps": 0,
                "deadline": time.monotonic() + self.max_seconds, "history": [],
                "answer": None, "last_action": None, "duplicates": 0,
                "pending_task": None, "proposal": None, "model_origin": "local_model_only",
                "reads": {},
            }
            return self.snapshot(ident)

    def snapshot(self, id: str) -> dict:
        with self.lock:
            job = self._job(id)
            return {key: job[key] for key in ("id", "goal", "status", "steps", "answer", "model_origin")} | {
                "max_steps": self.max_steps, "history": list(job["history"][-12:]),
                "proposal": dict(job["proposal"]) if job["proposal"] else None,
            }

    def pause(self, id: str) -> dict:
        with self.lock:
            job = self._job(id)
            if job["status"] == "running":
                job["status"] = "paused"
            return self.snapshot(id)

    def resume(self, id: str) -> dict:
        with self.lock:
            job = self._job(id)
            if job["status"] != "paused":
                raise AgentError("Der Auftrag ist nicht pausiert.")
            if time.monotonic() > job["deadline"]:
                job["status"] = "failed"
                raise AgentError("Auftrag ist abgelaufen.")
            job["status"] = "running"
            return self.snapshot(id)

    def cancel(self, id: str) -> dict:
        with self.lock:
            job = self._job(id)
            if job["status"] in ("running", "paused", "awaiting_approval"):
                if job["pending_task"]:
                    self.engine.cancel(job["pending_task"])
                job["pending_task"] = None
                job["proposal"] = None
                job["status"] = "cancelled"
            return self.snapshot(id)

    def reject(self, id: str) -> dict:
        return self.cancel(id)

    def approve(self, id: str, token: str) -> dict:
        with self.lock:
            job = self._job(id)
            if job["status"] != "awaiting_approval" or not job["pending_task"]:
                raise AgentError("Kein freizugebender Patch.")
            if time.monotonic() >= job["deadline"]:
                raise AgentError("Auftrag ist abgelaufen.")
            self.engine.approve(job["pending_task"], token)
            job["history"].append({"action": "propose_edit", "status": "applied_after_approval"})
            job["pending_task"] = None
            job["proposal"] = None
            job["status"] = "running"
            return self.snapshot(id)

    def _history_messages(self, job: dict) -> list[dict]:
        history = json.dumps(job["history"][-7:], ensure_ascii=False)[:8500]
        remaining = self.max_steps - job["steps"]
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Aufgabe: {job['goal']}\nNoch verfügbare Entscheidungen: {remaining}\n"
                                      f"Bisherige Werkzeugergebnisse: {history}\nWähle die nächste Aktion."},
        ]

    def step(self, id: str) -> dict:
        with self.lock:
            job = self._job(id)
            if job["status"] != "running":
                return self.snapshot(id)
            if time.monotonic() >= job["deadline"]:
                job["status"] = "failed"
                job["history"].append({"action": "deadline", "status": "failed"})
                return self.snapshot(id)
            if job["steps"] >= self.max_steps:
                job["status"] = "limit_reached"
                return self.snapshot(id)
            job["steps"] += 1
            try:
                command = _parse_choice(self.backend.respond(self._history_messages(job)))
                action = command["action"]
                if action not in self.TOOL_NAMES:
                    raise AgentModelError("Unbekanntes Werkzeug.")
                signature = json.dumps({k: command.get(k) for k in ("action", "path", "query")}, sort_keys=True)
                if signature == job["last_action"]:
                    job["duplicates"] += 1
                else:
                    job["duplicates"] = 0
                job["last_action"] = signature
                if job["duplicates"] >= 2:
                    raise AgentModelError("Agent wiederholt denselben Schritt.")
                if action == "finish":
                    summary = command.get("summary")
                    if not isinstance(summary, str) or not 1 <= len(summary) <= 4000:
                        raise AgentModelError("Ungültiger Abschluss.")
                    job["answer"] = summary
                    job["status"] = "completed"
                    job["history"].append({"action": "finish", "status": "completed"})
                elif action == "propose_edit":
                    path, content = command.get("path"), command.get("content")
                    if not isinstance(path, str) or not isinstance(content, str):
                        raise AgentModelError("Patch erfordert Pfad und Inhalt.")
                    # Use the existing audited CAS/approval engine. Never auto-approve.
                    target = self.engine.safe_path(path)
                    existing = target.read_bytes() if target.is_file() else None
                    expected = None
                    if existing is not None:
                        expected = job["reads"].get(path)
                        if expected is None:
                            raise AgentError("Vor Änderungen muss die Datei gelesen werden.")
                        if len(existing) > 5000 or self.engine._digest(existing) != expected:
                            raise AgentError("Datei zu groß oder seit dem Lesen verändert.")
                    mutation = self.engine.create(f"Model proposal: {job['goal']}")
                    try:
                        pending = self.engine.propose(mutation["id"], path=path, content=content, expected_sha=expected)
                    except Exception:
                        self.engine.cancel(mutation["id"])
                        raise
                    job["pending_task"] = mutation["id"]
                    job["proposal"] = pending["proposal"]
                    job["status"] = "awaiting_approval"
                    job["history"].append({"action": action, "path": path, "status": "approval_required"})
                else:
                    result = self._execute(action, command, job)
                    job["history"].append({"action": action, "status": "ok", "observation": result})
            except (AgentError, AgentModelError, OSError, UnicodeError, ValueError, TypeError) as exc:
                if isinstance(exc, AgentError):
                    job["history"].append({"action": "tool", "status": "blocked",
                                           "observation": "Pfad oder Parameter nicht freigegeben."})
                else:
                    job["history"].append({"action": "model", "status": "invalid_or_unavailable"})
                    job["status"] = "failed"
            if job["status"] == "running" and job["steps"] >= self.max_steps:
                job["status"] = "limit_reached"
            return self.snapshot(id)

    def _execute(self, action: str, command: dict, job: dict) -> dict:
        if action == "read":
            path = command.get("path")
            file = self.engine.safe_path(path)
            if file.suffix.lower() not in self.READ_EXTENSIONS or not file.is_file():
                raise AgentError("Datei nicht freigegeben.")
            data = self.engine._read(file)
            job["reads"][path] = self.engine._digest(data.encode("utf-8"))
            return {"file": path, "sha256": job["reads"][path],
                    "preview": data[:5000], "truncated": len(data) > 5000}
        if action == "search":
            query = command.get("query")
            if not isinstance(query, str) or not 1 <= len(query) <= 120:
                raise AgentError("Suchbegriff erforderlich.")
            result = self.engine._search({"goal": job["goal"], "query": query})
            return {"matches": result["matches"][:10], "truncated": result["truncated"] or len(result["matches"]) > 10}
        if action in self.engine.TOOLS and action != "summary":
            result = self.engine.TOOLS[action](self.engine, {"goal": job["goal"], "query": "",
                                                              "path": "." ,"results": {}})
            encoded = json.dumps(result, ensure_ascii=False)
            return result if len(encoded) <= 5000 else {"summary": encoded[:5000], "truncated": True}
        raise AgentModelError("Tool nicht verfügbar.")

    def run(self, id: str) -> dict:
        for _ in range(self.max_steps):
            state = self.snapshot(id)
            if state["status"] != "running":
                return state
            self.step(id)
        return self.snapshot(id)
