"""Flask endpoints for the bounded NERAL agent. Inherit app-wide session/origin guards."""
from __future__ import annotations

from flask import jsonify, request
from agent_mode import AgentEngine, AgentError


def register_agent_api(app, workspace):
    engine = AgentEngine(workspace)

    def body():
        if not request.is_json:
            return None, (jsonify(error="Nur JSON erlaubt."), 415)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, (jsonify(error="JSON-Objekt erwartet."), 400)
        return data, None

    def attempt(operation, *, status=200):
        try:
            result = operation()
            return jsonify(result), status
        except AgentError as exc:
            return jsonify(error=str(exc)), 400
        except (OSError, UnicodeError):
            return jsonify(error="Lokaler Dateizugriff fehlgeschlagen."), 409

    @app.get("/api/agent/capabilities")
    def agent_capabilities():
        return jsonify(engine.capabilities())

    @app.post("/api/agent/start")
    def agent_start():
        data, err = body()
        if err:
            return err
        return attempt(lambda: engine.create(data.get("goal"), data.get("strategy", "audit"),
                                             data.get("path", "."), data.get("query", "")), status=201)

    @app.post("/api/agent/check")
    def agent_check():
        data, err = body()
        if err:
            return err

        def run():
            file = engine.safe_path(data.get("path"))
            if not file.is_file():
                return {"exists": False, "sha256": None, "preview": ""}
            size = file.stat().st_size
            if size > engine.MAX_EDIT:
                raise AgentError("Datei überschreitet Bearbeitungsgrenze.")
            payload = file.read_bytes()
            try:
                txt = payload.decode("utf-8")
            except UnicodeError:
                raise AgentError("Datei ist kein UTF-8-Text.")
            return {"exists": True, "sha256": engine._digest(payload),
                    "preview": txt[:8000], "bytes": size}
        return attempt(run)

    @app.get("/api/agent/<ident>")
    def agent_snapshot(ident):
        return attempt(lambda: engine.snapshot(ident))

    @app.post("/api/agent/<ident>/<action>")
    def agent_action(ident, action):
        allowed = {"next", "cancel", "pause", "resume", "propose", "quickfix", "approve", "undo"}
        if action not in allowed:
            return jsonify(error="Unbekannte Agentenaktion."), 404
        data, err = body()
        if err:
            return err

        def run():
            if action == "next":
                return engine.next(ident)
            if action == "cancel":
                return engine.cancel(ident)
            if action == "pause":
                return engine.pause(ident)
            if action == "resume":
                return engine.resume(ident)
            if action == "undo":
                return engine.undo(ident)
            if action == "propose":
                return engine.propose(ident, path=data.get("path"), content=data.get("content"),
                                      expected_sha=data.get("expected_sha"))
            if action == "quickfix":
                return engine.propose_fix(ident, path=data.get("path"), kind=data.get("kind"))
            return engine.approve(ident, data.get("token"))

        return attempt(run)

    # Intentionally do not expose a mutable global task dict to HTTP clients.
    return engine
