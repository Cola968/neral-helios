"""HTTP endpoints for NERAL model-directed local agent; inherit host/origin/session guard."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from flask import jsonify, request
from agent_mode import AgentError
from neral_agent_autonomy import AutonomousAgent, OllamaLocalBackend, AgentModelError
from neral_mission_journal import MissionJournal


def register_autonomous_agent_api(app, workspace, backend=None):
    name = os.environ.get("NERAL_AGENT_OLLAMA_MODEL", "qwen3.5:9b")
    if backend is None:
        engine = os.environ.get("NERAL_AGENT_BACKEND", "auto")
        if engine == "auto":
            home = Path.home()
            base = Path(os.environ.get("NERAL_AGENT_BASE_PATH") or
                        (home / ".cache" / "neral" / "qwen25-smoke" /
                         "c03e6d358207e414f1eca0bb1891e29f1db0e242" /
                         "installed_models" / "hf-qwen25-coder-7b"))
            adapter = Path(os.environ.get("NERAL_AGENT_ADAPTER_PATH") or
                           (home / "NERAL-QLoRA-Research" / "actions-r32-42078722575-1" / "adapter"))
            if base.is_dir() and (adapter / "adapter_model.safetensors").is_file():
                from neral_agent_hf import HFAdapterLocalBackend
                try:
                    backend = HFAdapterLocalBackend(base, adapter)
                except (AgentModelError, OSError, ValueError):
                    # Auto mode must not prevent the existing local Helios app
                    # from booting if experimental adapter files are invalid.
                    # Provenance will truthfully show foundation-only Ollama.
                    backend = OllamaLocalBackend(name)
            else:
                backend = OllamaLocalBackend(name)
        elif engine == "hf_adapter":
            from neral_agent_hf import HFAdapterLocalBackend
            base = os.environ.get("NERAL_AGENT_BASE_PATH")
            adapter = os.environ.get("NERAL_AGENT_ADAPTER_PATH")
            if not base or not adapter:
                # Do not silently switch models when a trained adapter is requested.
                raise ValueError("Für HF-Adapter sind lokale Modellpfade erforderlich.")
            backend = HFAdapterLocalBackend(base, adapter)
        elif engine == "ollama":
            backend = OllamaLocalBackend(name)
        else:
            raise ValueError("Unbekanntes lokales Agentenmodell.")
    controller = AutonomousAgent(workspace, backend)
    journal_path = os.environ.get("NERAL_MISSION_JOURNAL_PATH")
    journal = MissionJournal(journal_path) if journal_path else None

    def request_json():
        if not request.is_json:
            return None, (jsonify(error="Nur JSON erlaubt."), 415)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, (jsonify(error="JSON-Objekt erforderlich."), 400)
        return data, None

    def safe_call(fn, status=200, *, record=False):
        try:
            response = fn()
            if record and journal is not None:
                try:
                    journal.record(response)
                except (sqlite3.Error, OSError):
                    # Report persistence failure without hiding an agent action
                    # that may already have completed (including a file edit).
                    response = dict(response, mission_history_saved=False)
            return jsonify(response), status
        except AgentError as exc:
            return jsonify(error=str(exc)), 400
        except (OSError, UnicodeError, sqlite3.Error):
            return jsonify(error="Lokale Verarbeitung fehlgeschlagen."), 409

    @app.get("/api/agent/auto/status")
    def autonomous_status():
        is_adapter = hasattr(backend, "adapter_sha256")
        loaded = is_adapter and getattr(backend, "_model", None) is not None
        return jsonify(provider="hf_local_adapter" if is_adapter else "ollama_local_only",
                       model=getattr(backend, "model", "test_backend"),
                       confirmed_available=bool(loaded),
                       model_identity=("neral_local_adapter_loaded_experimental_not_quality_promoted" if loaded
                                       else "local_adapter_configured_not_yet_loaded" if is_adapter
                                       else "foundation_or_custom_model_not_verified_neral_adapter"),
                       capabilities=["model_tool_selection", "read", "search", "solve", "analyze", "patch_proposal",
                                     "isolated_docker_tests", "explicit_approval", "cancel", "pause", "resume"] +
                                    (["local_metadata_history"] if journal is not None else []),
                       limitations=["no_browser", "no_shell", "no_unattended_writes", "process_memory_only"] +
                                   (["history_does_not_resume_jobs"] if journal is not None else ["history_disabled_by_default"]))

    @app.get("/api/agent/auto/history")
    def autonomous_history():
        return safe_call(lambda: {"enabled": journal is not None,
                                  "missions": journal.list_recent() if journal is not None else []})

    @app.post("/api/agent/auto/start")
    def autonomous_start():
        data, err = request_json()
        if err:
            return err
        return safe_call(lambda: controller.start(data.get("goal")), 201, record=True)

    @app.get("/api/agent/auto/<ident>")
    def autonomous_get(ident):
        return safe_call(lambda: controller.snapshot(ident))

    @app.post("/api/agent/auto/<ident>/<action>")
    def autonomous_action(ident, action):
        if action not in {"step", "run", "pause", "resume", "cancel", "approve", "reject"}:
            return jsonify(error="Unbekannte Aktion."), 404
        data, err = request_json()
        if err:
            return err
        def do():
            if action == "step":
                return controller.step(ident)
            if action == "run":
                return controller.run(ident)
            if action == "pause":
                return controller.pause(ident)
            if action == "resume":
                return controller.resume(ident)
            if action == "cancel":
                return controller.cancel(ident)
            if action == "reject":
                return controller.reject(ident)
            return controller.approve(ident, data.get("token"))
        return safe_call(do, record=True)
    return controller
