"""HTTP endpoints for NERAL model-directed local agent; inherit host/origin/session guard."""
from __future__ import annotations

import os
from pathlib import Path
from flask import jsonify, request
from agent_mode import AgentError
from neral_agent_autonomy import AutonomousAgent, OllamaLocalBackend, AgentModelError


def resolve_local_adapter_paths(home: Path, env) -> tuple[Path, Path, bool]:
    """Select only preexisting local weights. v5 is an explicit experimental opt-in."""
    channel=env.get("NERAL_AGENT_ADAPTER_CHANNEL","legacy")
    if channel not in ("legacy","v5"):
        raise ValueError("Unbekannter NERAL-Adapterkanal.")
    for name in ("NERAL_AGENT_BASE_PATH","NERAL_AGENT_ADAPTER_PATH"):
        if name in env and (not isinstance(env[name],str) or not env[name].strip()):
            raise ValueError("Leerer lokaler Modellpfad.")
    base=Path(env.get("NERAL_AGENT_BASE_PATH") or
              (home/".cache"/"neral"/"qwen25-smoke"/
               "c03e6d358207e414f1eca0bb1891e29f1db0e242"/
               "installed_models"/"hf-qwen25-coder-7b"))
    adapter_name=("v5-r16-480-42378021782-1" if channel=="v5"
                  else "actions-r32-42078722575-1")
    adapter=Path(env.get("NERAL_AGENT_ADAPTER_PATH") or
                 (home/"NERAL-QLoRA-Research"/adapter_name/"adapter"))
    return base,adapter,(channel=="v5")


def register_autonomous_agent_api(app, workspace, backend=None):
    name = os.environ.get("NERAL_AGENT_OLLAMA_MODEL", "qwen3.5:9b")
    ollama_profile = os.environ.get("NERAL_AGENT_INFERENCE_PROFILE", "legacy")
    if backend is None:
        engine = os.environ.get("NERAL_AGENT_BACKEND", "auto")
        if engine == "auto":
            base,adapter,explicit_v5=resolve_local_adapter_paths(Path.home(),os.environ)
            if explicit_v5:
                # A user explicitly requested v5. Never mislabel Ollama as v5 on failure.
                if not base.is_dir() or not (adapter/"adapter_model.safetensors").is_file():
                    raise ValueError("NERAL v5 ist lokal nicht vollständig vorhanden.")
                from neral_agent_hf import HFAdapterLocalBackend
                backend=HFAdapterLocalBackend(base,adapter)
            elif base.is_dir() and (adapter/"adapter_model.safetensors").is_file():
                from neral_agent_hf import HFAdapterLocalBackend
                try:
                    backend = HFAdapterLocalBackend(base, adapter)
                except (AgentModelError, OSError, ValueError):
                    # Preserve the previous legacy-auto Ollama fallback.
                    backend = OllamaLocalBackend(name, inference_profile=ollama_profile)
            else:
                backend = OllamaLocalBackend(name, inference_profile=ollama_profile)
        elif engine == "hf_adapter":
            from neral_agent_hf import HFAdapterLocalBackend
            if os.environ.get("NERAL_AGENT_ADAPTER_CHANNEL")=="v5":
                base,adapter,_=resolve_local_adapter_paths(Path.home(),os.environ)
            else:
                base = os.environ.get("NERAL_AGENT_BASE_PATH")
                adapter = os.environ.get("NERAL_AGENT_ADAPTER_PATH")
                if not base or not adapter:
                    raise ValueError("Für HF-Adapter sind lokale Modellpfade erforderlich.")
            backend = HFAdapterLocalBackend(base, adapter)
        elif engine == "ollama":
            backend = OllamaLocalBackend(name, inference_profile=ollama_profile)
        else:
            raise ValueError("Unbekanntes lokales Agentenmodell.")
    controller = AutonomousAgent(workspace, backend)

    def request_json():
        if not request.is_json:
            return None, (jsonify(error="Nur JSON erlaubt."), 415)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, (jsonify(error="JSON-Objekt erforderlich."), 400)
        return data, None

    def safe_call(fn, status=200):
        try:
            return jsonify(fn()), status
        except AgentError as exc:
            return jsonify(error=str(exc)), 400
        except (OSError, UnicodeError):
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
                       capabilities=["model_tool_selection", "read", "search", "web_search_opt_in", "analyze", "patch_proposal",
                                     "solve_math_symbolic", "eval_python_expr", "syntax_check",
                                     "run_isolated_unit_test", "verify_before_finish",
                                     "isolated_docker_tests", "explicit_approval", "cancel", "pause", "resume"],
                       limitations=["no_browser", "no_shell", "no_unattended_writes", "process_memory_only",
                                    "web_search_sends_explicit_query_to_duckduckgo_only"])

    @app.post("/api/agent/auto/start")
    def autonomous_start():
        data, err = request_json()
        if err:
            return err
        return safe_call(lambda: controller.start(data.get("goal"),
                                test_cases=data.get("test_cases"),
                                test_path=data.get("test_path"),
                                allow_web_search=data.get("allow_web_search", False),
                                web_query=data.get("web_query")), 201)

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
        return safe_call(do)
    return controller
