"""Verified, explicitly configured V4.3 local web launcher. No model downloads."""
from __future__ import annotations

import os
import argparse
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent


def require_web_profile(app, port=9690):
    """Use the public session-protected endpoint as the boot contract."""
    with app.test_client() as client:
        base=f"http://127.0.0.1:{port}"
        home = client.get("/", base_url=base)
        status = client.get("/api/neral/chat/three-tier/status", base_url=base)
        data = status.get_json(silent=True) or {}
    web = data.get("web_search") or {}
    if (home.status_code != 200 or status.status_code != 200
            or data.get("v43_profile") is not True
            or data.get("amber") != "qwen3.5:4b"
            or data.get("quartz") != "qwen3.5:9b"
            or data.get("obsidian") != "qwen3.5:9b"
            or data.get("adapter_loaded") is not False
            or web.get("configured") is not True
            or web.get("live_enabled") is not True):
        raise RuntimeError("V4.3-Webprofil fehlt; Serverstart abgebrochen.")
    return data


def build_verified_app(legacy, luna, helios, *, execute_python=None, port=9690):
    from neral_preview_server import create_preview_app
    from neral_three_tier_runtime import ThreeTierRuntime
    from neral_web_search import WebSearch

    runtime = ThreeTierRuntime(
        luna, helios, legacy_backend=legacy if port==9690 else None, execute_python=execute_python,
        evidence_agent=False, v43_profile=True,
    )
    app = create_preview_app(
        legacy, three_tier_runtime=runtime, search_service=WebSearch(), port=port,
    )
    if port!=9690:
        app.config["SESSION_COOKIE_NAME"]=f"neral-v43-{port}"
    require_web_profile(app,port)
    return runtime, app


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--port",type=int,choices=(9690,9691,9692),default=9690)
    port=parser.parse_args().port
    # Resolve the source explicitly; do not reuse an installed preview module.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(ROOT / "local_state" / "three-tier-web-deps"))
    os.environ.update({
        "HF_HOME": "D:/neral-qlora-v2/hf-cache",
        "HF_HUB_CACHE": "D:/neral-qlora-v2/hf-cache/hub",
        "HF_XET_CACHE": "D:/neral-qlora-v2/hf-cache/xet",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NERAL_V3_AGENT_ENABLED": "0",
    })
    with socket.socket() as probe:
        probe.settimeout(2)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} ist belegt; V4.3-Start abgebrochen.")
    import neral_preview_server as server
    import neral_three_tier_runtime as tiers
    import neral_web_search as search
    import neral_v43_live_web as router
    from neral_agent_autonomy import OllamaLocalBackend
    from waitress import serve
    for module in (server, tiers, search, router):
        if Path(module.__file__).resolve().parent != ROOT:
            raise RuntimeError("Falsches NERAL-Modul: " + module.__name__)
    legacy = OllamaLocalBackend("qwen3:8b", timeout=90, inference_profile="legacy")
    luna = tiers.FrozenQwen35HF(Path("D:/neral-qlora-v2/qwen35-4b-hf"))
    helios = tiers.FrozenQwen35HF(Path("D:/neral-qlora/checkpoint"), raw_9b=True)
    runtime, app = build_verified_app(legacy, luna, helios, port=port)
    print("NERAL V4.3: Webprofil und Suchdienst bestaetigt.", flush=True)
    print(f"Lokale Oberflaeche: http://127.0.0.1:{port}", flush=True)
    server.warmup_three_tier_before_serve(
        runtime, app, serve, host="127.0.0.1", port=port, threads=4,
    )


if __name__ == "__main__":
    main()
