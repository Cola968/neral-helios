"""Verified, explicitly configured V4.3 local web launcher. No model downloads."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:9690"


def require_web_profile(app):
    """Use the public session-protected endpoint as the boot contract."""
    with app.test_client() as client:
        home = client.get("/", base_url=BASE_URL)
        status = client.get("/api/neral/chat/three-tier/status", base_url=BASE_URL)
        data = status.get_json(silent=True) or {}
    web = data.get("web_search") or {}
    if (home.status_code != 200 or status.status_code != 200
            or data.get("v43_profile") is not True
            or web.get("configured") is not True
            or web.get("live_enabled") is not True):
        raise RuntimeError("V4.3-Webprofil fehlt; Serverstart abgebrochen.")
    return data


def build_verified_app(legacy, luna, helios, *, execute_python=None):
    from neral_preview_server import create_preview_app
    from neral_three_tier_runtime import ThreeTierRuntime
    from neral_web_search import WebSearch

    runtime = ThreeTierRuntime(
        luna, helios, legacy_backend=legacy, execute_python=execute_python,
        evidence_agent=False, v43_profile=True,
    )
    app = create_preview_app(
        legacy, three_tier_runtime=runtime, search_service=WebSearch(), port=9690,
    )
    require_web_profile(app)
    return runtime, app


def main():
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
        if probe.connect_ex(("127.0.0.1", 9690)) == 0:
            raise RuntimeError("Port 9690 ist belegt. Bitte altes NERAL-Fenster zuerst schliessen.")
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
    runtime, app = build_verified_app(legacy, luna, helios)
    print("NERAL V4.3: Webprofil und Suchdienst bestaetigt.", flush=True)
    print("Lokale Oberflaeche: " + BASE_URL, flush=True)
    server.warmup_three_tier_before_serve(
        runtime, app, serve, host="127.0.0.1", port=9690, threads=4,
    )


if __name__ == "__main__":
    main()
