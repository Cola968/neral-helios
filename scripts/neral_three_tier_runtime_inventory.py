"""Read-only inventory for three-tier NERAL runtime on the real Windows PC.

No downloads, startup, model loading, credential export, or prompt access.
A frozen HF adapter and a configured search provider are NOT proof of a
running three-tier live chat or of any latency target.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.cookiejar
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
PREVIEW = "http://127.0.0.1:9690"


def _json_get(url: str, *, opener=None, limit: int = 65536):
    if not url.startswith((OLLAMA + "/", PREVIEW + "/")):
        raise ValueError("Only NERAL localhost URLs permitted")
    client = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(url, timeout=5) as resp:
        body = resp.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Unexpectedly large localhost response")
    result = json.loads(body)
    if type(result) is not dict:
        raise ValueError("Invalid localhost envelope")
    return result


def frozen_v2_full(root: Path) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
        manifest = root / "manifest.json"
        weights = root / "adapter" / "adapter_model.safetensors"
        config = root / "adapter" / "adapter_config.json"
        if any(not p.is_file() or p.is_symlink() for p in (manifest, weights, config)):
            return False
        m = json.loads(manifest.read_text("utf-8"))
        if not (m.get("status") == "frozen_adapter_ready_for_heldout_comparison"
                and m.get("family") == "helios"
                and m.get("rank") == 16
                and m.get("train_rows") == 1100
                and m.get("val_rows") == 140
                and m.get("test_rows_opened") == 0
                and m.get("docker_verified") is True):
            return False
        pin = m.get("frozen_adapter_sha256")
        if type(pin) is not str or len(pin) != 64:
            return False
        with weights.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest() == pin
    except (ValueError, TypeError, OSError, KeyError):
        return False


def collect_inventory(adapter_root: Path, *, fetch=_json_get) -> dict:
    out = {
        "v2_full_adapter_frozen_sha256_verified": frozen_v2_full(adapter_root),
        "ollama_4b_installed": False,
        "ollama_9b_installed": False,
        "ollama_available": False,
        "preview_running": False,
        "preview_active_foundation": None,
        "preview_three_tier_runtime_enabled": False,
        "preview_persistent_memory_available": False,
        "preview_document_rag_available": False,
        "preview_web_search_configured": False,
        "preview_web_search_live_verified": False,
        "luna_sub_3_seconds_proven": False,
        "helios_v2_full_adapter_live_proven": False,
        "end_to_end_chat_verified": False,
        "no_model_started_or_downloaded": True,
        "no_user_data_read": True,
    }
    try:
        tags = fetch(OLLAMA + "/api/tags")
        names = {
            item.get("name") for item in tags.get("models", [])
            if type(item) is dict and type(item.get("name")) is str
        }
        out["ollama_available"] = True
        out["ollama_4b_installed"] = "qwen3.5:4b" in names
        out["ollama_9b_installed"] = "qwen3.5:9b" in names
    except (OSError, TypeError, ValueError):
        pass
    try:
        jar = urllib.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
        with opener.open(PREVIEW + "/", timeout=5) as resp:
            resp.read(1)
        # Caller-injected fetch supports no-op CI fakes; production uses opener.
        if fetch is _json_get:
            session = fetch(PREVIEW + "/api/neral/preview/session", opener=opener)
            plugins = fetch(PREVIEW + "/api/neral/preview/plugins", opener=opener)
        else:
            session = fetch(PREVIEW + "/api/neral/preview/session")
            plugins = fetch(PREVIEW + "/api/neral/preview/plugins")
        out["preview_running"] = True
        out["preview_active_foundation"] = session.get("model") if type(session.get("model")) is str else None
        out["preview_three_tier_runtime_enabled"] = session.get("three_tier_enabled") is True
        capabilities = plugins.get("plugins", {})
        if type(capabilities) is dict:
            out["preview_persistent_memory_available"] = capabilities.get("persistent_memory", {}).get("available") is True
            out["preview_document_rag_available"] = capabilities.get("document_rag", {}).get("available") is True
            out["preview_web_search_configured"] = capabilities.get("web_search", {}).get("configured") is True
            out["preview_web_search_live_verified"] = capabilities.get("web_search", {}).get("live_verified") is True
    except (OSError, TypeError, ValueError):
        pass
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = collect_inventory(args.adapter_root)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as file:
            json.dump(report, file, sort_keys=True, indent=2)
    print("NERAL_THREE_TIER_READONLY_INVENTORY " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
