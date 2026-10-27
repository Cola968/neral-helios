"""One-click local NERAL Assistant entry point.

No auto-download, model modification, network/cloud hosting or GPU training.
Only the local Ollama model 'qwen3.5:9b' may be used.
"""
from __future__ import annotations

import http.cookiejar
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_URL = "http://127.0.0.1:9680"
OLLAMA_URL = "http://127.0.0.1:11434/api/tags"
MODEL = "qwen3.5:9b"


class LauncherError(RuntimeError):
    """User-actionable local startup failure."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        NoRedirect(),
    )


def _load_json(opener, url):
    with opener.open(url, timeout=4) as response:
        raw = response.read(8192 + 1)
    if len(raw) > 8192:
        raise LauncherError("Lokaler Dienst sendet eine unzulässig große Antwort.")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherError("Lokaler Dienst sendet kein gültiges JSON.") from exc


def installed_ollama_model(fetch_json) -> bool:
    data = fetch_json(OLLAMA_URL)
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise LauncherError("Ollama liefert keine gültige Modellliste.")
    for item in data["models"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise LauncherError("Ollama liefert eine ungültige Modellliste.")
    return any(item["name"] == MODEL for item in data["models"])


def assistant_ready(fetch_json) -> bool:
    data = fetch_json(APP_URL + "/api/info")
    return (
        isinstance(data, dict)
        and data.get("name") == "Neral / Helios"
        and isinstance(data.get("assistant"), dict)
        and data["assistant"].get("enabled") is True
        and data["assistant"].get("base_model") == MODEL
        and data["assistant"].get("local_only") is True
    )


def assistant_environment(source=None):
    env = dict(os.environ if source is None else source)
    env.update({
        "NERAL_ASSISTANT_ENABLED": "1",
        "NERAL_ASSISTANT_MODEL": MODEL,
        "NERAL_ASSISTANT_THINK": "0",
        "HELIOS_PREPARATION_MODE": "0",
        "HELIOS_DEVICE": "cpu",
        "HELIOS_PORT": "9680",
        "CUDA_VISIBLE_DEVICES": "",
    })
    return env


def check_dependencies():
    missing = [name for name in ("torch", "numpy", "flask", "waitress")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise LauncherError(
            "NERAL-Python-Abhängigkeiten fehlen: " + ", ".join(missing) +
            ". Im richtigen Python-Environment requirements.txt und CPU-PyTorch installieren; "
            "Anleitung: NERAL_ASSISTANT_LOCAL.md."
        )


def current_app_status(opener):
    # First GET sets NERAL's HttpOnly loopback-only session cookie.
    try:
        with opener.open(APP_URL + "/", timeout=3) as response:
            response.read(1024)
    except (urllib.error.URLError, TimeoutError, OSError):
        return "offline"
    try:
        return "ready" if assistant_ready(lambda url: _load_json(opener, url)) else "wrong"
    except (LauncherError, urllib.error.URLError, ValueError, TimeoutError, OSError):
        return "wrong"


def main():
    check_dependencies()
    opener = _opener()
    try:
        has_model = installed_ollama_model(lambda url: _load_json(opener, url))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LauncherError(
            "Ollama läuft nicht. Starte Ollama und prüfe http://127.0.0.1:11434."
        ) from exc
    if not has_model:
        raise LauncherError(
            "Qwen3.5-9B ist nicht installiert. Führe einmalig 'ollama pull qwen3.5:9b' aus."
        )

    existing = current_app_status(opener)
    if existing == "ready":
        print("NERAL Assistant ist bereits aktiv.")
        webbrowser.open(APP_URL)
        return
    if existing == "wrong":
        raise LauncherError(
            "Port 9680 ist durch eine andere/alte NERAL-App belegt. "
            "Schließe die alte App zuerst; vorhandene Chats nicht löschen."
        )

    logs = ROOT / "local_state"
    logs.mkdir(exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with (logs / "neral-assistant-launch.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "app.py")],
            cwd=ROOT, env=assistant_environment(), stdout=log, stderr=log,
            creationflags=flags,
        )
    for _ in range(45):
        if process.poll() is not None:
            raise LauncherError(
                "NERAL ist beim Start abgestürzt. Details: local_state/neral-assistant-launch.log"
            )
        status = current_app_status(opener)
        if status == "ready":
            print("NERAL Assistant läuft. Qwen3.5-9B wird lokal über Ollama genutzt.")
            webbrowser.open(APP_URL)
            return
        if status == "wrong":
            process.terminate()
            raise LauncherError(
                "Unerwarteter Dienst an Port 9680. NERAL Assistant wurde nicht gestartet."
            )
        time.sleep(1)
    process.terminate()
    raise LauncherError(
        "NERAL antwortet beim Start nicht. Details: local_state/neral-assistant-launch.log"
    )


if __name__ == "__main__":
    try:
        main()
    except LauncherError as exc:
        print("NERAL Assistant: " + str(exc), file=sys.stderr)
        sys.exit(1)
