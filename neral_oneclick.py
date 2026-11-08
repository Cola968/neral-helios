"""NERAL for Windows: download NERAL.exe and double-click.

This EXE bundles only the local web application and Python runtime; the existing
Ollama model stays on the user's PC. No shell, cloud API, model download, or
package installer is invoked. Build with the audited GitHub Windows workflow.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

HOST = "127.0.0.1"
PORT = 9690
URL = f"http://{HOST}:{PORT}/"
OLLAMA_TAGS = "http://127.0.0.1:11434/api/tags"
MODELS = ("qwen3:8b", "qwen3.5:9b")


def has_model(payload: object, model: str) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return False
    return any(isinstance(item, dict) and item.get("name") == model
               for item in payload["models"])


def _tags() -> dict:
    req = urllib.request.Request(OLLAMA_TAGS, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=2.5) as response:
        if response.status != 200:
            raise OSError("Ollama ist nicht bereit")
        data = response.read(256_000)
    info = json.loads(data)
    if not isinstance(info, dict):
        raise OSError("Ungültige Ollama-Antwort")
    return info


def _local_ollama_executable() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
        ]
        return next((str(path) for path in candidates if path.is_file()), None)
    return None


def ensure_existing_local_model() -> tuple[str, subprocess.Popen | None]:
    """Start installed Ollama if needed, never pull or silently install models."""
    owned = None
    try:
        inventory = _tags()
    except (OSError, ValueError, urllib.error.URLError):
        executable = _local_ollama_executable()
        if executable is None:
            raise RuntimeError("Ollama fehlt. Bitte zuerst Ollama installieren.")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            owned = subprocess.Popen([executable, "serve"], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=flags)
        except OSError as exc:
            raise RuntimeError("Ollama konnte nicht gestartet werden.") from exc
        inventory = None
        for _ in range(32):
            try:
                inventory = _tags()
                break
            except (OSError, ValueError, urllib.error.URLError):
                if owned.poll() is not None:
                    break
                time.sleep(0.35)
        if inventory is None:
            owned.terminate()
            raise RuntimeError("Ollama ist installiert, antwortet aber nicht.")
    for tag in MODELS:
        if has_model(inventory, tag):
            return tag, owned
    if owned is not None and owned.poll() is None:
        owned.terminate()
    raise RuntimeError("Das lokale Modell qwen3:8b oder qwen3.5:9b fehlt in Ollama.")


def _port_taken() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, PORT)) == 0


def run_smoke_test() -> int:
    """No Ollama needed: verify actual frozen Flask assets over loopback."""
    from waitress import create_server
    from neral_preview_server import create_preview_app
    class Fake:
        model = "qwen3:8b"
        def respond_chat(self, messages):
            return "Lokaler Test"
    if _port_taken():
        return 9
    server = create_server(create_preview_app(Fake(), secret_key=b"Q"*32),
                           host=HOST, port=PORT, threads=2)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        for path, marker in (
            ("", b"NERAL 1.0 Local"),
            ("static/neral-preview.js", b"neral"),
            ("static/neral-accounts.js", b"neral"),
            ("static/icon.svg", b"<svg"),
        ):
            with opener.open(URL+path, timeout=5) as response:
                body = response.read(250_000)
                if response.status != 200 or marker not in body:
                    return 7
        with opener.open(URL+"api/neral/preview/session", timeout=5) as response:
            info = json.load(response)
            if info.get("model") != "qwen3:8b" or info.get("adapter_promoted") is not False:
                return 8
        return 0
    except (OSError, ValueError, urllib.error.URLError):
        return 6
    finally:
        server.close()
        thread.join(timeout=3)


def start_app() -> None:
    """Tiny native status window: open the actual app in the default browser."""
    import tkinter as tk
    from tkinter import messagebox
    from waitress import create_server
    from neral_preview_server import create_preview_app
    from neral_agent_autonomy import OllamaLocalBackend

    root = tk.Tk()
    root.title("NERAL")
    root.geometry("360x155")
    root.resizable(False, False)
    root.configure(bg="#f8f8f7")
    root.attributes("-topmost", False)
    tk.Label(root, text="neral", font=("Segoe UI", 18, "bold"),
             bg="#f8f8f7", fg="#202120").pack(pady=(16, 5))
    status = tk.StringVar(value="NERAL wird geöffnet …")
    tk.Label(root, textvariable=status, font=("Segoe UI", 10),
             bg="#f8f8f7", fg="#666966").pack()
    controls = tk.Frame(root, bg="#f8f8f7")
    controls.pack(pady=(13, 0))
    open_button = tk.Button(controls, text="Chat öffnen", command=lambda: webbrowser.open(URL),
                            width=15, state="disabled")
    open_button.pack(side="left", padx=6)
    server = None
    owned = None

    def quit_app():
        nonlocal server, owned
        if server is not None:
            server.close()
            server = None
        if owned is not None and owned.poll() is None:
            owned.terminate()
            owned = None
        root.destroy()

    tk.Button(controls, text="Beenden", command=quit_app,
              width=12).pack(side="left", padx=6)
    root.protocol("WM_DELETE_WINDOW", quit_app)

    def boot():
        nonlocal server, owned
        if _port_taken():
            status.set("NERAL oder ein anderer Dienst ist bereits geöffnet.")
            messagebox.showwarning(
                "NERAL", "Port 9690 ist bereits belegt. Bitte die vorhandene NERAL-Instanz prüfen.")
            return
        try:
            tag, owned = ensure_existing_local_model()
            app = create_preview_app(OllamaLocalBackend(tag, timeout=90))
            server = create_server(app, host=HOST, port=PORT, threads=4)
            worker = threading.Thread(target=server.run, name="neral-local-http", daemon=True)
            worker.start()
            status.set("NERAL läuft lokal · " + tag)
            open_button.configure(state="normal")
            root.after(250, lambda: webbrowser.open(URL))
        except Exception as exc:
            # Do not show internal paths, tracebacks or credentials in the UI.
            status.set("Start nicht möglich.")
            message = str(exc) if isinstance(exc, RuntimeError) else (
                "Die lokale Anwendung konnte nicht gestartet werden.")
            messagebox.showerror("NERAL", message)

    root.after(100, boot)
    root.mainloop()


def main() -> int:
    if sys.argv[1:] == ["--smoke-test"]:
        return run_smoke_test()
    if len(sys.argv) > 1:
        return 2
    start_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
