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

import neral_auto_update

try:
    from neral_build_info import BUILD_COMMIT, BUILD_UTC
except ImportError:
    # Source checkout / developer use is never an automatically updating EXE.
    BUILD_COMMIT, BUILD_UTC = "", ""

HOST = "127.0.0.1"
PORT = 9690
URL = f"http://{HOST}:{PORT}/"
OLLAMA_TAGS = "http://127.0.0.1:11434/api/tags"
MODELS = ("qwen3.5:9b", "qwen3:8b")


def _update_state() -> Path:
    """Keep updates out of model folders, repo files and account data."""
    appdata = os.environ.get("LOCALAPPDATA")
    base = Path(appdata) if appdata and Path(appdata).is_absolute() else (
        Path.home() / "AppData" / "Local")
    return base / "NERAL" / "updates"


def _launch_staged_update() -> bool:
    """A previously verified update becomes active on the NEXT app launch."""
    if not getattr(sys, "frozen", False) or os.environ.get("NERAL_AUTO_UPDATE") == "0":
        return False
    staged = neral_auto_update.staged_windows_exe(
        _update_state(), BUILD_COMMIT, BUILD_UTC)
    if staged is None:
        return False
    if Path(sys.executable).resolve() == staged.resolve():
        return False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen([str(staged)], cwd=str(staged.parent),
                         stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags)
    except OSError:
        return False
    return True


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


def backend_for_model(tag: str):
    """Create an approved local backend using the matching measured profile.

    The stronger Qwen3.5 release is opt-in in source ZIP, and preferred for
    the one-click EXE only if its weights are already installed. This is
    not evidence of a newly trained NERAL foundation model.
    """
    if tag not in MODELS:
        raise ValueError("Nur geprüfte lokale Modellnamen sind zulässig.")
    from neral_agent_autonomy import OllamaLocalBackend
    profile = "measured_nonthinking_2048" if tag == "qwen3.5:9b" else "legacy"
    return OllamaLocalBackend(tag, timeout=90, inference_profile=profile)


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
    root = tk.Tk()
    root.title("NERAL")
    root.geometry("420x210")
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
    update_button = tk.Button(root, text="Update herunterladen …",
                              state="disabled", width=31)
    # Leave disabled/hidden until a verified release is downloaded.
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
            app = create_preview_app(backend_for_model(tag))
            server = create_server(app, host=HOST, port=PORT, threads=4)
            worker = threading.Thread(target=server.run, name="neral-local-http", daemon=True)
            worker.start()
            status.set("NERAL läuft lokal · " + tag +
                       (" · starkes Testprofil" if tag == "qwen3.5:9b" else " · Standard"))
            open_button.configure(state="normal")
            root.after(250, lambda: webbrowser.open(URL))
        except Exception as exc:
            # Do not show internal paths, tracebacks or credentials in the UI.
            status.set("Start nicht möglich.")
            message = str(exc) if isinstance(exc, RuntimeError) else (
                "Die lokale Anwendung konnte nicht gestartet werden.")
            messagebox.showerror("NERAL", message)

    def restart_into_update() -> None:
        # Explicit restart avoids losing in-flight conversations.
        if os.environ.get("NERAL_AUTO_UPDATE") == "0":
            return
        staged = neral_auto_update.staged_windows_exe(
            _update_state(), BUILD_COMMIT, BUILD_UTC)
        if staged is None:
            status.set("Update nicht mehr gültig.")
            return
        quit_app()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            # Wait briefly after Waitress closes before binding the same port.
            time.sleep(0.4)
            subprocess.Popen([str(staged)], cwd=str(staged.parent),
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError:
            pass  # The verified version stays staged for the next manual start.

    update_button.configure(command=restart_into_update)

    def start_update_check() -> None:
        if not getattr(sys, "frozen", False) or os.environ.get("NERAL_AUTO_UPDATE") == "0":
            return

        def check() -> None:
            try:
                staged = neral_auto_update.check_windows_release(
                    _update_state(), BUILD_COMMIT, BUILD_UTC)
                if staged is not None:
                    def show_available() -> None:
                        if root.winfo_exists():
                            status.set("Geprüftes Update bereit · Neustart möglich")
                            update_button.configure(
                                text="Update installieren und neu starten",
                                state="normal")
                            update_button.pack(pady=(8, 0))
                    root.after(0, show_available)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                pass  # Offline is normal. Never interrupt local inference.
            finally:
                try:
                    root.after(3_600_000, start_update_check)
                except RuntimeError:
                    pass

        threading.Thread(target=check, name="neral-update-check",
                         daemon=True).start()

    root.after(100, boot)
    root.after(1_000, start_update_check)
    root.mainloop()


def main() -> int:
    if sys.argv[1:] == ["--smoke-test"]:
        return run_smoke_test()
    if len(sys.argv) > 1:
        return 2
    if _launch_staged_update():
        return 0
    start_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
