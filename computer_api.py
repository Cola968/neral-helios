"""Direct-localhost-only Flask endpoints for opt-in NERAL computer use.

Remote LAN/paired-phone gateway is explicitly blocked for every endpoint.
The existing app-wide NERAL session guard still authenticates local calls.
"""
from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit
from flask import jsonify, request

from computer_use import ComputerUse, ComputerUseError, WindowsDesktop


def register_computer_api(app, *, enabled: bool = False, backend=None, clock=None):
    broker = ComputerUse(backend, clock=clock) if enabled and backend else None

    @app.before_request
    def computer_local_only_guard():
        if not request.path.startswith("/api/computer/"):
            return None
        try:
            address = ipaddress.ip_address(request.remote_addr or "")
        except ValueError:
            return jsonify(error="Computer Use ist nur lokal verfügbar."), 403
        if (not address.is_loopback or request.headers.get("X-Helios-Gateway") is not None
                or request.headers.get("Sec-Fetch-Site") == "cross-site"):
            return jsonify(error="Computer Use ist nur lokal verfügbar."), 403
        # Do not trust reverse-proxy authority or cross-origin browser requests.
        authority = request.host
        if authority.split(":", 1)[0].lower() not in {"localhost", "127.0.0.1", "[::1]"}:
            return jsonify(error="Ungültiger Computer-Use-Host."), 403
        origin = request.headers.get("Origin")
        if origin:
            u = urlsplit(origin)
            if u.scheme != "http" or u.netloc != authority or u.path or u.query or u.fragment:
                return jsonify(error="Fremde Browser-Origin blockiert."), 403

    def body():
        if not request.is_json:
            return None, (jsonify(error="JSON erforderlich."), 415)
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):
            return None, (jsonify(error="JSON-Objekt erforderlich."), 400)
        return payload,None

    def run(fn):
        try:
            return jsonify(fn())
        except ComputerUseError as exc:
            return jsonify(error=str(exc)),400
        except (OSError, RuntimeError):
            return jsonify(error="Computer Use ist lokal nicht verfügbar."),503

    @app.post("/api/computer/status")
    def computer_status():
        if not enabled:
            return jsonify(enabled=False,computer_use=False,
                           note="Computer Use ist standardmäßig deaktiviert.")
        return jsonify(broker.status() if broker is not None
                       else {"enabled":False,"computer_use":True,
                             "note":"Lokale Aktivierung erforderlich."})

    @app.post("/api/computer/start")
    def computer_start():
        nonlocal broker
        data,err=body()
        if err:return err
        if not enabled:
            return jsonify(error="Computer Use wurde nicht lokal freigegeben."),403
        if data:
            return jsonify(error="Aktivierung erwartet keine Parameter."),400
        if broker is None:
            try:
                broker=ComputerUse(WindowsDesktop(),clock=clock)
            except ComputerUseError as exc:
                return jsonify(error=str(exc)),503
        return run(broker.start)

    @app.post("/api/computer/screenshot")
    def computer_screenshot():
        if not enabled:return jsonify(error="Computer Use deaktiviert."),403
        _,err=body()
        if err:return err
        return run(broker.screenshot if broker else lambda: (_ for _ in ()).throw(
            ComputerUseError("Computer Use ist noch nicht gestartet.")))

    @app.post("/api/computer/preview")
    def computer_preview():
        if not enabled:return jsonify(error="Computer Use deaktiviert."),403
        payload,err=body()
        if err:return err
        if broker is None:
            return jsonify(error="Computer Use nicht aktiv."),400
        return run(lambda: broker.preview(payload))

    @app.post("/api/computer/execute")
    def computer_execute():
        if not enabled:return jsonify(error="Computer Use deaktiviert."),403
        payload,err=body()
        if err:return err
        if set(payload)!={"approval_token"} or broker is None:
            return jsonify(error="Freigabe fehlt."),400
        return run(lambda: broker.execute(payload["approval_token"]))

    @app.post("/api/computer/stop")
    def computer_stop():
        if not enabled:return jsonify(error="Computer Use deaktiviert."),403
        _,err=body()
        if err:return err
        return run(broker.stop if broker is not None else lambda: {"enabled":False})

    return broker
