"""Private HTTPS phone companion for NERAL Mini, same GPU model and no cloud.

Starts the loopback Mini server and an IP-specific HTTPS LAN gateway in the
same process. An explicit pairing code and expiring HTTPS cookie protect the
private device. No model weights or prompts leave the PC except via the
encrypted, paired phone connection. Never bind 0.0.0.0 or open router ports.
"""
from __future__ import annotations

import argparse
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import socket
import ssl
import threading
import time
from urllib.parse import parse_qs

from mini.neral_mini import (
    BASE_REVISION, MAX_BODY, NeralModel, MiniError, MiniServer,
    create_server, offline, parse_chat,
)

SESSION_SECONDS = 8 * 3600
MAX_SESSIONS = 16
MAX_ATTEMPTS = 5
RATE_WINDOW = 60


def _private_ipv4(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (ip.version == 4 and ip.is_private and not ip.is_loopback
            and not ip.is_unspecified and not ip.is_link_local
            and not ip.is_multicast and not ip.is_reserved)


def detect_host() -> str:
    """Select the host's outgoing IPv4 without sending application data."""
    candidate = os.environ.get("NERAL_MINI_MOBILE_HOST")
    if not candidate:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            candidate = probe.getsockname()[0]
    if not _private_ipv4(candidate):
        raise ValueError("Keine konkrete private WLAN-IPv4 erkannt. NERAL_MINI_MOBILE_HOST setzen.")
    return candidate


class MobileGateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, bind: tuple[str, int], config: dict, desktop: MiniServer):
        self.config = dict(config)
        self.desktop = desktop
        self.origin = f"https://{config['host']}:{config['port']}"
        self.mobile_sessions: dict[str, float] = {}
        self.attempts: dict[str, list[float]] = {}
        self.lock = threading.RLock()
        super().__init__(bind, MobileHandler)

    def valid_session(self, token: str) -> bool:
        if not token or len(token) > 128:
            return False
        with self.lock:
            now = time.monotonic()
            expiry = self.mobile_sessions.get(token, 0)
            if expiry <= now:
                self.mobile_sessions.pop(token, None)
                return False
            return True

    def pair(self, address: str, code: str) -> str | None:
        now = time.monotonic()
        with self.lock:
            recent = [at for at in self.attempts.get(address, []) if now-at < RATE_WINDOW]
            self.attempts[address] = recent + [now]
            if len(recent) >= MAX_ATTEMPTS:
                raise MiniError("rate-limit")
            supplied = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(supplied, self.config["pair_hash"]):
                return None
            for key, expiry in list(self.mobile_sessions.items()):
                if expiry <= now:
                    del self.mobile_sessions[key]
            if len(self.mobile_sessions) >= MAX_SESSIONS:
                raise MiniError("too-many-sessions")
            token = secrets.token_urlsafe(32)
            self.mobile_sessions[token] = now + SESSION_SECONDS
            return token


class MobileHandler(BaseHTTPRequestHandler):
    server: MobileGateway

    def setup(self):
        super().setup()
        self.connection.settimeout(180)

    def log_message(self, *_args):
        # Never record prompts, pairing tokens, paths or user IPs.
        pass

    def _send(self, code: int, payload: str | bytes, mime="text/plain; charset=utf-8",
              cookie: str | None = None):
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; connect-src 'self'; "
                         "img-src 'self'; form-action 'self'; base-uri 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Strict-Transport-Security", "max-age=86400")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _json(self, status, data):
        self._send(status, json.dumps(data, ensure_ascii=False),
                   "application/json; charset=utf-8")

    def _host_ok(self):
        return self.headers.get("Host") == f"{self.server.config['host']}:{self.server.config['port']}"

    def _same_origin(self):
        return (self.headers.get("Origin") == self.server.origin
                and self.headers.get("Sec-Fetch-Site", "same-origin") in ("same-origin", "none"))

    def _token(self):
        jar = cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
            cookie = jar.get("neral_mobile")
            if cookie:
                return cookie.value
        except (cookies.CookieError, ValueError):
            pass
        return ""

    def _valid(self, token: str) -> bool:
        return self.server.valid_session(token)

    def _request_body(self):
        if self.headers.get("Transfer-Encoding"):
            raise MiniError("Chunked HTTP data forbidden")
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except (TypeError, ValueError):
            raise MiniError("Invalid request length") from None
        if not 1 <= length <= MAX_BODY:
            raise MiniError("Invalid request size")
        return self.rfile.read(length)

    def do_GET(self):
        if not self._host_ok() or self.headers.get("Sec-Fetch-Site") == "cross-site":
            return self._json(403, {"error":"Kein lokaler WLAN-Zugriff."})
        if self.path not in ("/", "/api/status"):
            return self._json(404, {"error":"Unbekannter Pfad."})
        token = self._token()
        if not self._valid(token):
            if self.path != "/":
                return self._json(401, {"error":"Erst mit dem PC koppeln."})
            return self._send(200, (
                '<!doctype html><html lang="de"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<meta name="color-scheme" content="dark"><title>NERAL Mini verbinden</title>'
                '<style>body{background:#151719;color:#edf2ee;font:16px system-ui;'
                'max-width:440px;margin:15vh auto;padding:24px;line-height:1.55}'
                'input,button{font:inherit;border-radius:8px;padding:11px;width:100%;'
                'margin:8px 0}button{background:#cadacc;border:0;color:#1a211d}'
                'input{background:#272c2c;color:white;border:1px solid #515959}</style>'
                '<h1>NERAL Mini</h1><p>Verbinde dein Handy sicher mit deinem PC.</p>'
                '<p>Kopplungscode aus der Datei <b>HANDY-ZUGANG.txt</b> auf deinem PC:</p>'
                '<form method="post" action="/pair"><label for="pair">Kopplungscode</label>'
                '<input id="pair" name="code" type="password" required '
                'autocomplete="off" maxlength="40">'
                '<button type="submit">Verbinden</button></form>'
                '<p>Die KI läuft nur auf deinem eigenen PC im Heim-WLAN.</p>'
                '</html>'), "text/html; charset=utf-8")
        if self.path == "/api/status":
            return self._json(200, self.server.desktop.backend.status())
        # Never expose the loopback server's master token to the phone.
        html = self.server.desktop.html.replace("%%NERAL_SESSION%%", token)
        return self._send(200, html, "text/html; charset=utf-8")

    def do_POST(self):
        if not self._host_ok():
            return self._json(403, {"error":"Unbekannter Host."})
        if not self._same_origin():
            return self._json(403, {"error":"Fremde Webseite blockiert."})
        if self.path not in ("/pair", "/api/chat", "/logout"):
            return self._json(404, {"error":"Unbekannter Pfad."})
        if self.path == "/pair":
            if self.headers.get("Content-Type","").split(";")[0].strip().lower() != "application/x-www-form-urlencoded":
                return self._json(415, {"error":"Ungueltiger Kopplungs-Request."})
            try:
                raw = self._request_body()
                pairs = parse_qs(raw.decode("utf-8"), keep_blank_values=True, max_num_fields=3)
                if set(pairs) != {"code"} or len(pairs["code"]) != 1:
                    raise MiniError("Ungültige Kopplungsparameter")
                code = pairs["code"][0]
                if not re.fullmatch(r"[0-9a-f]{16,40}", code):
                    # Rate limit even incorrect-format pairing via a dummy invalid code.
                    code = ""
                token = self.server.pair(self.client_address[0], code)
            except MiniError as e:
                return self._json(429 if str(e) in ("rate-limit","too-many-sessions") else 400,
                                  {"error":"Kopplung derzeit nicht möglich."})
            except (UnicodeError, ValueError):
                return self._json(400, {"error":"Ungültige Kopplungsanfrage."})
            if token is None:
                return self._json(403, {"error":"Kopplungscode stimmt nicht."})
            return self._send(
                200, '<!doctype html><html lang="de"><meta name="viewport" content="width=device-width">'
                '<meta name="color-scheme" content="dark"><title>NERAL verbunden</title>'
                '<body style="font:16px system-ui;background:#151719;color:#e6eee6;padding:30px">'
                '<h1>NERAL verbunden</h1><p>Dein PC ist gekoppelt.</p>'
                '<a style="color:#a9d6bb" href="/">NERAL Mini öffnen</a></body></html>',
                "text/html; charset=utf-8",
                "neral_mobile="+token+"; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=28800")
        token = self._token()
        if not self._valid(token):
            return self._json(401, {"error":"Kopplung abgelaufen."})
        supplied = self.headers.get("X-Neral-Session", "")
        if not supplied or not secrets.compare_digest(token, supplied):
            return self._json(403, {"error":"Chat-Sitzung fehlt."})
        if self.path == "/logout":
            with self.server.lock:
                self.server.mobile_sessions.pop(token, None)
            return self._send(200, "Abgemeldet.", cookie="neral_mobile=; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=0")
        if self.headers.get("Content-Type","").split(";")[0].strip().lower() != "application/json":
            return self._json(415, {"error":"Nur JSON-Chat erlaubt."})
        try:
            messages = parse_chat(json.loads(self._request_body()))
        except (UnicodeError, MiniError, TypeError, ValueError):
            return self._json(400, {"error":"Ungültige oder zu lange Nachricht."})
        # Share GPU lock with local desktop Mini; never load two models.
        if not self.server.desktop.busy.acquire(blocking=False):
            return self._json(429, {"error":"NERAL beantwortet bereits eine Frage."})
        try:
            try:
                started = time.monotonic()
                response = self.server.desktop.backend.respond(messages)
                if not isinstance(response, str) or not response.strip():
                    raise MiniError("Leere Modellantwort")
                return self._json(200, {
                    "text":response,
                    "seconds":round(time.monotonic()-started,2),
                    **self.server.desktop.backend.status()
                })
            except Exception:
                return self._json(503, {"error":"Die lokale Modellantwort ist fehlgeschlagen. PC und GPU prüfen."})
        finally:
            self.server.desktop.busy.release()


def create_mobile_gateway(config: dict, desktop: MiniServer, *,
                          bind_host: str | None = None, bind_port: int | None = None) -> MobileGateway:
    if not isinstance(config, dict):
        raise ValueError("Invalid mobile config")
    host = config.get("host")
    port = config.get("port")
    pair_hash = config.get("pair_hash")
    if not isinstance(host,str) or not _private_ipv4(host):
        raise ValueError("Mobile gateway requires a concrete private WLAN IPv4")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Invalid mobile gateway port")
    if not isinstance(pair_hash,str) or not re.fullmatch(r"[0-9a-f]{64}",pair_hash):
        raise ValueError("Missing valid secret pairing code digest")
    if not isinstance(desktop,MiniServer):
        raise ValueError("Only local NERAL Mini server instances accepted")
    if bind_host is not None and bind_host != "127.0.0.1":
        raise ValueError("Test host override only permits loopback")
    if bind_port is not None and bind_port != 0:
        raise ValueError("Test port override only permits ephemeral port")
    return MobileGateway((bind_host or host, bind_port if bind_port is not None else port),config,desktop)


def _setup_config(host: str, port: int, out: Path) -> dict:
    from setup_phone import setup
    if not (out/"config.json").exists():
        return setup(host,out,port=port)
    config = json.loads((out/"config.json").read_text(encoding="utf-8"))
    if config.get("host") != host or config.get("port") != port:
        raise ValueError("IP oder Port geändert. Alten Mini-Handy-Zugang sichern und neue Kopplung einrichten.")
    return config


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="NERAL Mini – private HTTPS mobile companion")
    parser.add_argument("--host",default=None)
    parser.add_argument("--port",type=int,default=9444)
    parser.add_argument("--mini-port",type=int,default=9691)
    args=parser.parse_args(argv)
    host=args.host or detect_host()
    if not _private_ipv4(host):
        raise ValueError("Only concrete private IPv4 LAN addresses are supported")
    offline()
    root=Path(__file__).resolve().parents[1]
    data_dir=root/"local_state"/"mini_mobile"
    config=_setup_config(host,args.port,data_dir)
    cert=Path(config["certificate"])
    key=Path(config["key"])
    for file in (cert,key):
        if file.is_symlink() or not file.is_file():
            raise ValueError("TLS private key/certificate missing or linked")
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(cert),str(key))
    home=Path.home()
    base=Path(os.environ.get("NERAL_MINI_BASE_PATH") or
        str(home/".cache"/"neral"/"qwen25-smoke"/BASE_REVISION/
            "installed_models"/"hf-qwen25-coder-7b"))
    adapter=Path(os.environ.get("NERAL_MINI_ADAPTER_PATH") or
        str(home/"NERAL-QLoRA-Research"/"actions-r32-42078722575-1"/"adapter"))
    desktop=create_server(NeralModel(base,adapter),port=args.mini_port)
    gateway=None
    try:
        gateway=create_mobile_gateway(config,desktop)
        gateway.socket=context.wrap_socket(gateway.socket,server_side=True)
        worker=threading.Thread(target=desktop.serve_forever,daemon=True)
        worker.start()
        print("NERAL Mini Mobile gestartet: https://"+host+":"+str(args.port)+"/",flush=True)
        print("Die private Adresse und den Code findest du in: "+str(data_dir/"HANDY-ZUGANG.txt"),flush=True)
        print("Nur WLAN, verschlüsseltes HTTPS, keine Cloud-KI, keine Router-Freigabe.",flush=True)
        print("Beenden mit Strg+C.",flush=True)
        if os.name=="nt":
            try: os.startfile(str(data_dir/"HANDY-ZUGANG.txt"))
            except OSError: pass
        try: gateway.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt: pass
    finally:
        if gateway is not None:
            gateway.server_close()
        desktop.shutdown()
        desktop.server_close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
