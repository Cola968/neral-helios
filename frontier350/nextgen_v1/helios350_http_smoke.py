"""Bounded read-only end-to-end HTTP smoke: own SAFE weights, actual localhost server.

No training, data export, persistent service or checkpoint modification. The local
server is shut down in finally and generated text is never uploaded or logged.
"""
from __future__ import annotations

import argparse
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading

from helios350_preview import SAFE_RELATIVE, load_verified_model, make_handler


def request(port: int, method: str, endpoint: str, *, payload: bytes | None = None,
            origin: str | None = None) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=60)
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    if origin is not None:
        headers["Origin"] = origin
    try:
        conn.request(method, endpoint, body=payload, headers=headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", required=True)
    args = ap.parse_args()
    root = Path(args.assets_root)
    model, tok, bos, eos, processed, torch = load_verified_model(
        root, root / SAFE_RELATIVE, "safe", "cpu"
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(model, tok, bos, eos, torch, processed, "safe", "cpu", 0),
    )
    port = server.server_address[1]
    # make_handler must know the actual ephemeral port to reject foreign Host/Origin.
    server.server_close()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        make_handler(model, tok, bos, eos, torch, processed, "safe", "cpu", port),
    )
    worker = threading.Thread(target=server.serve_forever, name="helios-http-smoke", daemon=True)
    worker.start()
    try:
        status, health = request(port, "GET", "/health")
        if status != 200 or health.get("processed_tokens") != 1_010_003_584 or health.get("device") != "cpu":
            raise RuntimeError("actual own model health check failed")
        status, _ = request(port, "GET", "/health", origin="https://untrusted.invalid")
        if status != 403:
            raise RuntimeError("cross-origin request was not rejected")
        payload = json.dumps({"prompt": "Hallo", "max_new_tokens": 1}).encode("utf-8")
        status, answer = request(port, "POST", "/generate", payload=payload,
                                 origin=f"http://127.0.0.1:{port}")
        if status != 200 or answer.get("model") != "safe" or type(answer.get("answer")) is not str:
            raise RuntimeError("real model HTTP generation failed")
        if type(answer.get("generated_tokens")) is not int or not 0 <= answer["generated_tokens"] <= 1:
            raise RuntimeError("unbounded or missing generated-token counter")
        print(json.dumps({"status": "own_350m_http_generation_passed",
                          "checkpoint_tokens": processed, "health_http": 200,
                          "forbidden_origin_http": 403, "generation_http": 200,
                          "generated_tokens": answer["generated_tokens"], "training_started": False,
                          "persistent_server_started": False}), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
