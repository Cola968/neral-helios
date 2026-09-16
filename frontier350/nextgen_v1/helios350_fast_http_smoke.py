"""Actual SAFE model and HTTP smoke for the optional KV-cached preview.

No background service is left running and generated text is not logged.
"""
from __future__ import annotations

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import threading
import time

import helios350_preview as baseline
original_generate = baseline.generate
import helios350_fast_preview as fast  # Explicitly installs KV-only adapter.
from helios350_http_smoke import request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", type=Path, required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    root = args.assets_root
    model, tok, bos, eos, processed, loaded_torch = baseline.load_verified_model(
        root, root / baseline.SAFE_RELATIVE, "safe", "cpu"
    )
    if processed != 1_010_003_584 or loaded_torch is not torch or baseline.generate is not fast.cached_generate:
        raise AssertionError("wrong model or generation implementation")
    original = original_generate(model, tok, bos, eos, torch, "Hallo", 4)
    # One actual localhost server, with the same verified Host/Origin handler.
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        baseline.make_handler(model, tok, bos, eos, torch, processed, "safe", "cpu", 0),
    )
    port = server.server_address[1]
    server.server_close()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        baseline.make_handler(model, tok, bos, eos, torch, processed, "safe", "cpu", port),
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=60)
        try:
            conn.request("GET", "/")
            response = conn.getresponse()
            if response.status != 200 or b"Helios" not in response.read():
                raise RuntimeError("HTML preview unavailable")
        finally:
            conn.close()
        status, health = request(port, "GET", "/health")
        if status != 200 or health.get("processed_tokens") != processed:
            raise RuntimeError("real model status failed")
        forbidden, _ = request(port, "GET", "/health", origin="https://untrusted.invalid")
        if forbidden != 403:
            raise RuntimeError("foreign origin allowed")
        payload = json.dumps({"prompt": "Hallo", "max_new_tokens": 4}).encode("utf-8")
        start = time.perf_counter()
        code, response = request(port, "POST", "/generate", payload=payload,
                                 origin=f"http://127.0.0.1:{port}")
        elapsed = time.perf_counter() - start
        if code != 200 or (response.get("answer"), response.get("generated_tokens")) != original:
            raise RuntimeError("cached HTTP generation differs from original greedy output")
        print(json.dumps({"status": "real_SAFE_fast_HTTP_four_token_parity_passed",
                          "checkpoint_tokens": processed, "html": 200,
                          "health": 200, "foreign_origin": 403, "generation": code,
                          "generated_tokens": response["generated_tokens"],
                          "http_seconds_diagnostic": round(elapsed, 4),
                          "gpu_used": False, "training_started": False,
                          "checkpoint_modified": False,
                          "persistent_server_started": False}), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
