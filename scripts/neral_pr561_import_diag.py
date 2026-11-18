"""Read-only diagnosis for NERAL installed source imports on user's Windows runner.

Never creates an account, touches a service, executes generated code or prints
model paths, personal files, source code, credentials or arbitrary stderr.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import traceback

NAMES = (
    "neral_preview_server", "neral_three_tier_runtime",
    "neral_everyday_verified", "neral_web_search",
    "neral_adaptive_planner", "neral_tir_experimental",
)

def main() -> int:
    for name in NAMES:
        outcome = {"module": name, "ok": False}
        try:
            importlib.import_module(name)
            outcome["ok"] = True
        except BaseException as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            last = frames[-1] if frames else None
            outcome.update({
                "exception": type(exc).__name__,
                "missing_module": getattr(exc, "name", None),
                "file": Path(last.filename).name if last else None,
                "line": int(last.lineno) if last else None,
            })
        print("NERAL_SAFE_IMPORT " + json.dumps(outcome, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
