"""Neral Scale V4 CI bootstrap.

Only the self-hosted GitHub Actions Scale V4 checkout uses this hook. It applies the
source-only recovery patch before the workflow's existing py_compile gate, so syntax
validation still happens before training starts. Local Python sessions are untouched.
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path


def _scale_v4_ci_checkout() -> bool:
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return False
    root = Path.cwd()
    request = root / "frontier350_scale_v4_requests" / "request.json"
    patcher = root / "frontier350" / "apply_scale_v4_recovery_patch.py"
    if not request.is_file() or not patcher.is_file():
        return False
    try:
        data = json.loads(request.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return data.get("version") == 4 and data.get("action") == "run_scale_v4"


if _scale_v4_ci_checkout():
    try:
        runpy.run_path(
            str(Path.cwd() / "frontier350" / "apply_scale_v4_recovery_patch.py"),
            run_name="__main__",
        )
    except Exception as exc:
        print(f"Scale V4 CI recovery bootstrap failed: {exc}", flush=True)
        os._exit(86)
