"""Opt-in ephemeral integration demo of the two existing DRAFT branches.

Default is no execution. Both explicit consents are required to run exactly
one local Qwen plan, one approved public Wikipedia lookup, one local Qwen
draft and one temporary SQLite audit record. No state or model text uploaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from neral_agent_plan import propose_agent_plan
from neral_orbit_agent_loop import run_confirmed_research, draft_with_qwen
from neral_qwen_plan_probe import MODEL, installed_models


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orbit-source-dir",type=Path)
    parser.add_argument("--confirm-model",action="store_true")
    parser.add_argument("--confirm-public",action="store_true")
    a=parser.parse_args(argv)
    if not a.confirm_model and not a.confirm_public and a.orbit_source_dir is None:
        print(json.dumps({"status":"NO_EXECUTION","tools_executed":False}))
        return 0
    if not a.confirm_model or not a.confirm_public or a.orbit_source_dir is None:
        parser.error("Two separate confirmations and reviewed Orbit source are mandatory")
    src=a.orbit_source_dir.resolve()
    if (not (src/"neral_orbit.py").is_file() or
        not (src/"neral_orbit_research.py").is_file()):
        raise ValueError("Reviewed Orbit dependency source unavailable")
    if MODEL not in installed_models():
        print(json.dumps({"status":"MODEL_UNAVAILABLE","execution_enabled":False}))
        return 0

    sys.path.insert(0,str(src))
    from neral_orbit import OrbitStore
    from neral_orbit_research import run_public_research

    root=Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    # Temporary local evidence exists only during this expressly authorized
    # execution. No upload-artifact or GitHub API write.
    with tempfile.TemporaryDirectory(prefix="neral-orbit-e2e-",dir=root) as directory:
        store=OrbitStore(Path(directory)/"orbit.sqlite3")
        result=run_confirmed_research(
            "Flughafen Berlin Brandenburg",confirm_model=True,
            confirm_public=True,plan=propose_agent_plan,
            research=run_public_research,draft=draft_with_qwen,store=store)
        if result.get("status")!="REVIEW_READY":
            raise RuntimeError("Unreviewed research was not persisted")
        task=store.list()
        if len(task)!=1 or task[0].get("report") is None:
            raise RuntimeError("No durable local source report in one-shot task")
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
