"""Manually opt-in two-model OLD-CASE NL-schema probe for local Ollama only.

Does not touch sealed prospective-100 data, runtime routing, or main. No
default network activity; --live and two distinct installed model tags are
required. Real model claims require inspecting the returned status/CI.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

from neral_nl_schema_bridge import assess_dual_extraction
from scripts.neral_blind_100_cases import build as historical_old100

_OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
_MODEL_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,127}\Z", re.ASCII)
_CODE_21453 = (
    "Ein fünfstelliger Code verwendet 1, 2, 3, 4 und 5 genau einmal. "
    "Die 2 steht vor der 4, nicht direkt davor. Zwischen der 1 und 5 "
    "steht genau eine Ziffer. Die 3 steht unmittelbar rechts von der 5. "
    "Die 4 steht nicht am Rand. Bestimme alle gültigen Codes."
)


def old_case_question(case: str) -> str:
    """Only historical public cases; not the second frozen blind100 corpus."""
    if case == "21453":
        return _CODE_21453
    if case not in ("logic-02", "logic-06"):
        raise ValueError("Unsupported non-historical probe case")
    rows = {row["id"]: row for row in historical_old100()}
    return rows[f"blind100-{case}"]["question"]


def _local_transport(payload: dict) -> dict:
    req = urllib.request.Request(
        _OLLAMA_CHAT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Explicit local-only endpoint; never relay questions to a remote API.
    with urllib.request.urlopen(req, timeout=90) as response:
        data = response.read(262145)
    if len(data) > 262144:
        raise ValueError("Ollama response too large")
    parsed = json.loads(data)
    if type(parsed) is not dict:
        raise ValueError("Invalid local model response")
    return parsed


def make_local_extractor(model_tag: str, *, transport=None):
    """Wrap an explicitly installed local Ollama model without schema repair."""
    if type(model_tag) is not str or not _MODEL_TAG.fullmatch(model_tag):
        raise ValueError("Invalid local Ollama model identifier")
    if transport is not None and not callable(transport):
        raise ValueError("Invalid local test transport")
    send = transport if transport is not None else _local_transport

    def extract(messages: list[dict]) -> str:
        if (type(messages) is not list or len(messages) != 2 or
                any(type(row) is not dict or set(row) != {"role", "content"}
                    or type(row["content"]) is not str for row in messages)):
            raise ValueError("Invalid local extraction turns")
        payload = {
            "model": model_tag,
            "messages": [dict(row) for row in messages],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_predict": 768},
        }
        response = send(payload)
        if (type(response) is not dict or type(response.get("message")) is not dict
                or type(response["message"].get("content")) is not str):
            raise ValueError("Invalid local model JSON response envelope")
        if response.get("done_reason") == "length":
            raise ValueError("Truncated local model JSON")
        return response["message"]["content"]

    return extract


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NERAL OLD logic cases; local-only dual extraction (not NLP proof)")
    parser.add_argument("--case", required=True, choices=("logic-02", "logic-06", "21453"))
    parser.add_argument("--live", action="store_true", help="explicitly run two locally installed Ollama models")
    parser.add_argument("--qlora-model", help="local Ollama tag for QLoRA variant; no model downloaded")
    parser.add_argument("--base-model", help="local Ollama tag for unchanged base model")
    parser.add_argument("--diagnostics-file", help="create a LOCAL JSON report with unredacted old-case model responses and timing; requires --live")
    args = parser.parse_args(argv)
    if args.diagnostics_file and not args.live:
        parser.error("--diagnostics-file requires --live")
    if not args.live:
        print(json.dumps({"status": "live_not_requested", "case": args.case,
                          "verified": False, "safe_to_promote": False}))
        return 0
    if (not args.qlora_model or not args.base_model
            or args.qlora_model == args.base_model):
        parser.error("--live requires two explicitly named, different local model tags")
    if args.diagnostics_file:
        # Explicit opt-in file stored locally only; never overwrite a
        # previous measurement or ship raw generated text to GitHub.
        from pathlib import Path
        from scripts.neral_nl_schema_diagnostics import collect_live_diagnostics
        output_path = Path(args.diagnostics_file)
        if output_path.exists():
            raise FileExistsError("Diagnostic file exists; choose a new path")
        report = collect_live_diagnostics(
            args.case, args.qlora_model, args.base_model)
        with output_path.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
        print(json.dumps({
            "case": args.case,
            "diagnostics_file": str(output_path),
            "consensus_status": report["consensus_status"],
            "verified": False, "safe_to_promote": False,
        }, ensure_ascii=False))
        return 0 if report["consensus_status"] == "consensus_unverified" else 1
    try:
        a = make_local_extractor(args.qlora_model)
        b = make_local_extractor(args.base_model)
        result = assess_dual_extraction(old_case_question(args.case), a, b)
    except (ValueError, OSError):
        result = {"status": "local_setup_error", "candidate_count": None,
                  "proof_backend": None, "verified": False, "safe_to_promote": False}
    # No raw user text or generated schemas in output, and never report
    # consensus as a semantically verified answer.
    print(json.dumps({"case": args.case, "live_invocation_attempted": True,
                      "status": result["status"],
                      "candidate_count": result.get("candidate_count"),
                      "proof_backend": result.get("proof_backend"),
                      "verified": False, "safe_to_promote": False}))
    return 0 if result["status"] == "consensus_unverified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
