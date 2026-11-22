"""Exactly three public synthetic diagnostic prompts per tier, never Blind200.

No private sealed suite is read, no answer text stored or logged, and no model
weights, training data or personal context is captured. Only numeric telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

from scripts.neral_v41_blind200_live import OwnerClient, _exact_text
from scripts.neral_v42_resumable import sample_gpu_mb, wait_until_available

DIAGNOSTIC_CASES = (
    ("everyday", "Warum beschlägt ein Badezimmerspiegel nach dem Duschen? Erkläre es in zwei verständlichen Sätzen."),
    ("math_logic", "Ein Lager erhält zuerst 17 Kisten mit jeweils 23 Bauteilen und gibt danach 46 Bauteile ab. Berechne zuerst die Gesamtlieferung und danach den Rest. Zeige den Rechenweg und schreibe als letzte Zeile exakt ENDERGEBNIS: <ganze Zahl>. Danach keinen weiteren Text."),
    ("knowledge", "In welchem Jahr landete Apollo 11 erstmals Menschen auf dem Mond? Nenne die Jahreszahl und belege sie mit einem wörtlichen Zitat aus einer tatsächlich gelesenen Quelle."),
)
TIERS = ("Quartz", "Obsidian")
_ALLOWED_ERRORS = {
    "request_deadline", "runtime_unavailable", "hf_time_budget",
    "hf_token_budget", "hf_invalid_final", "hf_reasoning_unclosed",
    "checker_veto", "no_tool_called", "z3_schema_error", "sandbox_error",
    "format_mismatch", "context_exceeded", "gpu_memory", "server_busy",
}


def web_query_for_diagnostic(category, prompt):
    # Only this public synthetic knowledge question is explicitly approved for
    # fresh Web V4 lookup. No private Blind200 prompts ever leave the owner PC.
    return "Apollo 11 erste bemannte Mondlandung Jahr NASA" if category == "knowledge" and prompt == DIAGNOSTIC_CASES[2][1] else None


def _gpu_poll_until(done, peaks):
    """Only GPU used-MiB integers; no process list or command lines."""
    while not done.is_set():
        v = sample_gpu_mb()
        if isinstance(v, int):
            peaks.append(v)
        done.wait(0.7)


def run(tier, summary_path):
    if tier not in TIERS:
        raise ValueError("Only Quartz and Obsidian are authorized")
    client = OwnerClient()
    client.connect()
    rows = []
    for category, prompt in DIAGNOSTIC_CASES:
        if not wait_until_available(client, max_wait_seconds=90):
            raise RuntimeError("Owner GPU was not released before next diagnostic task")
        done = threading.Event()
        peaks = []
        sampler = threading.Thread(target=_gpu_poll_until, args=(done, peaks), daemon=True)
        sampler.start()
        try:
            code, result, total = client.post(
                tier, prompt, [], diagnostic_telemetry=True,
                web_query=web_query_for_diagnostic(category, prompt))
        finally:
            done.set()
            sampler.join(timeout=5)
        if type(result) is not dict:
            raise RuntimeError("Malformed diagnostic payload")
        diagnostics = result.get("telemetry")
        if not isinstance(diagnostics, dict):
            raise RuntimeError("Owner server did not return numeric request telemetry")
        err = result.get("error_code")
        err = err if err in _ALLOWED_ERRORS else "none" if err is None else "other"
        quality = ("exact_numeric_match" if _exact_text(
            result.get("text", ""), "integer") == "345" else "not_exact_number"
            ) if category == "math_logic" and code == 200 else "not_machine_graded"
        row = {
            "tier":tier, "category":category, "http_code":code,
            "latency_ms":round(total*1000),
            "error_code":err, "response_nonempty":bool(str(result.get("text","")).strip()),
            "numeric_math_probe":quality,
            "checker_veto_reason_code": (
                result.get("checker_veto_reason_code")
                if result.get("checker_veto_reason_code") in (
                    "generic_quote_veto","contradiction_reported",
                    "source_or_quote_issue","tool_issue",
                    "no_external_evidence","tool_not_executed","other_checker_issue",
                    "no_read_page","invalid_quote","unsupported_year","critic_contradiction")
                else "not_available"),
            "checker_veto_detail_code": (
                result.get("checker_veto_detail_code")
                if result.get("checker_veto_detail_code") in (
                    "unsupported_year_answer", "unsupported_year_quote_conflict",
                    "unsupported_year_missing_event", "unsupported_year_no_passage")
                else "not_available"),
            "vram_peak_mib":max(peaks) if peaks else None,
            "telemetry":diagnostics,
        }
        rows.append(row)
        # Safe summary only. The prompt and answer are NEVER emitted.
        print("NERAL_DIAG3_ROW "+json.dumps(row,sort_keys=True,separators=(",",":")),flush=True)
    if not wait_until_available(client, max_wait_seconds=90):
        raise RuntimeError("Owner GPU lock retained after three diagnostic requests")
    result={"source":"synthetic_public_questions_no_blind200",
            "tier":tier,"rows":rows,"n":len(rows),
            "http_409":sum(row["http_code"]==409 for row in rows)}
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"NERAL_DIAG3_COMPLETE tier={tier} n={len(rows)} HTTP409={result['http_409']}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tier",choices=TIERS,required=True)
    p.add_argument("--summary",type=Path,required=True)
    args=p.parse_args()
    run(args.tier,args.summary)


if __name__=="__main__":
    main()
