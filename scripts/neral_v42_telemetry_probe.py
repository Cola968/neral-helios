"""Exactly three public synthetic diagnostic prompts per tier, never Blind200.

No private sealed suite is read, no answer text stored or logged, and no model
weights, training data or personal context is captured. Only numeric telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit

from scripts.neral_v41_blind200_live import OwnerClient, _exact_text
from scripts.neral_v42_resumable import wait_until_available
from scripts.neral_v42_gpu_memory import read_vram_snapshot, validate_owner_snapshot

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


def grade_live_probe(category, http_code, response):
    """Tightly scoped synthetic diagnostic grader; no benchmark inputs or logs."""
    if http_code != 200 or not isinstance(response, dict):
        return "failed_http"
    text = response.get("text")
    if not isinstance(text, str) or not text.strip() or response.get("error_code") is not None:
        return "failed_response"
    if category == "math_logic":
        return ("exact_numeric_match" if _exact_text(text, "integer") == "345"
                else "not_exact_number")
    if category == "knowledge":
        # The server first checks the literal citation against an actually
        # read HTTPS page; the client then independently checks the answer
        # and the presence of an HTTPS read entry. No quote is recorded.
        body = re.split(r"\n\s*\n\[web[1-9]\d*\]", text, maxsplit=1)[0]
        years = set(re.findall(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)", body))
        sources = response.get("sources")
        read_https = bool(isinstance(sources, list) and any(
            isinstance(item, dict) and item.get("page_read") is True
            and isinstance(item.get("url"), str)
            and urlsplit(item["url"]).scheme == "https"
            and bool(urlsplit(item["url"]).hostname)
            for item in sources))
        return ("exact_year_and_read_https_quote" if (
            years == {"1969"}
            and response.get("source_year_match") is True
            and response.get("year_quote_in_final") is True
            and response.get("year_source_pipeline_code") == "candidate_available"
            and read_https
            and "„" in text and "“" in text and "\n[web" in text
        ) else "failed_year_or_quote")
    if category == "everyday":
        sentences = [v.strip() for v in re.split(r"(?<=[.!?])\s+", text.strip()) if v.strip()]
        physical = bool(
            re.search(r"(?iu)(?:wasserdampf|dampf|feucht|warm\w*\s+luft)",text)
            and re.search(r"(?iu)(?:kalt|kühl|temperatur|abkühl|kälter)",text)
            and re.search(r"(?iu)(?:kondens|tröpf|beschlag)",text))
        return ("two_sentence_condensation_explanation"
                if len(sentences) == 2 and physical else "failed_everyday_explanation")
    return "not_a_diagnostic_case"


def _owner_gpu_preflight():
    """Read-only Windows identity preflight; PIDs never enter artifacts/logs."""
    if os.name != "nt":
        raise RuntimeError("Owner GPU preflight requires Windows")
    script = Path(__file__).with_name("neral_v42_gpu_owner_preflight.ps1")
    try:
        outcome = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-File", str(script)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=False, timeout=25)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Read-only GPU owner preflight unavailable") from exc
    if outcome.returncode != 0:
        raise RuntimeError("Read-only GPU owner/foreign-process preflight refused")
    try:
        row = json.loads(outcome.stdout)
        ids = row["owner_pids"]
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 3
                or any(type(pid) is not int or pid <= 0 for pid in ids)
                or len(set(ids)) != len(ids)
                or row.get("foreign_python_gpu_process_count") != 0
                or row.get("foreign_heavy_gpu_process_count") != 0
                or type(row.get("system_gpu_preflight_mib")) is not int):
            raise ValueError("Unrecognized owner summary")
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("GPU owner preflight provided no verifiable identity") from exc
    # Only safe integers and booleans are returned to the artifact formatter.
    summary = {
        "foreign_python_gpu_process_count":0,
        "foreign_heavy_gpu_process_count":0,
        "owner_gpu_pid_count": row.get("owner_gpu_pid_count"),
        "system_gpu_preflight_mib":row["system_gpu_preflight_mib"],
    }
    return tuple(ids), summary


def _gpu_poll_until(done, snapshots, owner_pids):
    while not done.is_set():
        snapshots.append(read_vram_snapshot(owner_pids))
        done.wait(0.7)


def _peak(snapshots, name):
    values = [row[name] for row in snapshots if type(row.get(name)) is int]
    return max(values) if values else None


def _vram_gate(snapshots):
    # A single missing WDDM process-value makes the gate unproven.
    return bool(snapshots) and all(
        validate_owner_snapshot(row, process_cap_mib=9500)
        for row in snapshots)



def run(tier, summary_path):
    if tier not in TIERS:
        raise ValueError("Only Quartz and Obsidian are authorized")
    client = OwnerClient()
    client.connect()
    owner_pids, owner_preflight = _owner_gpu_preflight()
    rows = []
    for category, prompt in DIAGNOSTIC_CASES:
        if not wait_until_available(client, max_wait_seconds=90):
            raise RuntimeError("Owner GPU was not released before next diagnostic task")
        done = threading.Event()
        snapshots = []
        sampler = threading.Thread(
            target=_gpu_poll_until, args=(done, snapshots, owner_pids), daemon=True)
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
        quality = grade_live_probe(category, code, result)
        row = {
            "tier":tier, "category":category, "http_code":code,
            "latency_ms":round(total*1000),
            "error_code":err, "response_nonempty":bool(str(result.get("text","")).strip()),
            "numeric_math_probe": quality if category == "math_logic" else "not_machine_graded",
            "final_answer_check": quality,
            "source_year_match": result.get("source_year_match") is True,
            "year_quote_in_final": result.get("year_quote_in_final") is True,
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
            "math_stdout_diagnostic_code": (
                result.get("math_stdout_diagnostic_code")
                if result.get("math_stdout_diagnostic_code") in (
                    "accepted", "execution_failed", "not_stock_task",
                    "empty_stdout", "wrong_value", "extra_lines", "unsupported_shape")
                else "not_available"),
            "math_sandbox_attempts": (
                result.get("sandbox_attempts")
                if type(result.get("sandbox_attempts")) is int
                and 0 <= result["sandbox_attempts"] <= 3 else None),
            "year_source_pipeline_code": (
                result.get("year_source_pipeline_code")
                if result.get("year_source_pipeline_code") in (
                    "no_read_page", "year_lost_in_clip", "year_missing_in_extract",
                    "year_event_not_contiguous", "quote_missing_from_evidence",
                    "candidate_available")
                else "not_applicable" if category != "knowledge" else "missing"),
            "vram_peak_mib": _peak(snapshots, "system_total_mib"),
            "vram_total_system_peak_mib": _peak(snapshots, "system_total_mib"),
            "vram_neral_process_peak_mib": _peak(snapshots, "neral_process_mib"),
            "vram_neral_process_under_9500": _vram_gate(snapshots),
            "vram_owner_process_memory_status": (
                "measured" if _vram_gate(snapshots)
                else "unmeasured_or_over_limit"),
            "gpu_preflight": owner_preflight,
            "telemetry":diagnostics,
        }
        rows.append(row)
        # Safe summary only. The prompt and answer are NEVER emitted.
        print("NERAL_DIAG3_ROW "+json.dumps(row,sort_keys=True,separators=(",",":")),flush=True)
    if not wait_until_available(client, max_wait_seconds=90):
        raise RuntimeError("Owner GPU lock retained after three diagnostic requests")
    expected = {
        "everyday": "two_sentence_condensation_explanation",
        "math_logic": "exact_numeric_match",
        "knowledge": "exact_year_and_read_https_quote",
    }
    functional_pass = all(
        row["http_code"] == 200 and row["error_code"] == "none"
        and row["final_answer_check"] == expected[row["category"]]
        and row["vram_neral_process_under_9500"] is True
        for row in rows)
    result={"source":"synthetic_public_questions_no_blind200",
            "tier":tier,"rows":rows,"n":len(rows),
            "http_409":sum(row["http_code"]==409 for row in rows),
            "functional_pass_count":sum(row["final_answer_check"] == expected[row["category"]]
                                        and row["http_code"] == 200
                                        for row in rows),
            "owner_vram_gate_pass":all(row["vram_neral_process_under_9500"]
                                       for row in rows),
            "functional_and_owner_vram_pass":functional_pass}
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"NERAL_DIAG3_COMPLETE tier={tier} n={len(rows)} HTTP409={result['http_409']} "
          f"functional={result['functional_pass_count']} process_vram_gate={result['owner_vram_gate_pass']}",
          flush=True)
    if not functional_pass:
        raise RuntimeError("Final diagnostic proof or process VRAM gate not accepted")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tier",choices=TIERS,required=True)
    p.add_argument("--summary",type=Path,required=True)
    args=p.parse_args()
    run(args.tier,args.summary)


if __name__=="__main__":
    main()
