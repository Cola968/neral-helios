"""Real local RTX/Ollama 24-case before/after structured-holdout evaluator.

Only the scalar result is exported. No model text, user prompts, holdout
answers, private thoughts, or generated program code are uploaded.
No training, no external API, no host-side execution of generated code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import time

from neral_formal_constraints import PREFIX
from neral_tir_experimental import run_experimental_tir
from neral_turn0_extractor import (
    EXTRACT_PROMPT, extraction_response_format, parse_extracted_constraints,
)
from scripts.neral_qlora_holdout_eval import load_holdout
from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS

BASE_MODEL = "qwen3.5:9b"
BASE_SEED = 202611120536
AUTH = "NERAL_QLORA_HOLDOUT_GPU_MANUAL_ONLY"
EXPECTED_BASE_SHA = "d280516e1a28a1cbdbc5b6b9b89d694769f6292f"
MAX_RESPONSE_BYTES = 128 * 1024
MAX_MODEL_OUTPUT_BYTES = 4096


def request(method: str, path: str, packet: dict | None = None,
            timeout: float = 100.0) -> dict:
    if method not in ("GET", "POST") or path not in (
        "/api/tags", "/api/ps", "/api/chat"
    ):
        raise ValueError("Only trusted local Ollama endpoints are permitted")
    conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=timeout)
    try:
        payload = None if packet is None else json.dumps(
            packet, ensure_ascii=False, separators=(",", ":")).encode("utf8")
        conn.request(method, path, payload,
                     {"Content-Type": "application/json"} if payload else {})
        response = conn.getresponse()
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if response.status != 200 or len(raw) > MAX_RESPONSE_BYTES:
            raise RuntimeError("Local Ollama request failed")
        obj = json.loads(raw)
        if type(obj) is not dict:
            raise RuntimeError("Local Ollama returned non-object response")
        return obj
    finally:
        conn.close()


def require_base_digest() -> str:
    tags = request("GET", "/api/tags")
    models = tags.get("models")
    if type(models) is not list:
        raise RuntimeError("No Ollama model inventory")
    for model in models:
        if type(model) is dict and model.get("name") == BASE_MODEL:
            digest = model.get("digest")
            if type(digest) is not str:
                raise RuntimeError("Missing model digest")
            # Ollama /api/tags normally returns the bare 64-hex SHA,
            # while the historic benchmark manifest stores sha256:<hex>.
            # Normalize representation only; NEVER relax equality.
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                digest = "sha256:" + digest
            if digest != ORIGINAL_DIGESTS[BASE_MODEL]:
                raise RuntimeError("Base model digest differs from pinned PR533 baseline")
            return digest
    raise RuntimeError("Expected local Qwen3.5 9B model missing")


def query_structured(model: str, turns: list[dict], *, seed: int,
                     repair: bool = False) -> str:
    if type(seed) is not int or not 0 <= seed < (2**31):
        raise ValueError("Invalid decoder seed")
    if type(turns) is not list or len(turns) != 2:
        raise ValueError("Only one trusted extraction question per request")
    question = turns[-1]["content"]
    # The first turn uses the exact same structured format as PR533.
    # A solver-repair turn contains original_question in canonical feedback.
    # For the reparative turn we must constrain based on the ORIGINAL question,
    # never a JSON blob with accidental 'how many' matches.
    if repair:
        feedback = json.loads(question)
        original = feedback.get("original_question")
        if type(original) is not str:
            raise ValueError("Repair requires a source question")
        format_question = original
    else:
        format_question = question
    packet = {
        "model": model,
        "messages": turns,
        "think": False,
        "stream": False,
        "format": extraction_response_format(format_question),
        "options": {
            "temperature": 0, "num_predict": 1536 if repair else 384,
            "num_ctx": 4096, "seed": seed,
        },
        "keep_alive": "5m",
    }
    response = request("POST", "/api/chat", packet)
    if response.get("done") is not True or response.get("done_reason") != "stop":
        raise RuntimeError("Incomplete Ollama JSON extraction")
    message = response.get("message")
    content = message.get("content") if type(message) is dict else None
    if (type(content) is not str or
            not 2 <= len(content.encode("utf8")) <= MAX_MODEL_OUTPUT_BYTES):
        raise RuntimeError("Missing or oversize Ollama JSON extraction")
    return content


def query_fallback(model: str, turns: list[dict], *, seed: int) -> str:
    packet = {
        "model": model, "messages": turns, "think": False,
        "stream": False, "keep_alive": "5m",
        "options": {
            "temperature": .7, "top_p": .8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5,
            "repeat_penalty": 1.0, "num_predict": 2048, "num_ctx": 4096,
            "seed": seed,
        },
    }
    response = request("POST", "/api/chat", packet)
    if response.get("done") is not True or response.get("done_reason") != "stop":
        raise RuntimeError("Fallback did not complete")
    message = response.get("message")
    content = message.get("content") if type(message) is dict else None
    if type(content) is not str or not content.strip():
        raise RuntimeError("Empty fallback response")
    if len(content.encode("utf8")) > 16000:
        raise RuntimeError("Fallback answer too long")
    return content


def extract_count(question: str, content: str) -> tuple[int | None, str]:
    rejection = []
    count = parse_extracted_constraints(
        question, content, rejection_reason_sink=rejection)
    return (count if type(count) is int else None,
            rejection[-1] if rejection else "invalid")


def run_case(case: dict, model: str, seed: int) -> dict:
    q = case["question"]
    expected = case["expected_count"]
    start = time.monotonic()
    first = ""
    first_error = False
    try:
        first = query_structured(
            model, [{"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": q}], seed=seed)
    except (RuntimeError, ValueError, OSError, TimeoutError):
        first_error = True
    original_count, original_reason = extract_count(q, first)
    # Reinitialize a deterministic controller run. The first extraction
    # should get the same seed/constraints; second attempt gets 1536 tokens.
    attempts = 0
    def extract(turns):
        nonlocal attempts
        attempts += 1
        if attempts > 2:
            raise RuntimeError("Controller exceeded one solver-repair turn")
        if attempts == 1 and first and not first_error:
            return first
        return query_structured(model, turns, seed=seed,
                                repair=attempts == 2)
    def fallback(turns):
        try:
            return query_fallback(model, turns, seed=seed)
        except (RuntimeError, ValueError, OSError, TimeoutError):
            return "UNSURE"

    try:
        result = run_experimental_tir(
            [{"role": "system", "content": "NERAL local structured evaluation"},
             {"role": "user", "content": q}],
            fallback, extract_constraints=extract)
    except Exception:
        result = {"text": "UNSURE", "extraction_status": "error",
                  "tool_used": None, "first_turn_telemetry": {}}
    tele = result.get("first_turn_telemetry")
    if type(tele) is not dict:
        tele = {}
    repair_state = tele.get("solver_repair_status")
    repaired = repair_state == "repaired"
    status = result.get("extraction_status")
    formal = result.get("tool_used") == "neral_logic_exact"
    text = result.get("text") if type(result.get("text")) is str else ""
    # Numeric fallback is scoreable as an answer but cannot establish a
    # semantically faithful structured schema.
    match = re.fullmatch(r"(?:FINAL_ANSWER:\s*)?(-?\d{1,10})", text.strip())
    final_num = int(match.group(1)) if match else None
    if formal is False:
        # Fallback text is allowed to be scored separately, never as a
        # validated formal extraction.
        pass
    return {
        "id": case["id"],
        "family": case["family"],
        "raw_correct": original_count == expected,
        "raw_schema_accepted": original_count is not None,
        "raw_error": first_error,
        "raw_rejection": original_reason if original_reason in (
            "accepted", "kind_none", "invalid_json", "invalid_input",
            "solver_type_error", "solver_unsat_fallback",
            "solver_schema_error", "solver_rejected", "invalid_structure",
            "invalid_envelope", "evidence_quote_invalid",
            "incomplete_formal", "variable_not_in_source"
        ) else "other",
        "controller_final_correct": final_num == expected,
        "controller_formal_correct": formal and final_num == expected,
        "controller_formal_accepted": formal,
        "repair_attempted": attempts >= 2,
        "repair_accepted": repaired,
        "repair_correct": repaired and final_num == expected,
        "status": status if status in ("accepted","repaired","fallback","error") else "other",
        "seconds": round(time.monotonic() - start, 3),
    }


def summarize(cases: list[dict],rows:list[dict],model:str,digest:str)->dict:
    if len(rows) != 24 or len({r["id"] for r in rows}) != 24:
        raise ValueError("24 unique real observations required")
    totals = {key: sum(bool(row[key]) for row in rows) for key in (
        "raw_correct","raw_schema_accepted","raw_error",
        "controller_final_correct","controller_formal_correct",
        "controller_formal_accepted","repair_attempted","repair_accepted",
        "repair_correct",
    )}
    domains = {}
    for f in sorted({r["family"] for r in rows}):
        match = [r for r in rows if r["family"]==f]
        domains[f]={"correct":sum(r["controller_final_correct"] for r in match),
                    "total":len(match)}
    return {
        "status": "REAL_OLLAMA_24_HOLDOUT_DEV_NONBLIND",
        "model": model, "model_digest": digest,
        "parent_baseline_sha": EXPECTED_BASE_SHA,
        "seed_base": BASE_SEED,
        "max_first_predict": 384, "max_repair_predict": 1536,
        "temperature": 0, "thinking": False,
        "total": len(rows), **totals,
        "by_family": domains,
        "cases": rows, # public case ids and scalars only; no prompts or JSON
        "limitations": "24 curated visible development cases, not blinded. "
        "Untrained Ollama quantization and HF base may differ.",
    }


def gpu_offload_mib(model:str)->int:
    live=request("GET","/api/ps")
    rows=live.get("models")
    if type(rows) is not list:
        return 0
    sizes=[r.get("size_vram",0) for r in rows if type(r) is dict
           and r.get("name")==model
           and type(r.get("size_vram")) is int]
    return max(sizes, default=0)//1048576


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--authorize",required=True)
    p.add_argument("--model",default=BASE_MODEL)
    p.add_argument("--holdout",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--baseline",action="store_true")
    args=p.parse_args()
    if args.authorize != AUTH or os.getenv("NERAL_EXPERIMENTAL_TIR") != "1":
        raise SystemExit("Explicit manual local holdout mode required")
    if args.baseline and args.model != BASE_MODEL:
        raise SystemExit("Baseline must use exact original Ollama tag")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Refusing to overwrite old holdout results")
    cases=load_holdout(args.holdout)
    model_digest=require_base_digest() if args.baseline else ""
    if not args.baseline:
        inventory=request("GET","/api/tags").get("models",[])
        entries=[x for x in inventory if type(x) is dict
                 and x.get("name")==args.model]
        if len(entries)!=1:
            raise RuntimeError("Exact local adapter model absent")
        model_digest=entries[0].get("digest","")
    rows=[]
    for i,case in enumerate(cases):
        s=(BASE_SEED+i) % (2**31)
        rows.append(run_case(case,args.model,s))
        print("HOLDOUT_PROGRESS "+str(i+1)+"/24",flush=True)
    output=summarize(cases,rows,args.model,model_digest)
    output["gpu_offload_mib"]=gpu_offload_mib(args.model)
    output["gpu_offload_observed"]=output["gpu_offload_mib"]>=4096
    if not output["gpu_offload_observed"]:
        raise RuntimeError("No RTX model residency; refuse CPU-backed benchmark")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf8") as file:
        json.dump(output,file,sort_keys=True,ensure_ascii=False,allow_nan=False)
        file.write("\n")
    print("NERAL_HOLDOUT_SCALARS "+json.dumps({
        k: v for k,v in output.items() if k not in ("cases","limitations")},
        ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
