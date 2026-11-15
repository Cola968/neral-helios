"""Fresh NERAL 1.1 prospective, case-sealed RTX4070 paired 100-run.

This harness NEVER prints case questions, seed, gold, generated programs or
model responses. Original future cases/gold are frozen to an immutable local
manifest *before the first answer*. The GPU and tools are REAL, not CPU mocks.
No training and no automatic code/prompt changes after seeing new data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
import tempfile
import time

SOURCE_SHA=os.environ.get("NERAL_PINNED_SOURCE_SHA","")
AUTHORIZE="NERAL_1_1_FRESH_PROSPECTIVE100_RTX4070_ONCE"
BASE_TAG="qwen3.5:9b"
TUNED_TAG="neral-qwen35-qlora-v1:latest"
CATS=("mathematics","logic","programming","mixed")
CONTROL_MAX_TOKENS=2048
REQUIRED_CATEGORIES={cat:25 for cat in CATS}
PILOT_SEED="cpu-dev-only-do-not-use-for-prospective-evaluation"

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def json_bytes(x)->bytes:
    return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode("utf8")

def write_new(p:Path,data:bytes):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())

def append_case(p:Path,obj:dict):
    with p.open("ab") as f:
        f.write(json_bytes(obj));f.flush();os.fsync(f.fileno())

def require_gpu():
    import neral_adaptive_bounded_fallback as bounded
    from neral_agent_autonomy import local_only_opener
    out=subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total",
                       "--format=csv,noheader,nounits"],capture_output=True,
                       text=True,timeout=12,check=True).stdout
    if "RTX 4070" not in out:
        raise RuntimeError("RTX4070 hardware required")
    with local_only_opener().open("http://127.0.0.1:11434/api/tags",timeout=12) as f:
        obj=json.loads(f.read(262145))
    tags={m.get("name"):m.get("digest") for m in obj.get("models",[])
          if type(m) is dict and type(m.get("digest")) is str}
    if not all(tags.get(k) for k in (BASE_TAG,TUNED_TAG)):
        raise RuntimeError("Required local model digest missing")
    info=subprocess.run(["docker","info","--format","{{.ServerVersion}}"],
             capture_output=True,timeout=12,check=False)
    if info.returncode!=0:
        raise RuntimeError("Docker server unavailable")
    if subprocess.run(["docker","image","inspect","python:3.11-alpine"],
             capture_output=True,timeout=12,check=False).returncode!=0:
        raise RuntimeError("Preinstalled trusted Python sandbox unavailable")
    if bounded.FALLBACK_MAX_TOKENS!=1536:
        raise RuntimeError("Unexpected fallback budget")
    return {"gpu":out.strip(),"model_digests":{x:tags[x] for x in (BASE_TAG,TUNED_TAG)},
            "docker_version":info.stdout.strip()}

def base_packet(original:list[dict])->dict:
    return {
        "model":BASE_TAG,
        "messages":[dict(x) for x in original],
        "stream":False,"think":False,"keep_alive":"5m",
        "options":{"temperature":0,"num_predict":CONTROL_MAX_TOKENS,
                   "num_ctx":4096,"repeat_penalty":1.0,"seed":0},
    }

def control_completion(original,transport):
    packet=base_packet(original)
    t=time.perf_counter()
    try:
        reply=transport(packet)
        if type(reply) is not dict:
            raise ValueError("Missing structured raw response")
        finished=reply.get("done") is True and reply.get("done_reason")=="stop"
        text=reply.get("message",{}).get("content") if finished else None
        if type(text) is not str or not text.strip() or "<think>" in text.lower():
            finished=False
        return {
            "completed":finished,"text":text if finished else "",
            "done_reason":reply.get("done_reason"),
            "raw_model_envelope":reply,
            "latency_seconds":round(time.perf_counter()-t,4),
            "load_ms":(reply.get("load_duration")/1e6
                       if type(reply.get("load_duration")) is int else None)
        }
    except Exception as exc:
        return {"completed":False,"text":"","done_reason":"transport_error",
                "exception_class":type(exc).__name__,
                "latency_seconds":round(time.perf_counter()-t,4)}

def score_final(text,done,gold):
    from neral_answer_contract import parse_terminal_final_integer
    from scripts.neral_blind_100_offline_score import conservative_final_integer
    if type(gold) is not int or type(text) is not str or done!="stop":
        return {"strict_correct":False,"offline_correct":False,
                "strict_answer":None,"offline_answer":None}
    strict=parse_terminal_final_integer(text)
    offline=conservative_final_integer(text)
    return {"strict_correct":strict==gold,"offline_correct":offline==gold,
            "strict_answer":strict,"offline_answer":offline}

def tally(cases,rows):
    if len(cases)!=100 or len(rows)!=100 or {x["id"] for x in cases}!={x["id"] for x in rows}:
        raise ValueError("Exactly 100 distinct sealed completed paired rows required")
    byid={x["id"]:x for x in rows}
    cats={}
    for cat in CATS:
        items=[byid[c["id"]] for c in cases if c["category"]==cat]
        if len(items)!=25:
            raise ValueError("Missing category count")
        base=sum(x["base_strict_correct"] for x in items)
        neral=sum(x["neral_strict_correct"] for x in items)
        losses=sum(x["base_strict_correct"] and not x["neral_strict_correct"] for x in items)
        wins=sum(not x["base_strict_correct"] and x["neral_strict_correct"] for x in items)
        cats[cat]={"base":base,"neral":neral,"won":wins,"lost":losses}
    lat_base=[x["base_seconds"] for x in rows]
    lat_neral=[x["neral_seconds"] for x in rows]
    verified=[x for x in rows if x.get("independent_verified")]
    z3=[x for x in rows if x.get("strategy")=="Z3_SCHEMA"]
    python=[x for x in rows if x.get("strategy")=="PYTHON_SANDBOX"]
    base_strict=sum(x["base_strict_correct"] for x in rows)
    final_strict=sum(x["neral_strict_correct"] for x in rows)
    base_offline=sum(x["base_offline_correct"] for x in rows)
    final_offline=sum(x["neral_offline_correct"] for x in rows)
    reg=sum(x["base_strict_correct"] and not x["neral_strict_correct"] for x in rows)
    secondary_losses=sum(x["base_offline_correct"] and not x["neral_offline_correct"] for x in rows)
    return {
        "cases":100,"category_breakdown":cats,
        "strict":{"base":base_strict,"neral":final_strict,
                  "gained":sum(x["neral_strict_correct"] and not x["base_strict_correct"] for x in rows),
                  "lost":reg},
        "offline_conservative":{"base":base_offline,"neral":final_offline,
                "gained":sum(x["neral_offline_correct"] and not x["base_offline_correct"] for x in rows),
                "lost":secondary_losses},
        "independently_verified_total":len(verified),
        "verified_z3":sum(x.get("independent_verified") for x in z3),
        "z3_candidates":len(z3),
        "verified_python":sum(x.get("independent_verified") for x in python),
        "python_candidates":len(python),
        "latency_seconds":{
            "base_median":statistics.median(lat_base),
            "neral_median":statistics.median(lat_neral),
            "base_p95":sorted(lat_base)[94],
            "neral_p95":sorted(lat_neral)[94]},
        "n_proposal_calls":sum(x.get("proposal_calls",0) for x in rows),
        "n_fallback_calls":sum(x.get("fallback_calls",0) for x in rows),
        "n_docker_calls":sum(x.get("docker_calls",0) for x in rows),
        "target_at_least_85":final_strict>=85,
        "zero_regression_strict":reg==0,
        "zero_regression_offline":secondary_losses==0,
        "pass_gate":final_strict>=85 and reg==0 and secondary_losses==0,
    }

def frozen_cases(seed_hex):
    from scripts.neral_prospective_100_cases import build,serialize,CATEGORIES
    rows=build(seed_hex)
    if len(rows)!=100 or CATEGORIES!=CATS:
        raise RuntimeError("100-case prospective generator integrity")
    for c in CATS:
        if sum(r["category"]==c for r in rows)!=25:
            raise RuntimeError("Unbalanced prospective categories")
    if len({x["question_sha256"] for x in rows})!=100:
        raise RuntimeError("Unexpected duplicate prospective questions")
    return rows,serialize(rows)

def generate_receipt(cases,train,holdout):
    from scripts.neral_prospective_100_cases import exclusion_receipt
    return exclusion_receipt(cases,train,holdout)

def locally_rebuild_training(holdout:Path,root:Path)->Path:
    # The existing 720-example QLoRA training-data generator is used ONLY
    # to validate overlap hashes, not to train or fine-tune anything.
    from scripts.neral_qlora_dataset import generate,compact,load_holdout
    excluded,_=load_holdout(holdout)
    rows=generate(pairs=360,exclusion_questions=excluded)
    data="".join(compact(x)+"\n" for x in rows).encode("utf8")
    t=root/"training-reconstruction-EXCLUSION-ONLY.tmp"
    write_new(t,data)
    return t

def pilot(transport):
    """Unseen relative to the sealed set; different human-written CPU quality cases.

    Not scored as 100-task evidence. No pilot answers or prompts printed.
    It prevents sealing a new blind corpus when fallback safety is broken.
    """
    from neral_adaptive_bounded_fallback import make_bounded_fallback
    from neral_answer_contract import formatted_benchmark_system
    questions=(
        ("How many integers from 1 to 123 inclusive are divisible by 7?",17),
        ("A capacitor holds 31 coulombs and then receives 47 more coulombs. "
         "How much charge does it hold now in coulombs?",78),
        ("For integer i from 0 through 14 inclusive, calculate the sum of i*i.",1015),
        ("How many arrangements of 4 distinct letters are possible?",24),
    )
    out=[]
    for i,(q,gold) in enumerate(questions):
        orig=[{"role":"system","content":formatted_benchmark_system()},
              {"role":"user","content":q}]
        base=control_completion(orig,transport)
        send_packets=[]
        def wrapped(packet):
            send_packets.append({"tokens":packet["options"]["num_predict"]})
            return transport(packet)
        t=time.perf_counter()
        try:
            candidate=make_bounded_fallback(BASE_TAG,
                transport=wrapped,formatted_contract=True)(orig)
            completed=True
        except Exception:
            candidate="";completed=False
        bb=score_final(base["text"],"stop" if base["completed"] else "length",gold)
        cc=score_final(candidate,"stop" if completed else "length",gold)
        out.append({
            "case_id":"pilot-"+str(i+1),
            "question_sha256":sha(q.encode("utf8")),"gold_integer":gold,
            "base_correct":bb["strict_correct"],
            "bounded_correct":cc["strict_correct"],
            "base_completed":base["completed"],
            "bounded_completed":completed,
            "base_seconds":base["latency_seconds"],
            "bounded_seconds":round(time.perf_counter()-t,4),
            "fallback_token_attempts":[x["tokens"] for x in send_packets],
            "base_raw_envelope":base.get("raw_model_envelope"),
            "bounded_response":candidate,
        })
    # Fail closed only on observed regression, not on unanswered hard pilots.
    losses=sum(x["base_correct"] and not x["bounded_correct"] for x in out)
    return {"cases":out,"losses":losses,
            "base_correct":sum(x["base_correct"] for x in out),
            "bounded_correct":sum(x["bounded_correct"] for x in out)}

def run_case(case,transport):
    """No gold is passed to either model, proposer, source parser or fallback."""
    from neral_answer_contract import (
        formatted_benchmark_system,format_verified_terminal_answer)
    from neral_adaptive_planner import run_adaptive_planner,run_python_in_docker
    from neral_adaptive_ollama import make_local_adaptive_proposer
    from neral_adaptive_bounded_fallback import make_bounded_fallback
    from neral_python_source_router import verify_python_full_source

    original=[{"role":"system","content":formatted_benchmark_system()},
              {"role":"user","content":case["question"]}]
    assert all("gold_integer" not in m["content"] for m in original)
    base=control_completion(original,transport)
    candidates=[];fallbacks=[];docker_traces=[]
    def traced_proposal(payload):
        started=time.perf_counter()
        record={"request":payload}
        candidates.append(record)
        try:
            answer=transport(payload)
            record["raw_model_envelope"]=answer
            return answer
        except Exception as exc:
            record["error_class"]=type(exc).__name__
            raise
        finally:
            record["elapsed_ms"]=round((time.perf_counter()-started)*1000,3)
    def traced_fallback(payload):
        started=time.perf_counter()
        record={"request":payload}
        fallbacks.append(record)
        try:
            answer=transport(payload)
            record["raw_model_envelope"]=answer
            return answer
        except Exception as exc:
            record["error_class"]=type(exc).__name__
            raise
        finally:
            record["elapsed_ms"]=round((time.perf_counter()-started)*1000,3)
    def traced_docker(program):
        record={"program":program}
        docker_traces.append(record)
        started=time.perf_counter()
        try:
            response=run_python_in_docker(program)
            record["result"]=response
            return response
        except Exception as exc:
            record["error_class"]=type(exc).__name__
            raise
        finally:
            record["elapsed_ms"]=round((time.perf_counter()-started)*1000,3)

    model=make_local_adaptive_proposer(TUNED_TAG,transport=traced_proposal)
    fallback=make_bounded_fallback(BASE_TAG,transport=traced_fallback,
                                   formatted_contract=True)
    start=time.perf_counter()
    response=run_adaptive_planner(original,model,fallback,
                                  traced_docker,verify_python_full_source)
    neral_seconds=round(time.perf_counter()-start,4)
    if type(response) is not dict or response.get("verified") is True and (
            response.get("safe_to_promote") is not True
            or response.get("response_source") not in (
                "independent_source_parser_z3","independent_source_python")):
        raise RuntimeError("Independent source promotion invariant violated")
    raw_answer=response.get("text")
    if type(raw_answer) is not str:
        raw_answer=""
    result_answer=format_verified_terminal_answer(
        raw_answer,independently_verified=response.get("verified") is True)
    gold=case["gold_integer"]  # first read AFTER both model decisions
    base_score=score_final(base["text"],
        "stop" if base["completed"] else "length",gold)
    neral_done=("stop" if response.get("response_source") in (
        "independent_source_parser_z3","independent_source_python","raw_qwen") else "length")
    neral_score=score_final(result_answer,neral_done,gold)
    # Distinguish the verified source trust from mere model output format.
    return {
       "id":case["id"],"category":case["category"],"family":case["family"],
       "question_sha256":case["question_sha256"],
       "base_strict_correct":base_score["strict_correct"],
       "neral_strict_correct":neral_score["strict_correct"],
       "base_offline_correct":base_score["offline_correct"],
       "neral_offline_correct":neral_score["offline_correct"],
       "base_seconds":base["latency_seconds"],
       "neral_seconds":neral_seconds,
       "base_completed":base["completed"],
       "base_done_reason":base["done_reason"],
       "independent_verified":response.get("verified") is True,
       "strategy":response.get("planner_route"),
       "verification_scope":response.get("verification_scope"),
       "proof_backend":response.get("proof_backend"),
       "selection_policy":response.get("selection_policy"),
       "proposal_calls":len(candidates),
       "fallback_calls":len(fallbacks),
       "docker_calls":len(docker_traces),
       "repair_rounds":max(0,len(candidates)-1),
       "base_final_integer":base_score["strict_answer"],
       "neral_final_integer":neral_score["strict_answer"],
       "raw":{"question":case["question"],"gold_integer":gold,
              "original_turns":original,
              "base_full_envelope":base.get("raw_model_envelope"),
              "base_raw_text":base["text"],
              "neral_raw_text":raw_answer,
              "neral_formatted_final":result_answer,
              "neral_controller":response,
              "qlora_proposals":candidates,
              "qwen_fallbacks":fallbacks,
              "sandbox_traces":docker_traces}
    }

def live(out:Path):
    import neral_nl_local_ollama as local
    if (type(SOURCE_SHA) is not str or len(SOURCE_SHA)!=40
            or any(c not in "0123456789abcdef" for c in SOURCE_SHA)):
        raise RuntimeError("Pinned source SHA missing or invalid")
    actual=subprocess.run(["git","rev-parse","HEAD"],
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if actual != SOURCE_SHA:
        raise RuntimeError("Code checkout does not match immutable source SHA")
    from neral_agent_autonomy import local_only_opener
    from scripts.neral_prospective_100_cases import serialize
    if out.exists() or out.is_symlink():
        raise FileExistsError("One-shot blind evidence path already exists")
    if out.parent.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError("Blind cases cannot be written into source checkout")
    hardware=require_gpu()
    out.mkdir(parents=True,exist_ok=False)
    # Pilot BEFORE drawing new seed. Never condition seed generation on any
    # candidate question, never change code after looking at prospective gold.
    p=pilot(local._localhost_transport)
    write_new(out/"pilot-safety-raw.json",json_bytes(p))
    print("BLIND100_PILOT "+json.dumps({
        "cases":len(p["cases"]),"base_correct":p["base_correct"],
        "bounded_correct":p["bounded_correct"],"losses":p["losses"]}),flush=True)
    if p["losses"]:
        raise RuntimeError("Qwen fallback pilot regression; prospective corpus was not generated")
    seed_hex=secrets.token_hex(32)
    cases,blob=frozen_cases(seed_hex)
    train=None
    try:
        with tempfile.TemporaryDirectory(prefix="neral-exclusion-",dir=str(out)) as folder:
            train=locally_rebuild_training(
                Path(__file__).resolve().parents[1]/"datasets/neral_qwen35_holdout_v1.jsonl",
                Path(folder))
            receipt=generate_receipt(cases,train,
                Path(__file__).resolve().parents[1]/"datasets/neral_qwen35_holdout_v1.jsonl")
    except Exception as exc:
        raise RuntimeError("Immutable prior-corpus exclusion failed: "+type(exc).__name__) from None
    corpus_sha=sha(blob)
    if receipt["prospective100_sha256"]!=corpus_sha:
        raise RuntimeError("Prospective corpus SHA mismatch")
    manifest={
        "status":"SEALED_BEFORE_MODEL_INFERENCE","source_sha":SOURCE_SHA,
        "seed_sha256":sha(seed_hex.encode("utf8")),
        "prospective100_sha256":corpus_sha,
        "generation_receipt":receipt,
        "case_count":100,"categories":REQUIRED_CATEGORIES,
        "model_hardware":hardware,
        "control":{"model":BASE_TAG,"max_tokens":2048,"think":False,
                   "temperature":0,"seed":0},
        "neral_fallback":{"model":BASE_TAG,
           "first_max_tokens":1536,"truncation_retry_max_tokens":2048,
           "think":False,"temperature":0,"seed":0},
        "scoring":"strict terminal FINAL_ANSWER primary; independent conservative offline secondary",
        "no_training":True,"no_main_write":True,"no_blind_preinspection":True
    }
    write_new(out/"prospective100-seed-secret.txt",(seed_hex+"\n").encode("utf8"))
    write_new(out/"prospective100-sealed-cases.jsonl",blob)
    write_new(out/"prospective100-seal.json",json_bytes(manifest))
    print("BLIND100_SEALED "+json.dumps({
        "sha256":corpus_sha,"seed_sha256":manifest["seed_sha256"],
        "categories":REQUIRED_CATEGORIES,"cases":100},sort_keys=True),flush=True)
    # Do not ever materialize gold in the model messages. Model functions
    # only consume original turns. Raw evidence is persisted per pair.
    progress=out/"paired100-progress.jsonl"
    raws=out/"paired100-full-raw.jsonl"
    write_new(progress,b"")
    write_new(raws,b"")
    rows=[]
    for ix,case in enumerate(cases):
        try:
            entry=run_case(case,local._localhost_transport)
        except Exception as exc:
            write_new(out/"failure.json",json_bytes({
                "at_case_index":ix,"case_id":case["id"],
                "error_class":type(exc).__name__,
                "sealed_sha256":corpus_sha,"completed_pairs":len(rows)}))
            raise
        raw=entry.pop("raw")
        append_case(raws,{"id":case["id"],"category":case["category"],
                         "question_sha256":case["question_sha256"],**raw})
        append_case(progress,entry)
        rows.append(entry)
        print("BLIND100_PROGRESS "+str(len(rows))+"/100",flush=True)
    stats=tally(cases,rows)
    final={**stats,"status":"FINISHED_PROSPECTIVE100_REAL_RTX4070",
           "source_sha":SOURCE_SHA,"seed_sha256":manifest["seed_sha256"],
           "sealed_casepack_sha256":corpus_sha,
           "raw_rows_sha256":sha(raws.read_bytes()),
           "progress_sha256":sha(progress.read_bytes()),
           "no_training":True,"no_main_write":True,
           "notes":"Observed losses are empirical; no future zero-regression guarantee. "
                    "This generator's template families are known; the random instances are new."}
    write_new(out/"prospective100-final-summary.json",json_bytes(final))
    print("BLIND100_FINAL "+json.dumps(
        {k:v for k,v in final.items()
         if k not in ("seed_hex","case_contents")},sort_keys=True),flush=True)
    return final

def cpu_preflight():
    # Fixed QA-only seed. Does not touch future seed; no future tasks printed.
    from neral_python_assert_contract import inspect_python_contract
    cases,blob=frozen_cases(PILOT_SEED)
    assert len(cases)==100 and len({x["question_sha256"] for x in cases})==100
    assert sha(blob)==hashlib.sha256(blob).hexdigest()
    assert inspect_python_contract("n=11\nassert n==5+6\nprint(n)") is None
    assert inspect_python_contract("n=11\nassert n>=0\nprint(n)")=="python_missing_assert"
    check=[{ "id":c["id"],"category":c["category"],
           "base_strict_correct":False,"neral_strict_correct":False,
           "base_offline_correct":False,"neral_offline_correct":False,
           "base_seconds":0.01,"neral_seconds":0.02,
           "strategy":"DIRECT","independent_verified":False} for c in cases]
    score=tally(cases,check)
    assert score["cases"]==100 and score["strict"]["lost"]==0
    print("BLIND100_CPU_PREFLIGHT_OK cases=100 categories=25_each no_gpu",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorize",required=True)
    p.add_argument("--cpu-preflight",action="store_true")
    p.add_argument("--live",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if (args.authorize!=AUTHORIZE or args.cpu_preflight==args.live
            or (args.live and args.out is None)):
        p.error("Explicit prospective-only authorization and mode required")
    if args.cpu_preflight:
        cpu_preflight();return
    stats=live(args.out)
    # Scoring is evidence: missing requested threshold does not erase raw data.
    # A separate release gate MUST reject a failed benchmark, never autopromote.
    print("BLIND100_RELEASE_GATE "+("PASS" if stats["pass_gate"] else "BLOCKED"),
          flush=True)

if __name__=="__main__":
    main()
