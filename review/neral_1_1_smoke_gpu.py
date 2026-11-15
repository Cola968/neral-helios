"""Pinned RTX4070 original eight diagnostics plus two strictly unseen unique logic probes for pinned NERAL 1.1 Z3-first failfast commit 43e527ed.

Eight new synthetic *diagnostic* cases. No use of any 100-case corpus,
generator, release gate, training file, or task ID. Never substitutes mocked
model replies. Each real model packet, generated program, Docker observation,
and final response is stored (the GitHub Actions artifact expires after the
repository's retention period). CLI prints scalar summaries only.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from urllib.request import Request

PIN="6ebe453af75c26d77767fb7e6cdd14c6896295ca"
RUN_AUTH="NERAL_1_1_AST_UNIQUE_BOUNDED_RTX4070_TEN_ONCE"
BASE="qwen3.5:9b"
TUNED="neral-qwen35-qlora-v1:latest"
QUESTION1=("Color vertices 0..6 with red, green or blue. Vertex 0 is red, "
           "and endpoints of edges [(0, 1), (1, 2), (2, 3), (3, 4), "
           "(4, 5), (5, 6), (6, 0), (1, 4)] must differ. "
           "How many valid colorings?")
QUESTION2=("Ein sechsstelliger Code verwendet 1, 2, 3, 4, 5 und 6 genau einmal. "
           "Die 2 steht vor der 6, nicht direkt davor. "
           "Zwischen der 5 und 4 steht genau eine Ziffer. "
           "Die 6 steht unmittelbar rechts von der 4. "
           "Die 1 steht nicht am Rand. Bestimme alle gültigen Codes.")
LOGIC1=("Count consistent truth assignments for A, B, C and D: "
        "A says B lies; B says C lies; C says A and D have different truth "
        "values; D says A is truthful. Exactly two people tell the truth.")
LOGIC2=("Count bijections assigning rooms 1 through 5 to Ada, Bo, Cy, Dee "
        "and Eli: Ada has a lower room than Bo; Cy is not in room 1; "
        "Dee is in the room immediately after Eli.")
MATH1=("Compute the exact integer (137 squared minus 91 squared) divided "
       "by 46, plus 17 times 29, minus 5 cubed.")
MATH2=("Compute the sum of (k cubed minus k) for integer k from 1 through "
       "24 inclusive, take the remainder modulo 97, then add 7 times 13.")
REPAIR=("Count 5-tuples with entries from 0 through 4. "
        "Their sum is congruent to 2 modulo 3. "
        "Exactly 3 entries are even. All entries are distinct.")
NEGATIVE=("Count permutations of numbers 1 through 6. "
          "Number 1 is before number 2. Number 6 must be lucky.")
FRESH_TRUTH=("Find the unique truth assignment for speakers A, B, C, D. "
             "A says B lies. B says C tells the truth. C says D tells the truth. "
             "D says A lies. Exactly 1 speakers tell the truth. "
             "Return the truth bitmask.")
FRESH_ROOM=("Find the unique room assignment for Lena, Milo, Nora, Otto, Pia "
            "to rooms 1 through 5. Lena is first. Pia is last. "
            "Milo is before Nora. Nora is immediately before Otto. "
            "Return Milo's room number.")
CASES=(
    {"id":"s01-new-graph7","group":"Z3","prompt":QUESTION1,"expected_route":"Z3_SCHEMA"},
    {"id":"s02-new-pin6","group":"Z3","prompt":QUESTION2,"expected_route":"Z3_SCHEMA"},
    {"id":"s03-truth-liars4","group":"GENERAL_LOGIC","prompt":LOGIC1,"expected_route":"PYTHON_SANDBOX"},
    {"id":"s04-room-assign5","group":"GENERAL_LOGIC","prompt":LOGIC2,"expected_route":"PYTHON_SANDBOX"},
    {"id":"s05-math-multistep1","group":"MATH","prompt":MATH1,"expected_route":"PYTHON_SANDBOX"},
    {"id":"s06-math-multistep2","group":"MATH","prompt":MATH2,"expected_route":"PYTHON_SANDBOX"},
    {"id":"s07-correction-synthetic","group":"REPAIR","prompt":REPAIR,"expected_route":"PYTHON_SANDBOX"},
    {"id":"s08-ambiguous-negative","group":"NEGATIVE","prompt":NEGATIVE,"expected_route":"PYTHON_SANDBOX"},
    {"id":"s09-fresh-truth4","group":"UNIQUE_LOGIC","prompt":FRESH_TRUTH,"expected_route":"PYTHON_SANDBOX"},
    {"id":"s10-fresh-room5","group":"UNIQUE_LOGIC","prompt":FRESH_ROOM,"expected_route":"PYTHON_SANDBOX"},
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    # Atomic per-case replacement, with disk flush to preserve partial runs.
    raw=(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n").encode("utf8")
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("wb") as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    temp.replace(path)


def gold_for(case_id):
    if case_id=="s03-truth-liars4":
        return sum(
            (a==(not b)) and (b==(not c)) and (c==(a!=d))
            and (d==a) and (sum((a,b,c,d))==2)
            for a,b,c,d in itertools.product((False,True),repeat=4))
    if case_id=="s04-room-assign5":
        return sum(
            rooms[0]<rooms[1] and rooms[2]!=1
            and rooms[3]==rooms[4]+1
            for rooms in itertools.permutations((1,2,3,4,5)))
    if case_id=="s05-math-multistep1":
        return (137**2-91**2)//46+17*29-5**3
    if case_id=="s06-math-multistep2":
        return sum(k**3-k for k in range(1,25))%97+7*13
    return None


def source_preflight():
    from neral_nl_verified_route import strict_source_certificate
    from neral_python_logic_verifier import (
        parse_bounded_logic_source,independently_count)
    from neral_typed_constraints_v2 import PREFIX,solve_typed_constraints
    from neral_python_unique_logic import parse_finite_unique_source
    assert digest(json.dumps(CASES[:8],sort_keys=True,ensure_ascii=False).encode("utf8")) == (
        "432148ba99fe507a60cdd619a68b6b69a985789544c9c63ed29192d8e04d5de8"
    ), "Original synthetic eight-case matrix changed"
    assert len(CASES)==10 and len({c["prompt"] for c in CASES})==10
    assert len({c["id"] for c in CASES})==10
    assert tuple(x["group"] for x in CASES)==(
        "Z3","Z3","GENERAL_LOGIC","GENERAL_LOGIC",
        "MATH","MATH","REPAIR","NEGATIVE","UNIQUE_LOGIC","UNIQUE_LOGIC")
    refs={}
    for c in CASES:
        question=c["prompt"]
        cert=strict_source_certificate(question)
        if c["group"]=="Z3":
            assert cert is not None,c["id"]
            answer=solve_typed_constraints(PREFIX+json.dumps(cert,ensure_ascii=False))
            assert type(answer) is int
            refs[c["id"]]=answer
        else:
            assert cert is None,c["id"]
            if c["group"]=="UNIQUE_LOGIC":
                unique=parse_finite_unique_source(question)
                assert unique is not None and unique.number_of_solutions==1, c["id"]
                assert type(unique.answer) is int, c["id"]
                refs[c["id"]]=unique.answer
                continue
            # No accidental source certification of unsupported general NL.
            if c["group"] in ("GENERAL_LOGIC","MATH","NEGATIVE"):
                assert parse_bounded_logic_source(question) is None,c["id"]
            if c["group"]=="REPAIR":
                task=parse_bounded_logic_source(question)
                assert task is not None
                refs[c["id"]]=independently_count(task)
            elif c["group"]!="NEGATIVE":
                refs[c["id"]]=gold_for(c["id"])
    assert len(refs)==9
    return refs


def info(endpoint):
    from neral_agent_autonomy import local_only_opener
    with local_only_opener().open(
        "http://127.0.0.1:11434"+endpoint,timeout=12) as response:
        raw=response.read(262145)
    if len(raw)>262144: raise ValueError("Oversized localhost response")
    return json.loads(raw)


def gpu_meta():
    data={}
    try:
        r=subprocess.run(
            ["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,text=True,timeout=10,check=True)
        data["nvidia_smi"]=r.stdout.strip()
    except Exception as exc:
        data["nvidia_smi_error_class"]=type(exc).__name__
    try:
        loaded=info("/api/ps")
        data["ollama_resident"]=[
            {"name":m.get("name"),"size_vram":m.get("size_vram"),
             "size":m.get("size")}
            for m in loaded.get("models",[]) if isinstance(m,dict)]
    except Exception as exc:
        data["ollama_ps_error_class"]=type(exc).__name__
    return data


def valid_gpu_environment():
    meta=gpu_meta()
    if "RTX 4070" not in str(meta.get("nvidia_smi","")):
        raise RuntimeError("RTX4070 was not detected on this runner")
    models=info("/api/tags").get("models",[])
    inventory={v.get("name"):v.get("digest") for v in models if isinstance(v,dict)}
    if not all(inventory.get(x) for x in (BASE,TUNED)):
        raise RuntimeError("Pinned local model tags or digest not installed")
    if BASE==TUNED:raise RuntimeError("Model identity mismatch")
    inspected=subprocess.run(["docker","image","inspect","python:3.11-alpine"],
        capture_output=True,timeout=12,check=False)
    if inspected.returncode!=0:
        raise RuntimeError("Docker Python image missing: no network pull authorized")
    service=subprocess.run(["docker","info","--format","{{.ServerVersion}}"],
        capture_output=True,timeout=12,check=False)
    if service.returncode!=0:
        raise RuntimeError("Local Docker daemon not ready")
    return {"inventory":{m:inventory[m] for m in (BASE,TUNED)},
            "gpu":meta,"runtime":"actual localhost Ollama; real Docker"}


def scalar(text):
    if not isinstance(text,str):return None
    text=text.strip()
    if re.fullmatch(r"(?:0|[1-9]\d{0,11})",text):
        return int(text)
    # A very narrow identical terminal extraction for both arms, not a
    # forgiving search for arbitrary digits in a long explanation.
    match=re.search(r"(?:^|\n)(?:FINAL_ANSWER:\s*|Answer:\s*)([0-9]{1,12})\s*\Z",
                    text,re.I)
    if match:return int(match.group(1))
    return None


def run(out):
    from flask import Flask
    from neral_agent_autonomy import OllamaLocalBackend
    from neral_local_chat_api import register_neral_local_chat_api
    import neral_adaptive_ollama as adaptive
    import neral_adaptive_bounded_fallback as bounded
    import neral_adaptive_planner as planner
    from neral_nl_verified_route import strict_source_certificate
    from neral_typed_constraints_v2 import PREFIX,solve_typed_constraints

    refs=source_preflight()
    environment=valid_gpu_environment()
    out.mkdir(parents=True,exist_ok=False)
    manifest={"source_commit":PIN,"authorization":RUN_AUTH,
        "no_blind100_access":True,"no_training":True,"no_merge":True,
        "case_count":len(CASES),"case_definitions_sha256":
        digest(json.dumps(CASES,sort_keys=True,ensure_ascii=False).encode("utf8")),
        "script_sha256":digest(Path(__file__).read_bytes()),
        "cases":[dict(c) for c in CASES],
        "gpu_environment":environment}
    write_json(out/"manifest.json",manifest)
    backend=OllamaLocalBackend(BASE,inference_profile="measured_nonthinking_2048")
    app=Flask("neral-1-1-synthetic-gpu-once")
    register_neral_local_chat_api(app,backend)
    client=app.test_client()
    actual_transport=adaptive._localhost_transport
    actual_bounded_transport=bounded._localhost_transport
    actual_fallback=backend.respond_chat
    actual_docker=planner.run_python_in_docker
    model_calls=[];fallback_calls=[];docker_calls=[]
    def record_model(packet):
        record={"model":packet.get("model"),"request":packet,
                "time_start_ns":time.time_ns(),"gpu_before":gpu_meta()}
        model_calls.append(record)
        t=time.perf_counter()
        try:
            result=actual_transport(packet)
            record["raw_response"]=result
            for field in ("load_duration","total_duration","eval_duration",
                          "prompt_eval_duration"):
                n=result.get(field)
                record[field+"_ms"]=(n/1e6 if isinstance(n,int) else None)
            record["done_reason"]=result.get("done_reason")
            return result
        except Exception as exc:
            record["error_type"]=type(exc).__name__
            raise
        finally:
            record["elapsed_ms"]=(time.perf_counter()-t)*1000
            record["gpu_after"]=gpu_meta()
    def record_bounded(packet):
        record={"original_messages":packet.get("messages"),
                "raw_request":packet,"time_start_ns":time.time_ns(),
                "gpu_before":gpu_meta(),"transport":"budgeted_qwen_local"}
        fallback_calls.append(record)
        t=time.perf_counter()
        try:
            result=actual_bounded_transport(packet)
            record["raw_model_envelope"]=result
            record["raw_response"]=result.get("message",{}).get("content")
            for key in ("load_duration","total_duration","eval_duration","prompt_eval_duration"):
                n=result.get(key)
                record[key+"_ms"]=n/1000000 if type(n) is int else None
            return result
        except Exception as exc:
            record["error_type"]=type(exc).__name__
            raise
        finally:
            record["elapsed_ms"]=(time.perf_counter()-t)*1000
            record["gpu_after"]=gpu_meta()
    def record_base(turns):
        record={"original_messages":turns,"time_start_ns":time.time_ns(),
                "gpu_before":gpu_meta()}
        fallback_calls.append(record)
        t=time.perf_counter()
        try:
            text=actual_fallback(turns)
            record["raw_response"]=text
            record["backend_metadata"]=backend.last_inference_diagnostic
            return text
        except Exception as exc:
            record["error_type"]=type(exc).__name__
            record["error_reason"]=getattr(exc,"reason",None)
            record["backend_metadata"]=backend.last_inference_diagnostic
            raise
        finally:
            record["elapsed_ms"]=(time.perf_counter()-t)*1000
            record["gpu_after"]=gpu_meta()
    def record_docker(code):
        record={"program":code,"time_start_ns":time.time_ns()}
        docker_calls.append(record)
        t=time.perf_counter()
        try:
            result=actual_docker(code)
            record["raw_result"]=result
            return result
        except Exception as exc:
            record["error_type"]=type(exc).__name__
            raise
        finally:
            record["elapsed_ms"]=(time.perf_counter()-t)*1000
    os.environ["NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER"]="1"
    os.environ["NERAL_EXPERIMENTAL_BOUNDED_QWEN_FALLBACK"]="1"
    adaptive._localhost_transport=record_model
    bounded._localhost_transport=record_bounded
    planner.run_python_in_docker=record_docker
    backend.respond_chat=record_base
    records=[]
    try:
        for c in CASES:
            model_calls.clear();fallback_calls.clear();docker_calls.clear()
            case_id=c["id"]
            t=time.perf_counter()
            result=None;status=None;error_type=None
            try:
                resp=client.post("/api/neral/chat/think",
                                 json={"prompt":c["prompt"],"adaptive_planner":True})
                status=resp.status_code
                result=resp.get_json()
            except Exception as exc:
                error_type=type(exc).__name__
            elapsed=(time.perf_counter()-t)*1000
            # No content is ever sent to stdout; raw JSON only in evidence.
            selected=(result or {}).get("planner_route")
            attempts=(result or {}).get("planner_attempts")
            gold=refs.get(case_id)
            final=(result or {}).get("text")
            expected=gold if type(gold) is int else None
            ref_verified=(expected is not None and scalar(final)==expected)
            model_output=[x.get("raw_response",{}).get("message",{}).get("content")
                for x in model_calls if isinstance(x.get("raw_response"),dict)]
            decisions=[]
            for obj in model_output:
                try:
                    proposal=json.loads(obj)
                    decisions.append(proposal.get("strategy"))
                except Exception:
                    decisions.append("invalid_or_truncated")
            if len(model_calls)>0:
                vram=[m for x in model_calls for m in
                      x.get("gpu_after",{}).get("ollama_resident",[])
                      if m.get("name")==TUNED and
                      isinstance(m.get("size_vram"),int)]
                tuned_gpu_proven=bool(vram and max(m["size_vram"] for m in vram)>=512*1024*1024)
            else:
                tuned_gpu_proven=False
            record={
                "id":case_id,"category":c["group"],"question":c["prompt"],
                "expected_route":c["expected_route"],"observed_proposal_strategies":decisions,
                "selected_route":selected,"proposal_calls":len(model_calls),
                "repair_rounds":max(0,len(model_calls)-1),
                "planner_attempts":attempts,"http_status":status,
                "elapsed_ms":elapsed,"gpu_loaded_qlora_512mb_proven":tuned_gpu_proven,
                "trusted_verified":(result or {}).get("verified") is True,
                "trusted_safe_to_promote":(result or {}).get("safe_to_promote") is True,
                "trusted_verification_status":(result or {}).get("verification_status"),
                "response_source":(result or {}).get("response_source"),
                "result_text":final,"computed_scalar":scalar(final),
                "offline_reference":expected,"offline_match":ref_verified,
                "offline_graded":expected is not None,
                "reference_is_not_promotion_certificate":True,
                "raw_controller_response":result,
                "raw_qlora_requests_and_replies":list(model_calls),
                "raw_qwen_fallbacks":list(fallback_calls),
                "raw_docker_attempts":list(docker_calls),
                "error_class":error_type,"gpu_after":gpu_meta(),
            }
            if (record["trusted_verified"] and
                    record["response_source"] not in (
                      "independent_source_parser_z3","independent_source_python")):
                raise AssertionError("Untrusted model answer promoted")
            if (record["response_source"]=="raw_qwen" and
                    (not fallback_calls or final!=fallback_calls[-1].get("raw_response"))):
                raise AssertionError("PR537 byte-identical fallback violated")
            write_json(out/(case_id+".json"),record)
            records.append(record)
            print("SMOKE_CASE "+json.dumps({
                "id":case_id,"http":status,"strategy":selected,
                "repairs":record["repair_rounds"],"verified":record["trusted_verified"],
                "elapsed_ms":round(elapsed,2),
                "gpu_qlora_proven":tuned_gpu_proven},ensure_ascii=False),flush=True)
    finally:
        adaptive._localhost_transport=actual_transport
        bounded._localhost_transport=actual_bounded_transport
        planner.run_python_in_docker=actual_docker
        backend.respond_chat=actual_fallback
    summary={
      "status":"completed" if len(records)==10 else "partial",
      "source_commit":PIN,"case_count":len(records),"gpu_source":"real self-hosted RTX4070",
      "case_definitions_sha256":manifest["case_definitions_sha256"],
      "aggregate":{"verified_promotions":sum(x["trusted_verified"] for x in records),
                   "correct_reference_matches":sum(x["offline_match"] for x in records),
                   "qlora_calls":sum(x["proposal_calls"] for x in records),
                   "fallback_calls":sum(len(x["raw_qwen_fallbacks"]) for x in records),
                   "docker_calls":sum(len(x["raw_docker_attempts"]) for x in records)},
      "cases":[{k:r[k] for k in ("id","category","expected_route",
        "observed_proposal_strategies","selected_route","repair_rounds",
        "trusted_verified","trusted_verification_status","result_text",
        "computed_scalar","offline_reference","offline_match","elapsed_ms",
        "gpu_loaded_qlora_512mb_proven","http_status")} for r in records],
      "artifact_retention_warning":"GitHub Actions artifact retention is time-limited; not permanent",
    }
    write_json(out/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--authorize",default="")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if (args.live==args.preflight or args.authorize!=RUN_AUTH
            or (args.live and args.out is None)):
        parser.error("Explicit exclusive mode and exact authorization required")
    if args.preflight:
        refs=source_preflight()
        print("SMOKE_CPU_PREFLIGHT_OK cases=8 certified_sources=2 reference_count="+str(len(refs)))
        return
    result=run(args.out)
    print("SMOKE_FINAL_SCALARS "+json.dumps(result["aggregate"],sort_keys=True))
    if result["case_count"]!=10:raise SystemExit(2)


if __name__=="__main__":
    main()
