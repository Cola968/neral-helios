"""One-shot fair prospective Blind100 evaluation for unmodified PR542 source.

The task generator, TIR, Docker sandbox, NL-to-Z3 route and PR537 guard
are imported from the pinned PR542 source checkout; only this harness lives
on the isolated verification branch. No task text or solver raw proposal is
ever printed to stdout. Model raw replies are persisted locally AND uploaded
only in the explicitly authorized restricted GitHub Actions artifact.

Modes: --cpu-dry-run, --seal, --gpu, --regrade. Never merge or train.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

SOURCE_SHA="737413f03b347676108d507ff3fe7693975dd269"
OLD_SHA="74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258"
GENERATOR_SHA="477d78aa72a2f52f1b4b37bea6f6c3afa8ffabbce41b44d4743215cdf7487fcd"
NEW_GENERATOR_SHA="6842cdd9475c8a891b22d838191fbe19978ecf938942b950b77377af302b344c"
BASE_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
GUARD_BLOB="b87c594ee54a20b2209b6bf176bc0aacfe256ce8"
TUNED_TAG="neral-qwen35-qlora-v1:latest"
TUNED_DIGEST="9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8"
BASE_TAG="qwen3.5:9b"
CATEGORIES=("mathematics","logic","programming","mixed")
FORMAT_SUFFIX="Gib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."
SYSTEM=("You are a precise local assistant. Answer the question accurately. "
        "You may explain your reasoning. End with one line FINAL_ANSWER: <integer>. "
        "Use an integer, no units in the final line.")
SEED=1511261847
AUTH_GPU="NERAL_PR542_V2_FRESH_SEALED100_GPU_ONCE"
STRICT=re.compile(r"\s*(?:(?:FINAL_ANSWER|NERAL_FINAL_TEXT):\s*)?(-?\d{1,10})\s*")

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def make_cases(seed:int)->tuple[list[dict],str]:
    """Reuse pre-PR542 generator without editing it; check old exact/normalized sets."""
    from scripts import neral_blind_100_cases as oldgen
    from scripts import neral_pr542_fresh100_v2_cases as gen
    from scripts.neral_generalization_100_cases import build as older, normalized, sha
    old_bytes=Path(oldgen.__file__).read_bytes().replace(b"\r\n",b"\n")
    fresh_bytes=Path(gen.__file__).read_bytes().replace(b"\r\n",b"\n")
    if (hashlib.sha256(old_bytes).hexdigest()!=GENERATOR_SHA or
        hashlib.sha256(fresh_bytes).hexdigest()!=NEW_GENERATOR_SHA):
        raise RuntimeError("Immutable old/fresh generator sources mismatch")
    old=oldgen.build()
    old_source_hash=hashlib.sha256(oldgen.serialize(old)).hexdigest()
    if old_source_hash!=OLD_SHA:
        raise RuntimeError("Old sealed reference not reproduced")
    fresh=gen.build(seed=seed)
    blob=gen.serialize(fresh)
    digest=hashlib.sha256(blob).hexdigest()
    if digest==OLD_SHA or len(fresh)!=100 or Counter(x["category"] for x in fresh)!={c:25 for c in CATEGORIES}:
        raise RuntimeError("New dataset is not fresh balanced 100")
    previous=old+older()
    exact={sha(x["question"]) for x in previous}
    normalized_set={normalized(x["question"]) for x in previous}
    if any(sha(x["question"]) in exact or normalized(x["question"]) in normalized_set
           for x in fresh):
        raise RuntimeError("Previously evaluated question was regenerated")
    # The templates are previously published. This is a fresh draw, not a
    # previously unseen *task family* nor a new external benchmark.
    return fresh,digest

def verify_manifest(manifest:dict,cases:list[dict],digest:str)->None:
    if (type(manifest) is not dict or manifest.get("source_commit")!=SOURCE_SHA
        or manifest.get("old_sealed_sha256")!=OLD_SHA
        or manifest.get("dataset_sha256")!=digest
        or manifest.get("generator_source_sha256")!=NEW_GENERATOR_SHA
        or manifest.get("historical_generator_sha256")!=GENERATOR_SHA
        or manifest.get("seed")!=SEED
        or manifest.get("count")!=100
        or manifest.get("categories")!={x:25 for x in CATEGORIES}):
        raise RuntimeError("Prospective frozen manifest mismatch")
    if len(cases)!=100:
        raise RuntimeError("Not exactly 100")

def prompt_for(case:dict)->str:
    return case["question"]+"\n"+FORMAT_SUFFIX

def messages_for(case:dict)->list[dict]:
    return [{"role":"system","content":SYSTEM},
            {"role":"user","content":prompt_for(case)}]

def append_durable(path:Path,payload:dict)->None:
    # Filename is created with exclusive 'x' before first inference.
    with path.open("a",encoding="utf8",newline="\n") as f:
        f.write(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
        f.flush()
        os.fsync(f.fileno())

def records(path:Path,expect:int)->list[dict]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Raw evidence missing or link")
    result=[]
    with path.open("r",encoding="utf8") as f:
        for line in f:
            if line.strip():
                item=json.loads(line)
                if type(item) is not dict:
                    raise RuntimeError("Malformed evidence row")
                result.append(item)
    if len(result)!=expect:
        raise RuntimeError("Raw archive incomplete: expected "+str(expect)+" records")
    percase={}
    for item in result:
        case=item.get("id")
        arm=item.get("arm")
        if (type(case) is not str or arm not in ("qwen","neral")
            or type(item.get("reply")) not in (str,type(None))
            or type(item.get("proof")) is not dict):
            raise RuntimeError("Malformed persisted arm")
        percase.setdefault(case,set())
        if arm in percase[case]:raise RuntimeError("Duplicate stored arm")
        percase[case].add(arm)
    if len(percase)!=expect//2 or any(a!={"qwen","neral"} for a in percase.values()):
        raise RuntimeError("Not exactly two arms for every task")
    return result

def extract(text:object)->tuple[int|None,str]:
    # Fixed, gold-independent, same interpretation for both arms.
    # The previously committed offline parser handles final prose/boxed/math.
    from scripts.neral_pr542_offline_regrade_saved_raw import extract as prior
    return prior(text)

def classify_proof(body:dict,route:str)->dict:
    if route=="logic_z3":
        yes=(body.get("selection_policy")=="independently_source_verified_nl"
             and body.get("proof_backend")=="z3_unsat"
             and body.get("verified") is True
             and body.get("safe_to_promote") is True
             and body.get("selected_extractor") in ("qlora","base"))
        return {"z3":bool(yes),"sandbox":False,
                "scope":"independent_full_source" if yes else "none",
                "selected_extractor":body.get("selected_extractor") if yes else None}
    sig=(body.get("verification_status"),body.get("verification_scope"),
         body.get("review_method"))
    yes=(body.get("selection_policy")=="independently_verified_tir"
         and body.get("verified") is True
         and body.get("model_generated") is False
         and body.get("tool_used")=="neral_python_sandbox"
         and body.get("tool_calls_succeeded")==1
         and sig in (("verified","exact_user_python_execution","exact_user_ast_match"),
                     ("verified","proven_integer_ast_equivalence","semantic_execution_match")))
    formal=(body.get("selection_policy")=="independently_verified_tir"
            and body.get("verification_status")=="formal_schema_exact"
            and body.get("verification_scope")=="explicit_finite_domain_schema"
            and body.get("tool_used")=="neral_logic_exact")
    return {"z3":bool(formal),"sandbox":bool(yes),
            "scope":"exact_source_python" if yes else ("formal_schema" if formal else "none"),
            "selected_extractor":None}

def row(case:dict,arm:str,reply:str|None,seconds:float,
        *,decision:str,proof:dict,error:str|None=None,details:dict|None=None)->dict:
    if arm not in ("qwen","neral"):raise ValueError("Invalid arm")
    result={
      "id":case["id"],"category":case["category"],"family":case["family"],
      "question_sha256":case["question_sha256"],
      # An independent offline checker needs immutable gold without task text.
      "gold_integer":case["gold_integer"],"arm":arm,"reply":reply,
      "seconds":round(seconds,3),"decision":decision,"proof":proof,
      "error_code":error,"source_commit":SOURCE_SHA,
      "extra":details or {},
    }
    value,method=extract(reply)
    if decision=="explicit_safe_refusal" or error is not None:
        value,method=None,"refusal_or_error"
    result["extracted_integer"]=value
    result["extraction_method"]=method
    result["correct"]=value==case["gold_integer"] if value is not None else False
    return result

def summarize(raw:list[dict],*,sealed_hash:str)->dict:
    if len(raw)!=200:raise RuntimeError("No partial summary")
    if any(r["source_commit"]!=SOURCE_SHA for r in raw):
        raise RuntimeError("Unexpected source revision")
    pair={}
    for item in raw:
        key=item["id"]
        if item["arm"] in pair.setdefault(key,{}):
            raise RuntimeError("Duplicate record")
        pair[key][item["arm"]]=item
    if len(pair)!=100 or any(set(x)!={"qwen","neral"} for x in pair.values()):
        raise RuntimeError("Unpaired records")
    cats={};regression_ids=[];gained_ids=[]
    proof_z3=proof_sandbox=0
    policies=Counter()
    ambiguous=Counter()
    for c in CATEGORIES:
        pairs=[v for v in pair.values() if v["qwen"]["category"]==c
               and v["neral"]["category"]==c]
        if len(pairs)!=25:raise RuntimeError("Category mismatch")
        b=sum(v["qwen"]["correct"] for v in pairs)
        n=sum(v["neral"]["correct"] for v in pairs)
        gains=[v["neral"]["id"] for v in pairs if v["neral"]["correct"] and not v["qwen"]["correct"]]
        losses=[v["neral"]["id"] for v in pairs if v["qwen"]["correct"] and not v["neral"]["correct"]]
        gained_ids.extend(gains);regression_ids.extend(losses)
        cats[c]={"qwen":b,"neral":n,"gained":len(gains),"regressions":len(losses),"count":25}
        proof_z3+=sum(bool(v["neral"]["proof"]["z3"]) for v in pairs)
        proof_sandbox+=sum(bool(v["neral"]["proof"]["sandbox"]) for v in pairs)
        policies.update(v["neral"]["decision"] for v in pairs)
        for v in pairs:
            for arm in ("qwen","neral"):
                if v[arm]["extracted_integer"] is None:
                    ambiguous[arm]+=1
    B=sum(v["qwen"] for v in cats.values())
    N=sum(v["neral"] for v in cats.values())
    base_s=[v["qwen"]["seconds"] for v in pair.values()]
    # NERAL end-to-end latency includes measuring Qwen cached fallback.
    end_s=[v["qwen"]["seconds"]+v["neral"]["seconds"] for v in pair.values()]
    out={
      "status":"complete_offline_regrade_from_persisted_raw",
      "source_commit":SOURCE_SHA,"sealed_sha256":sealed_hash,"count":100,
      "categories":cats,"baseline_correct":B,"candidate_correct":N,
      "net_gain":N-B,"gained_cases":len(gained_ids),"regressions":len(regression_ids),
      "regression_ids":sorted(regression_ids),"gain_ids":sorted(gained_ids),
      "z3_verified_count":proof_z3,
      "sandbox_verified_count":proof_sandbox,
      "sandbox_verified_rate":proof_sandbox/100,
      "selection_counts":dict(policies),
      "unscorable_counts":dict(ambiguous),
      "baseline_latency_median_s":round(statistics.median(base_s),3),
      "baseline_latency_max_s":round(max(base_s),3),
      "neral_end_to_end_latency_median_s":round(statistics.median(end_s),3),
      "neral_end_to_end_latency_max_s":round(max(end_s),3),
      "release_gate_passed":bool(N-B>=8 and not regression_ids
                                 and ambiguous.get("qwen",0)==0
                                 and ambiguous.get("neral",0)==0),
      "release_requirements":"net>=8, zero regressions, zero unscorable answers",
      "raw_records":len(raw),"raw_uploaded":True,"merged_to_main":False,
      "note":"Synthetic template families are preexisting; only case instances newly seeded. Zero test regressions is not a universal guarantee.",
    }
    out["z3_source_verified_logic_count"]=sum(
        1 for v in pair.values()
        if v["neral"]["category"]=="logic" and v["neral"]["proof"]["z3"])
    out["z3_source_verified_logic_rate"]=out["z3_source_verified_logic_count"]/25
    return out

def dry_run(out:Path)->None:
    if out.exists():raise FileExistsError("CPU dry-run evidence exists")
    out.mkdir(parents=True)
    dummy=[
      {"id":f"synthetic-{j}","category":"mathematics","family":"dummy",
       "question_sha256":hashlib.sha256(f"synthetic-{j}".encode()).hexdigest(),
       "gold_integer":j+4} for j in range(2)]
    raw=out/"raw-200.jsonl"
    with raw.open("x",encoding="utf8",newline="\n"):pass
    for j,c in enumerate(dummy):
        append_durable(raw,row(c,"qwen",f"The answer is {j+4}.",0.001,
                               decision="raw_qwen",proof={"z3":False,"sandbox":False}))
        append_durable(raw,row(c,"neral",f"FINAL_ANSWER: {j+4}",0.002,
                               decision="independently_verified_tir",
                               proof={"z3":False,"sandbox":True}))
    reread=records(raw,4)
    assert len(reread)==4 and all(r["correct"] for r in reread)
    receipt={"status":"cpu_dummy_persisted_and_reread","records":4,
             "sha256":sha256_file(raw),"source_commit":SOURCE_SHA,
             "no_models_called":True,"no_gpu_used":True}
    with (out/"dummy-receipt.json").open("x",encoding="utf8") as f:
        json.dump(receipt,f,sort_keys=True,indent=2)
    print("CPU_DUMMY_ARCHIVE_OK records=4 sha256="+receipt["sha256"])

def seal(out:Path)->None:
    if out.exists():raise FileExistsError("Cannot replace existing seal")
    cases,digest=make_cases(SEED)
    manifest={"source_commit":SOURCE_SHA,"seed":SEED,"count":100,
              "categories":{c:25 for c in CATEGORIES},
              "dataset_sha256":digest,"old_sealed_sha256":OLD_SHA,
              "generator_source_sha256":NEW_GENERATOR_SHA,
              "historical_generator_sha256":GENERATOR_SHA,
              "reused_task_families":False,"task_text_uploaded":False,
              "created_before_gpu":True}
    out.mkdir(parents=True)
    # No prompts or oracle values in GitHub workflow artifacts.
    with (out/"new-seal.json").open("x",encoding="utf8",newline="\n") as f:
        json.dump(manifest,f,sort_keys=True,indent=2);f.write("\n")
    verify_manifest(manifest,cases,digest)
    print("FRESH100_SEALED_SHA256 "+digest)
    print("FRESH100_COUNTS mathematics=25 logic=25 programming=25 mixed=25")

def measure(out:Path,manifest:dict)->None:
    from neral_agent_autonomy import OllamaLocalBackend,OllamaCompletionError
    from neral_tir_experimental import run_experimental_tir
    from neral_tir_regression_guard import choose_guarded_tir_reply
    from neral_nl_verified_route import run_guarded_nl_bridge
    from neral_nl_local_ollama import make_local_extractor
    from scripts.neral_qlora_holdout_live import request,gpu_offload_mib
    from scripts.neral_generalization_100_run import require_exact_guard
    cases,digest=make_cases(SEED)
    verify_manifest(manifest,cases,digest)
    if require_exact_guard()!=GUARD_BLOB:
        raise RuntimeError("PR537 guard changed")
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                       capture_output=True,text=True,check=True,timeout=15).stdout
    if "RTX 4070" not in gpu:
        raise RuntimeError("Explicit RTX4070 requirement failed")
    inventory=request("GET","/api/tags").get("models")
    modelmap={v.get("name"):v.get("digest") for v in inventory if type(v) is dict}
    if modelmap.get(TUNED_TAG)!=TUNED_DIGEST or modelmap.get(BASE_TAG)!=BASE_DIGEST:
        raise RuntimeError("Pinned model digest/untuned baseline unavailable")
    high=OllamaLocalBackend(BASE_TAG,inference_profile="measured_nonthinking_2048")
    raw_path=out/"raw-200.jsonl"
    if raw_path.exists():raise FileExistsError("Refuse rerun of this one-shot archive")
    with raw_path.open("x",encoding="utf8",newline="\n"):pass
    for ix,case in enumerate(cases,1):
        original=messages_for(case)
        high.set_benchmark_seed(0)
        started=time.monotonic()
        base_text=None;base_error=None
        try:
            base_text=high.respond_chat(original)
        except OllamaCompletionError as exc:
            base_error=exc.reason
        base_s=time.monotonic()-started
        if gpu_offload_mib(BASE_TAG)<4096:
            raise RuntimeError("Baseline did not execute on RTX4070")
        if base_error is not None:
            base_text=None
        append_durable(raw_path,row(case,"qwen",base_text,base_s,
          decision="raw_qwen",proof={"z3":False,"sandbox":False},
          error=base_error))
        def cached_base(turns):
            if turns!=original:raise ValueError("Fallback may access original Qwen prompt only")
            if base_text is None:
                raise OllamaCompletionError(base_error or "empty_answer")
            return base_text
        def tuned_respond(turns):
            if turns==original:
                return cached_base(turns)
            packet={"model":TUNED_TAG,"messages":turns,"stream":False,"think":False,
                    "keep_alive":"5m","options":{"temperature":0,"num_predict":2048,
                                               "num_ctx":4096,"seed":0}}
            data=request("POST","/api/chat",packet,timeout=180)
            msg=data.get("message")
            content=msg.get("content") if type(msg) is dict else None
            if (data.get("done") is not True or data.get("done_reason")!="stop"
                or type(content) is not str or not content.strip()
                or len(content.encode("utf8"))>32768):
                raise OllamaCompletionError(
                    "token_limit" if data.get("done_reason")=="length" else "incomplete")
            return content
        started=time.monotonic()
        body={};candidate_error=None
        route="logic_z3" if case["category"]=="logic" else "tir_sandbox_pr537"
        try:
            if route=="logic_z3":
                body=run_guarded_nl_bridge(original,
                    make_local_extractor(TUNED_TAG),make_local_extractor(BASE_TAG),
                    cached_base)
            else:
                proposal=run_experimental_tir(original,tuned_respond)
                body=choose_guarded_tir_reply(proposal,original,cached_base)
            candidate_text=body.get("text")
            if type(candidate_text) is not str or not candidate_text.strip():
                raise RuntimeError("empty_candidate_answer")
            # PR537-guarded fallback must be byte-identical to tested Qwen raw.
            if str(body.get("selection_policy","")).startswith("raw_qwen"):
                if candidate_text!=base_text:
                    raise RuntimeError("guard_fallback_not_identical")
        except (OllamaCompletionError,RuntimeError,ValueError,TypeError) as exc:
            candidate_error=(exc.reason if isinstance(exc,OllamaCompletionError)
                             else type(exc).__name__)
            candidate_text=None
            body={"selection_policy":"explicit_safe_refusal","verified":False,
                  "safe_to_promote":False}
        candidate_s=time.monotonic()-started
        proof=classify_proof(body,route)
        decision=body.get("selection_policy","unknown")
        details={
          "route":route,"verification_status":body.get("verification_status"),
          "verification_scope":body.get("verification_scope"),
          "tool_used":body.get("tool_used"),
          "safe_to_promote":body.get("safe_to_promote") is True,
          "verified":body.get("verified") is True,
          "proof_backend":body.get("proof_backend"),
          "model_generated":body.get("model_generated") is True,
          "selected_extractor":body.get("selected_extractor") if proof["z3"] else None,
          "tool_calls_succeeded":body.get("tool_calls_succeeded"),
        }
        append_durable(raw_path,row(case,"neral",candidate_text,candidate_s,
          decision=decision,proof=proof,error=candidate_error,details=details))
        print("FRESH100_PROGRESS "+str(ix)+"/100",flush=True)
    high.set_benchmark_seed(None)
    full=records(raw_path,200)
    if Counter(r["category"] for r in full)!={x:50 for x in CATEGORIES}:
        raise RuntimeError("Incomplete category records")
    with (out/"inference-complete.json").open("x",encoding="utf8") as f:
        json.dump({"records":200,"source_commit":SOURCE_SHA,
                   "dataset_sha256":digest,"raw_sha256":sha256_file(raw_path),
                   "no_training":True,"main_unchanged":True},f,sort_keys=True,indent=2)
    print("FRESH100_RAW_PERSISTED "+sha256_file(raw_path))

def regrade(out:Path,manifest:dict)->None:
    raw=out/"raw-200.jsonl"
    saved=records(raw,200)
    cases,digest=make_cases(SEED)
    verify_manifest(manifest,cases,digest)
    expected={x["id"]:x for x in cases}
    for v in saved:
        c=expected.get(v["id"])
        if (c is None or v["question_sha256"]!=c["question_sha256"]
            or v["gold_integer"]!=c["gold_integer"]
            or v["category"]!=c["category"]):
            raise RuntimeError("Persisted file does not match sealed dataset")
        # Independently re-run the exact fixed extractor from DISK, no models.
        val,method=extract(v["reply"])
        if v["decision"]=="explicit_safe_refusal" or v["error_code"]:
            val,method=None,"refusal_or_error"
        if val!=v["extracted_integer"] or method!=v["extraction_method"]:
            raise RuntimeError("Posthoc extraction differs from prespecified scorer")
        correct=(val==v["gold_integer"]) if val is not None else False
        if correct != v["correct"]:
            raise RuntimeError("Posthoc grading differs")
    report=summarize(saved,sealed_hash=digest)
    report.update({"raw_sha256":sha256_file(raw),"offline_reread":True,
       "source_run_id":os.environ.get("GITHUB_RUN_ID"),
       "model_weights_changed":False,"case_contents_uploaded":False})
    with (out/"summary.json").open("x",encoding="utf8",newline="\n") as f:
        json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
    print("FRESH100_OFFLINE_SCALARS "+json.dumps(report,sort_keys=True),flush=True)

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    for x in ("cpu-dry-run","seal","gpu","regrade"):
        g.add_argument("--"+x,action="store_true")
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--manifest",type=Path)
    ap.add_argument("--authorize",default="")
    args=ap.parse_args()
    if args.cpu_dry_run:dry_run(args.out);return
    if args.seal:seal(args.out);return
    if args.manifest is None or not args.manifest.is_file():
        raise RuntimeError("Original immutable pre-GPU seal absent")
    man=json.loads(args.manifest.read_text(encoding="utf8"))
    if args.gpu:
        if args.authorize!=AUTH_GPU:raise RuntimeError("One-shot GPU authorization absent")
        if not args.out.exists() or list(args.out.iterdir()):
            raise RuntimeError("One-shot local evidence directory not empty")
        measure(args.out,man)
    else:
        regrade(args.out,man)

if __name__=="__main__":main()
