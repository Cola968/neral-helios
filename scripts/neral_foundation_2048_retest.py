"""Read-only NERAL Qwen3:8B vs Qwen3.5:9B follow-up.

Step A: CPU-only heuristic classification of the last 800 characters from
93 original, source-verified 512-token incomplete responses.
Step B: local offline 2048-token follow-up on ALL identical 72 tasks with
official non-thinking sampling and measured nvidia-smi sampled VRAM peaks.
No training, model download, weight promotion, raw transcript upload or
private-data grading claims. Only scalar aggregates are exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import threading

from scripts import neral_v7_foundation_tournament as original
from scripts.neral_foundation_output_audit import audit_rows as verify_original_rows

MODEL_A="qwen3:8b"
MODEL_B="qwen3.5:9b"
MODELS=(MODEL_A,MODEL_B)
SOURCE_RUN="42499079467"
REVIEW_RUN="42501672292"
GROUP_BOTH_CAPPED="both_previously_capped"
GROUP_ONLY_Q35_CAPPED="only_qwen35_previously_capped"
GROUP_BOTH_FINISHED="both_previously_finished"
EXPECTED_GROUP_COUNTS={
    GROUP_BOTH_CAPPED:31,
    GROUP_ONLY_Q35_CAPPED:31,
    GROUP_BOTH_FINISHED:10,
}
ORIGINAL_DIGESTS={
    MODEL_A:"sha256:500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
    MODEL_B:"sha256:56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90",
}
AUDIT_STATUS="NERAL_FOUNDATION_512_TAIL_HEURISTIC_AUDIT_ONLY"
RETEST_STATUS="NERAL_FOUNDATION_2048_PAIRED_DEVELOPMENT_RETEST_ONLY"
REPETITION="repetition_like"
PROGRESS="structured_progress_candidate"
DIGRESSION="unstructured_or_digression"
CLASSIFICATIONS=(PROGRESS,REPETITION,DIGRESSION)
SEED=20261108
TOKEN_BUDGET=2048
_MAX_ORIGINAL_FILE=4_000_000
_MAX_REVIEW_FILE=1_000_000


class RetestError(ValueError):
    """Reject mismatched original data, unsafe output or inadequate telemetry."""


def _original_index(rows:list[dict]) -> dict:
    if type(rows) is not list or len(rows)!=144:
        raise RetestError("Original source must contain 144 aligned model rows")
    index={}
    for row in rows:
        if (type(row) is not dict or row.get("model") not in MODELS or
                type(row.get("id")) is not str or
                type(row.get("prompt_private")) is not str or
                type(row.get("answer_private")) is not int or
                type(row.get("raw_private")) is not str or
                row.get("done_reason") not in ("length","stop") or
                type(row.get("grade")) is not dict):
            raise RetestError("Invalid original private response row")
        key=(row["model"],row["id"])
        if key in index:
            raise RetestError("Duplicate original model and task")
        index[key]=row
    if len(index)!=144:
        raise RetestError("Incomplete source")
    for _,cid in index:
        if any((m,cid) not in index for m in MODELS):
            raise RetestError("Unpaired original source")
    return index


def compute_original_groups(rows:list[dict]) -> dict[str,list[str]]:
    """Partition by ACTUAL original EOS/length results, never by indexes."""
    index=_original_index(rows)
    groups={name:[] for name in EXPECTED_GROUP_COUNTS}
    # Stable canonical order, independent of model-loop order in private JSONL.
    for cid in sorted({cid for _,cid in index}):
        a,b=(index[(m,cid)] for m in MODELS)
        for k in ("domain","family","prompt_private","answer_private"):
            if a.get(k)!=b.get(k):
                raise RetestError("Unmatched original task/prompt")
        x,y=(r["done_reason"]=="length" for r in (a,b))
        if x and y:
            group=GROUP_BOTH_CAPPED
        elif not x and y:
            group=GROUP_ONLY_Q35_CAPPED
        elif not x and not y:
            group=GROUP_BOTH_FINISHED
        else:
            raise RetestError("Unexpected original Qwen3-only cutoff group")
        groups[group].append(cid)
    if {k:len(v) for k,v in groups.items()}!=EXPECTED_GROUP_COUNTS:
        raise RetestError("Original three groups do not match frozen 512 results")
    return groups


def classify_tail(tail:str)->str:
    """Conservative pattern classification, NOT assessment of correctness.

    Only suffix text is available. A 'progress' label means visible structured
    reasoning; proximity to the true answer cannot be established from it.
    """
    if type(tail) is not str or not 0<len(tail)<=800:
        raise RetestError("Invalid 800-character local suffix")
    normalized=" ".join(tail.lower().split())
    tokens=re.findall(r"[\w]+|[=+\-*%<>]",normalized)
    lines=[re.sub(r"\s+"," ",ln.strip().lower()) for ln in tail.splitlines()]
    lines=[ln for ln in lines if len(ln)>=18]
    if any(cnt>=3 for cnt in Counter(lines).values()):
        return REPETITION
    # Repeated long contiguous phrases indicate obvious loops; do not treat
    # ordinary recurring mathematical operators as degeneration.
    words=re.findall(r"[a-zäöüß0-9]+",normalized)
    for size in (9,7,5):
        if len(words)<size*3:
            continue
        windows=Counter(tuple(words[i:i+size]) for i in range(len(words)-size+1))
        if any(freq>=3 and len(" ".join(phrase))>=25
               for phrase,freq in windows.items()):
            return REPETITION
    numbers=re.findall(r"\d+",tail)
    operators=re.findall(r"(?:\d\s*[-+*%/=]\s*\d|=|→|⇒|->|\bmod\b|\brange\s*\(|\bprint\s*\()",tail,re.I)
    markers=re.findall(
        r"\b(?:step|calculate|compute|then|thus|therefore|verify|check|"
        r"schritt|berechne|daher|damit|also|ergibt|prüfe|"
        r"sum|for|while|if|result|solution|lösung)\b",tail,re.I)
    if ((len(numbers)>=2 and len(operators)>=1)
            or (len(operators)>=2 and len(markers)>=1)
            or (len(markers)>=2 and len(numbers)>=1)):
        return PROGRESS
    return DIGRESSION


def classify_review_ends(review:list[dict],original_rows:list[dict]):
    """Validate every private review suffix against original source before audit."""
    index=_original_index(original_rows)
    compute_original_groups(original_rows)
    if type(review) is not list or len(review)!=93:
        raise RetestError("Exactly 93 original incomplete review rows are required")
    expected_keys={(m,cid) for m,cid in index
                   if index[(m,cid)]["done_reason"]=="length"}
    public={m:Counter() for m in MODELS}
    private=[]
    seen=set()
    expected_fields={
        "case_id","model","prompt_private","gold_private",
        "last_800_chars_private","requires_manual_reasoning_review"
    }
    for row in review:
        if type(row) is not dict or set(row)!=expected_fields:
            raise RetestError("Unexpected local review schema")
        model,cid=row["model"],row["case_id"]
        key=(model,cid)
        if key not in expected_keys or key in seen:
            raise RetestError("Review missing original provenance")
        seen.add(key)
        source=index[key]
        if (row["gold_private"]!=source["answer_private"] or
            row["prompt_private"]!=source["prompt_private"] or
            row["last_800_chars_private"]!=source["raw_private"][-800:] or
            row["requires_manual_reasoning_review"] is not True):
            raise RetestError("Local review data modified or inconsistent")
        category=classify_tail(row["last_800_chars_private"])
        public[model][category]+=1
        private.append({
            "case_id":cid,"model":model,"category_heuristic":category,
            "prompt_private":row["prompt_private"],
            "gold_private":row["gold_private"],
            "last_800_chars_private":row["last_800_chars_private"],
            "needs_human_verification":True
        })
    if seen!=expected_keys:
        raise RetestError("Incomplete private audit review")
    summary={
        "schema":1,"status":AUDIT_STATUS,
        "source_run_id":SOURCE_RUN,"review_run_id":REVIEW_RUN,
        "total_incomplete":93,
        "models":{
            model:{
                **{c:public[model][c] for c in CLASSIFICATIONS},
                "total":sum(public[model].values())
            } for model in MODELS
        },
        "method":"deterministic_suffix_pattern_heuristics_only",
        "heuristics_not_semantic_validation":True,
        "near_goal_not_proven":True,
        "repetition_cause_temperature_zero_not_proven":True,
        "reasoning_correctness_proven":False,
        "raw_responses_uploaded":False,"private_review_uploaded":False,
        "no_gpu_or_model_calls":True,"release_validated":False
    }
    if (summary["models"][MODEL_A]["total"]!=31 or
        summary["models"][MODEL_B]["total"]!=62):
        raise RetestError("Unexpected previous cutoff counts")
    return summary,private


def chat_payload(model:str,prompt:str)->dict:
    """Qwen3/Qwen3.5 official non-thinking general sampling (same two arms)."""
    if model not in MODELS or type(prompt) is not str or not 1<=len(prompt)<=1800:
        raise RetestError("Unknown installed model or unsafe task prompt")
    return {
        "model":model,
        "messages":[{"role":"system","content":original.SYSTEM},
                    {"role":"user","content":prompt}],
        "think":False,"stream":False,"keep_alive":"10m",
        "options":{
            "temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,
            "presence_penalty":1.5,"repeat_penalty":1.0,
            "num_predict":TOKEN_BUDGET,"num_ctx":4096,"seed":SEED
        }
    }


def parse_gpu_memory(line:str)->tuple[int,int]:
    """nvidia-smi GPU-wide allocated MiB; not process-specific or Torch-only."""
    if type(line) is not str or "\n" in line.strip() or "\r" in line.strip():
        raise RetestError("Multiple or invalid GPU sample lines")
    parts=[p.strip() for p in line.split(",")]
    if len(parts)!=3 or "RTX 4070" not in parts[2]:
        raise RetestError("Pinned RTX 4070 not found")
    try:
        used,total=int(parts[0]),int(parts[1])
    except ValueError as exc:
        raise RetestError("Invalid nvidia-smi memory fields") from exc
    if not 0<=used<=total or not 10000<=total<=17000:
        raise RetestError("Invalid 12GB RTX 4070 memory sample")
    return used,total


def _read_gpu_memory()->tuple[int,int]:
    try:
        p=subprocess.run(
            ["nvidia-smi","--query-gpu=memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            check=True,capture_output=True,text=True,timeout=8
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RetestError("nvidia-smi unavailable during mandatory VRAM audit") from exc
    return parse_gpu_memory(p.stdout)


class SampledVram:
    """Sample actual GPU-wide memory every 0.3s while model is executing.

    Report maximum OBSERVED GPU total, not impossible-to-observe instantaneous
    hardware peak. Include model allocation, runtime and any other GPU consumers.
    """
    def __init__(self):
        self.peak_used_mib=0
        self.total_mib=None
        self.sample_count=0
        self.error=None
        self._stop=threading.Event()
        self._thread=None

    def _sample(self):
        used,total=_read_gpu_memory()
        if self.total_mib is not None and self.total_mib!=total:
            raise RetestError("GPU changed while measuring")
        self.total_mib=total
        self.peak_used_mib=max(self.peak_used_mib,used)
        self.sample_count+=1

    def _worker(self):
        while not self._stop.wait(0.30):
            try:
                self._sample()
            except Exception:
                self.error="VRAM sampler lost required observations"
                return

    def __enter__(self):
        self._sample()
        self._thread=threading.Thread(target=self._worker,daemon=True)
        self._thread.start()
        return self

    def __exit__(self,*_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        if self._thread is not None and self._thread.is_alive():
            raise RetestError("VRAM sampler failed to terminate")
        if self.error:
            raise RetestError(self.error)
        self._sample()
        if self.sample_count<3:
            raise RetestError("Too few real GPU memory samples")

    def report(self):
        if not self.sample_count or self.total_mib is None:
            raise RetestError("Missing GPU peak telemetry")
        return {
            "peak_used_mib":self.peak_used_mib,
            "total_mib":self.total_mib,
            "sample_count":self.sample_count
        }


def summarise_retest(groups:dict,responses:dict,per_model_vram:dict)->dict:
    if (type(groups) is not dict or set(groups)!=set(EXPECTED_GROUP_COUNTS) or
        any(type(v) is not list or len(v)!=EXPECTED_GROUP_COUNTS[k]
            for k,v in groups.items()) or
        len(set().union(*map(set,groups.values())))!=72):
        raise RetestError("Incomplete frozen source group manifest")
    if type(responses) is not dict or set(responses)!=set(MODELS):
        raise RetestError("Missing model arm")
    if type(per_model_vram) is not dict or set(per_model_vram)!=set(MODELS):
        raise RetestError("Missing per-model VRAM peak")
    expected_ids=set().union(*map(set,groups.values()))
    result_models={}
    for model in MODELS:
        arm=responses[model]
        if type(arm) is not dict or set(arm)!=expected_ids:
            raise RetestError("Missing, duplicate, or mismatched response")
        gpu=per_model_vram[model]
        if (type(gpu) is not dict or
            set(gpu)!={"peak_used_mib","total_mib","sample_count"} or
            any(type(gpu[k]) is not int for k in gpu) or
            not 0<gpu["peak_used_mib"]<=gpu["total_mib"] or
            not 10000<=gpu["total_mib"]<=17000 or gpu["sample_count"]<3):
            raise RetestError("Invalid mandatory VRAM samples")
        by_group={}
        total_tokens=0
        total_duration=0
        for group,ids in groups.items():
            counts=Counter()
            for cid in ids:
                item=arm[cid]
                if (type(item) is not dict or
                    set(item)!={"status","done_reason","eval_count","duration_ns"} or
                    item["status"] not in
                        ("correct","wrong_final","ungraded","incomplete") or
                    item["done_reason"] not in ("stop","length") or
                    (item["status"]=="incomplete")!=(item["done_reason"]=="length") or
                    type(item["eval_count"]) is not int or
                    not 0<=item["eval_count"]<=TOKEN_BUDGET or
                    type(item["duration_ns"]) is not int or
                    not 0<=item["duration_ns"]<=10**15):
                    raise RetestError("Invalid response grade or generation metrics")
                counts[item["status"]]+=1
                total_tokens+=item["eval_count"]
                total_duration+=item["duration_ns"]
            by_group[group]={
                "total":len(ids),
                "finished":len(ids)-counts["incomplete"],
                "correct":counts["correct"],
                "wrong_final":counts["wrong_final"],
                "ungraded":counts["ungraded"],
                "incomplete":counts["incomplete"]
            }
        result_models[model]={
            "by_original_group":by_group,
            "correct":sum(x["correct"] for x in by_group.values()),
            "finished":sum(x["finished"] for x in by_group.values()),
            "incomplete":sum(x["incomplete"] for x in by_group.values()),
            "total_new_tokens":total_tokens,
            "total_duration_seconds":round(total_duration/1e9,3),
            "sampled_peak_vram_mib":gpu["peak_used_mib"],
            "sampled_peak_vram_gib":round(gpu["peak_used_mib"]/1024,3),
            "gpu_total_vram_mib":gpu["total_mib"],
            "vram_observations":gpu["sample_count"],
        }
    return {
        "schema":1,"status":RETEST_STATUS,"original_source_run_id":SOURCE_RUN,
        "n_per_model":72,"original_group_counts":EXPECTED_GROUP_COUNTS,
        "generation_budget":TOKEN_BUDGET,
        "sampling":{"think":False,"temperature":0.7,"top_p":0.8,
                    "top_k":20,"min_p":0.0,"presence_penalty":1.5,
                    "repeat_penalty":1.0,"num_ctx":4096,"seed":SEED},
        "models":result_models,
        "memory_telemetry":"sampled_gpu_wide_used_memory_not_instantaneous_process_peak",
        "no_model_downloads":True,"training_performed":False,
        "raw_prompts_or_answers_uploaded":False,
        "release_validated":False,"model_superiority_proven":False,
        "adapter_promoted":False,
        "limitations":(
            "Developer-authored 72-case original suite has previously been opened. "
            "Same prompts and decoding policies, but differing model architecture "
            "and GGUF quantization; 2048 output cap still censors some responses. "
            "Sampled GPU-wide maximum (0.3s nominal interval) includes unrelated "
            "allocations, may miss brief peaks and is NOT process-specific. "
            "Matched seed cannot make probabilistic decoding identical. "
            "No blinded-generalization claim."
        )
    }


def _load_json(path:Path,max_bytes:int)->object:
    if (path.is_symlink() or not path.is_file() or
        not 1<=path.stat().st_size<=max_bytes):
        raise RetestError("Missing, symlinked or oversized private source")
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (ValueError,UnicodeError) as exc:
        raise RetestError("Corrupt private manifest") from exc


def _read_jsonl(path:Path,max_bytes:int)->list[dict]:
    if (path.is_symlink() or not path.is_file() or
        not 1<=path.stat().st_size<=max_bytes):
        raise RetestError("Missing, symlinked or oversized private source")
    try:
        return [json.loads(line) for line in
                path.read_text(encoding="utf8").splitlines()]
    except (ValueError,UnicodeError) as exc:
        raise RetestError("Corrupt private JSONL") from exc


def _source_rows(source:Path)->tuple[list[dict],list[dict]]:
    if source.name != f"v7-foundation-local-{SOURCE_RUN}-1":
        raise RetestError("Only original pinned source-run directory is accepted")
    rows=_read_jsonl(source/"private-responses.jsonl",_MAX_ORIGINAL_FILE)
    manifest=_load_json(source/"private-manifest.json",20000)
    aggregate=_load_json(source/"scalar-summary.json",50000)
    cases=original.generate_cases()
    digest=original.task_digest(cases)
    if (type(manifest) is not dict or
        manifest.get("suite_sha256")!=digest or
        manifest.get("task_count")!=72 or
        manifest.get("models")!=ORIGINAL_DIGESTS or
        manifest.get("think") is not False or
        manifest.get("num_predict")!=512 or
        manifest.get("temperature")!=0 or
        type(aggregate) is not dict or
        aggregate.get("foundation_digests")!=ORIGINAL_DIGESTS or
        aggregate.get("count_per_model")!=72):
        raise RetestError("Pinned previous source/decoder provenance mismatch")
    try:
        verify_original_rows(rows)
    except ValueError as exc:
        raise RetestError("Original completed scores disagree with 512 audit") from exc
    index=_original_index(rows)
    for case in cases:
        for model in MODELS:
            row=index[(model,case["id"])]
            if any(row[k]!=case[c] for k,c in (
                    ("domain","domain"),("family","family"),
                    ("prompt_private","prompt"),("answer_private","answer"))):
                raise RetestError("Source prompts no longer match original 72-case suite")
    compute_original_groups(rows)
    return rows,cases


def _write_new(path:Path,contents:str)->None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise RetestError("Output must be new in an existing non-symlink directory")
    with path.open("x",encoding="utf8") as f:
        f.write(contents)


def run_audit(source:Path,review_path:Path,public_out:Path,private_out:Path):
    rows,_=_source_rows(source)
    if review_path.name!=f"foundation-private-512-review-{REVIEW_RUN}-1.jsonl":
        raise RetestError("Only pinned original private review is allowed")
    review=_read_jsonl(review_path,_MAX_REVIEW_FILE)
    summary,details=classify_review_ends(review,rows)
    if public_out.resolve()==private_out.resolve():
        raise RetestError("Private review must not be same as public aggregate")
    if public_out.exists() or private_out.exists():
        raise RetestError("Audit output exists")
    _write_new(private_out,"".join(json.dumps(x,ensure_ascii=False)+"\n"
                                    for x in details))
    _write_new(public_out,json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,sort_keys=True))
    return summary


def _client_chat(client:original.OllamaLocal,model:str,case:dict)->dict:
    request=chat_payload(model,case["prompt"])
    reply=client._request("/api/chat",request)
    message=reply.get("message")
    raw=message.get("content") if type(message) is dict else None
    reason=reply.get("done_reason")
    count=reply.get("eval_count")
    duration=reply.get("total_duration")
    if (type(raw) is not str or len(raw)>64000 or
        reason not in ("stop","length") or
        type(count) is not int or not 0<=count<=TOKEN_BUDGET or
        type(duration) is not int or not 0<=duration<=10**15):
        raise RetestError("Ollama response has incomplete telemetry or is malformed")
    grade=original.evaluate_response(case,raw,complete=(reason=="stop"))
    if grade["status"]=="incomplete" and reason!="length":
        raise RetestError("Wrong completion marker")
    return {
        "id":case["id"],"domain":case["domain"],"family":case["family"],
        "model":model,"prompt_private":case["prompt"],"answer_private":case["answer"],
        "raw_private":raw,"done_reason":reason,"grade":grade,
        "eval_count":count,"duration_ns":duration
    }


def _unload(client:original.OllamaLocal,model:str):
    # Local-only Ollama /api/chat empty message + keep_alive=0, per documentation.
    data=client._request("/api/chat",{
        "model":model,"messages":[],"stream":False,"keep_alive":0
    })
    if type(data) is not dict or data.get("done") is not True:
        raise RetestError("Ollama model unload not acknowledged")


def run_retest(source:Path,out_dir:Path):
    rows,cases=_source_rows(source)
    groups=compute_original_groups(rows)
    if out_dir.exists() or out_dir.is_symlink() or not out_dir.parent.is_dir():
        raise RetestError("Require new local private directory outside checkout")
    client=original.OllamaLocal(timeout=240)
    actual=client.inventory()
    if any(actual.get(model)!=ORIGINAL_DIGESTS[model] for model in MODELS):
        raise RetestError("Installed Ollama model SHA changed since original trial")
    preused,total=_read_gpu_memory()
    if total-preused < 7000:
        raise RetestError("RTX 4070 GPU is not free enough for isolated retest")

    out_dir.mkdir(exist_ok=False)
    private_path=out_dir/"private-2048-responses.jsonl"
    # Create-only private record file. Incomplete runs have NO public scores.
    answers={}
    vram={}
    with private_path.open("x",encoding="utf8") as fout:
        for model in MODELS:
            previous_memory,_=_read_gpu_memory()
            if total-previous_memory<7000:
                raise RetestError("GPU may still be busy; refuse simultaneous models")
            response_map={}
            with SampledVram() as sampler:
                try:
                    for case in cases:
                        record=_client_chat(client,model,case)
                        response_map[case["id"]]={
                            "status":record["grade"]["status"],
                            "done_reason":record["done_reason"],
                            "eval_count":record["eval_count"],
                            "duration_ns":record["duration_ns"]
                        }
                        fout.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")
                        fout.flush()
                finally:
                    _unload(client,model)
            answers[model]=response_map
            vram[model]=sampler.report()
    if client.inventory()!=actual:
        raise RetestError("Installed Ollama models changed during read-only run")
    report=summarise_retest(groups,answers,vram)
    report["model_digests"]=ORIGINAL_DIGESTS
    report["source_suite_sha256"]=original.task_digest(cases)
    _write_new(out_dir/"scalar-summary.json",
               json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":report["status"],"n_per_model":report["n_per_model"],
        "models":report["models"],
        "memory_telemetry":report["memory_telemetry"],
        "raw_prompts_or_answers_uploaded":False,
        "release_validated":False
    },sort_keys=True))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("audit","retest"),required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--private-review",type=Path)
    p.add_argument("--public-out",type=Path)
    p.add_argument("--private-out",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args(argv)
    if args.mode=="audit":
        if not all((args.private_review,args.public_out,args.private_out)) or args.out:
            p.error("Audit requires private-review, public-out and private-out")
        run_audit(args.source,args.private_review,args.public_out,args.private_out)
    else:
        if not args.out or any((args.private_review,args.public_out,args.private_out)):
            p.error("Retest requires --out only")
        run_retest(args.source,args.out)

if __name__=="__main__":
    main()
