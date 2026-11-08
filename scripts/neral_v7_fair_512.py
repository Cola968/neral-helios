"""Matched Qwen2.5-Coder-7B-Instruct / NERAL v5 / NERAL v7 evaluation.

Research diagnostic, never a release benchmark. Reuses immutable 72-case
developer suite. 24-case stratified PILOT uses three cases per family; FULL
uses all 72. Same exact chat messages, greedy decoding and 512 generated
tokens. Every answer and per-case ID stays on the private Windows runner.
No training, remote services, tool assistance or model promotion.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

from scripts import neral_v7_logic_compare as old
from scripts.neral_v7_logic_pilot import evaluation_cases
from scripts.neral_release_gate import exact_mcnemar_p

MAX_NEW_TOKENS=512
MAX_CHARS=16384
AUTH="EVALUATE_NERAL_V7_FAIR_512_READONLY"
STATUS="NERAL_V7_FAIR_EOS_512_DEV_DIAGNOSTIC"
ARMS=("qwen","v5","candidate")
BRIEF_SYSTEM=(
    "Answer this integer math or reasoning question. Use at most 90 tokens "
    "of explanation. End with exactly one line FINAL_ANSWER: <integer>. "
    "No tool calls. If uncertain, still supply the best integer answer."
)
_ALLOWED_MODES=("pilot","full")
_STOP_TOKENS=("<|im_end|>","<|endoftext|>")


def select_cases(cases:list[dict],mode:str)->list[dict]:
    if mode not in _ALLOWED_MODES or type(cases) is not list or len(cases)!=72:
        raise ValueError("Expected frozen 72-case logic suite and valid evaluation mode")
    if mode=="full":
        result=list(cases)
    else:
        counts={}
        result=[]
        for case in cases:
            family=case["family"]
            counts[family]=counts.get(family,0)+1
            if counts[family]<=3:
                result.append(case)
    expected=24 if mode=="pilot" else 72
    expected_parts={"retention":12,"transfer":12} if mode=="pilot" else {
        "retention":24,"transfer":48}
    if (len(result)!=expected or len({r["id"] for r in result})!=expected
            or {s:sum(r.get("part")==s for r in result)
                for s in expected_parts}!=expected_parts
            or len({r["family"] for r in result})!=8):
        raise ValueError("Unbalanced / unpinned measurement slice")
    return result


def stops_from_tokenizer(tokenizer)->tuple[int,...]:
    eos=tokenizer.eos_token_id
    unknown=getattr(tokenizer,"unk_token_id",None)
    if type(eos) is not int or eos<0:
        raise ValueError("Invalid tokenizer EOS")
    found={eos}
    for symbol in _STOP_TOKENS:
        tok=tokenizer.convert_tokens_to_ids(symbol)
        if type(tok) is int and tok>=0 and tok!=unknown:
            found.add(tok)
    if len(found)<2:
        raise ValueError("Both chat and pretraining EOS tokens must be recognized")
    return tuple(sorted(found))


def compute_end_reason(tokens, stops:tuple[int,...],budget:int)->str:
    if not tokens:
        return "other"
    if type(tokens[-1]) is not int:
        raise ValueError("Unexpected output token type")
    if tokens[-1] in stops:
        return "eos"
    if len(tokens)==budget:
        return "length"
    return "other"


def make_record(case_id:str,text:str,new_tokens:int,complete:bool,stop_reason:str)->dict:
    return {"id":case_id,"text":text,"new_tokens":new_tokens,
            "complete":complete,"stop_reason":stop_reason}


def validate_records(cases:list[dict],records:list[dict])->bool:
    if not isinstance(records,list) or len(records)!=len(cases) or len(cases) not in (24,72):
        raise ValueError("Wrong response count")
    seen=set()
    for case,record in zip(cases,records):
        if (type(record) is not dict or set(record)!=
              {"id","text","new_tokens","complete","stop_reason"}
            or record["id"]!=case["id"] or record["id"] in seen
            or type(record["text"]) is not str or len(record["text"])>MAX_CHARS
            or any(ord(c)<32 and c not in "\n\t\r" for c in record["text"])
            or type(record["complete"]) is not bool
            or type(record["new_tokens"]) is not int
            or not 1<=record["new_tokens"]<=MAX_NEW_TOKENS
            or record["stop_reason"] not in ("eos","length","other")
            or record["complete"]!=(record["stop_reason"]=="eos")
            or (record["stop_reason"]=="length" and
                record["new_tokens"]!=MAX_NEW_TOKENS)):
            raise ValueError("Unsafe, fabricated or reordered model answer")
        seen.add(record["id"])
    return True


def _summarize(cases:list[dict],rows:list[dict])->dict:
    counts={"total":len(cases),"correct":0,"wrong_final":0,
            "unreadable_complete":0,"length_censored":0,
            "non_eos_terminated":0,"possibly_correct_but_unfinished":0,
            "parseable_but_unfinished":0}
    for case,row in zip(cases,rows):
        parsed=old.base.parse_answer(case,row["text"])
        equal=parsed==Fraction(int(case["answer"]))
        if row["complete"]:
            if parsed is None:
                counts["unreadable_complete"]+=1
            elif equal:
                counts["correct"]+=1
            else:
                counts["wrong_final"]+=1
        else:
            counts["length_censored" if row["stop_reason"]=="length"
                   else "non_eos_terminated"]+=1
            if parsed is not None:
                counts["parseable_but_unfinished"]+=1
                counts["possibly_correct_but_unfinished"]+=int(equal)
    return counts


def _pair(a,b):
    better=sum(not x and y for x,y in zip(a,b))
    worse=sum(x and not y for x,y in zip(a,b))
    return {"improvements":better,"regressions":worse,"net":better-worse,
            "exact_mcnemar_p_exploratory":exact_mcnemar_p(better,worse)}


def score_responses(cases:list[dict],responses:dict,mode:str)->dict:
    chosen=select_cases(evaluation_cases(),mode)
    if [c["id"] for c in cases]!=[c["id"] for c in chosen]:
        raise ValueError("Nonfrozen pilot or full source")
    if type(responses) is not dict or set(responses)!=set(ARMS):
        raise ValueError("All three model arms required")
    for arm in ARMS:
        validate_records(cases,responses[arm])
    truth={arm:[bool(row["complete"] and
                  old.base.parse_answer(case,row["text"])==Fraction(int(case["answer"])))
                for case,row in zip(cases,responses[arm])] for arm in ARMS}
    report={
        "schema":1,"status":STATUS,"sample":mode,"total":len(cases),
        "max_new_tokens":MAX_NEW_TOKENS,
        "identical_messages_all_arms":True,
        "greedy_deterministic":True,
        "different_prompt_from_160_token_test":True,
        "stop_tokens_checked":["im_end","endoftext"],
        "models":{arm:_summarize(cases,responses[arm]) for arm in ARMS},
        "paired":{
            "v5_minus_qwen":_pair(truth["qwen"],truth["v5"]),
            "candidate_minus_v5":_pair(truth["v5"],truth["candidate"]),
            "candidate_minus_qwen":_pair(truth["qwen"],truth["candidate"]),
        },
        "slices":{},
        "model_only_no_tools":True,"raw_responses_uploaded":False,
        "training_performed":False,"release_validated":False,
        "weights_promoted":False,"generalization_proven":False,
        "limitations":"Developer-authored cases already used for evaluation. Pilot is 3 per family, correlated and underpowered; this is not an independent holdout or release approval.",
    }
    for part in ("retention","transfer"):
        ix=[i for i,c in enumerate(cases) if c["part"]==part]
        report["slices"][part]={
            "models":{arm:_summarize([cases[i] for i in ix],
                [responses[arm][i] for i in ix]) for arm in ARMS},
            "paired":{
                "candidate_minus_qwen":_pair(
                    [truth["qwen"][i] for i in ix],
                    [truth["candidate"][i] for i in ix]),
                "candidate_minus_v5":_pair(
                    [truth["v5"][i] for i in ix],
                    [truth["candidate"][i] for i in ix]),
            },
        }
    return report


def _budget_manifest(out:Path,mode:str)->dict:
    meta=old._load_json(out/"fair-manifest.json")
    chosen=select_cases(old._cases_folder(out),mode)
    digest=sha256("\n".join(c["id"] for c in chosen).encode()).hexdigest()
    if (meta.get("schema")!=1 or meta.get("status")!=STATUS
            or meta.get("mode")!=mode or meta.get("total")!=len(chosen)
            or meta.get("ids_sha256")!=digest
            or meta.get("max_new_tokens")!=MAX_NEW_TOKENS
            or meta.get("release_validated") is not False):
        raise ValueError("Unsealed task subset or budget drift")
    return meta


def prepare(out:Path,mode:str)->dict:
    if mode not in _ALLOWED_MODES:
        raise ValueError("Invalid measurement mode")
    old.prepare_run(out)
    chosen=select_cases(old._cases_folder(out),mode)
    meta={"schema":1,"status":STATUS,"mode":mode,"total":len(chosen),
          "max_new_tokens":MAX_NEW_TOKENS,
          "ids_sha256":sha256("\n".join(c["id"] for c in chosen).encode()).hexdigest(),
          "release_validated":False,"weights_promoted":False}
    (out/"fair-manifest.json").write_text(
        json.dumps(meta,sort_keys=True)+"\n",encoding="utf8")
    return meta


def _write_records(path:Path,records:list[dict]):
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ValueError("Never overwrite original/private responses")
    with path.open("x",encoding="utf8") as f:
        for row in records:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")


def _read_records(path:Path)->list[dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>1_500_000:
        raise ValueError("Missing or oversized private response file")
    return [json.loads(line) for line in path.read_text("utf8").splitlines()]


def run_arm(folder:Path,mode:str,arm:str,
            base_path:Path,v5_path:Path,candidate_path:Path,
            receipt:Path,authorize:str)->dict:
    if arm not in ARMS or authorize!=AUTH:
        raise ValueError("Explicit read-only GPU authorization missing")
    _budget_manifest(folder,mode)
    cases=select_cases(old._cases_folder(folder),mode)
    evidence=old.candidate_receipt(receipt,candidate_path)
    from neral_hf_identity import (validate_hf_snapshot,validate_adapter,CHAT_TEMPLATE)
    from scripts.neral_math_eval import _load_model
    from scripts.neral_v3_eval_isolation import prepare_cuda,clean_cuda
    old.base._gpu_ready()
    model_snapshot=validate_hf_snapshot(base_path,verify_hashes=True)
    if model_snapshot["revision"]!=old.base.BASE_SHA:
        raise ValueError("Wrong foundation revision")
    original=validate_adapter(v5_path,model_snapshot)
    if original["files"]["adapter_model.safetensors"]!=old.base.V5_SHA:
        raise ValueError("Wrong original v5 model")
    config=old._load_json(candidate_path/"adapter_config.json")
    if (config.get("peft_type")!="LORA" or config.get("r")!=16
        or config.get("task_type")!="CAUSAL_LM"
        or set(config.get("target_modules",[]))!=set(
            ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"))):
        raise ValueError("Candidate LoRA rank or structure unrecognized")
    os.environ.update(HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1",WANDB_DISABLED="true")
    import torch
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    tokenizer=Qwen2TokenizerFast.from_pretrained(
        str(base_path),local_files_only=True,
        trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    stop=stops_from_tokenizer(tokenizer)
    prepare_cuda(torch)
    model=None
    rows=[]
    try:
        model=_load_model(base_path)
        adapter={"qwen":None,"v5":v5_path,"candidate":candidate_path}[arm]
        if adapter is not None:
            model=PeftModel.from_pretrained(
                model,str(adapter),is_trainable=False,local_files_only=True)
        model.eval()
        for case in cases:
            messages=[{"role":"system","content":BRIEF_SYSTEM},
                      {"role":"user","content":case["prompt"]}]
            tokens=tokenizer.apply_chat_template(
                messages,chat_template=CHAT_TEMPLATE,tokenize=True,
                add_generation_prompt=True,return_tensors="pt").to("cuda:0")
            if tokens.shape[-1]>2048:
                raise ValueError("Prompt exceeds safe context")
            with torch.inference_mode():
                output=model.generate(
                    tokens,attention_mask=torch.ones_like(tokens),
                    do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=list(stop),
                    pad_token_id=tokenizer.pad_token_id)
            produced=output[0,tokens.shape[-1]:].tolist()
            reason=compute_end_reason(produced,stop,MAX_NEW_TOKENS)
            raw=tokenizer.decode(produced,skip_special_tokens=True)
            if len(raw)>MAX_CHARS or not produced:
                raise ValueError("Overlong or empty model response")
            rows.append(make_record(case["id"],raw,len(produced),
                                    reason=="eos",reason))
            del tokens,output,produced
        validate_records(cases,rows)
        _write_records(folder/(arm+".jsonl"),rows)
    finally:
        if model is not None: del model
        clean_cuda(torch)
    return {"status":"NERAL_FAIR_512_PRIVATE_ARM_DONE","arm":arm,
            "cases":len(rows),"candidate_sha256":evidence["candidate_sha256"],
            "complete":sum(r["complete"] for r in rows),
            "raw_responses_uploaded":False,
            "release_validated":False}


def combine(folder:Path,mode:str,receipt:Path,candidate:Path)->dict:
    _budget_manifest(folder,mode)
    cases=select_cases(old._cases_folder(folder),mode)
    evidence=old.candidate_receipt(receipt,candidate)
    responses={arm:_read_records(folder/(arm+".jsonl")) for arm in ARMS}
    report=score_responses(cases,responses,mode)
    report["candidate_adapter_sha256"]=evidence["candidate_sha256"]
    report["foundation_revision"]=old.base.BASE_SHA
    report["predecessor_adapter_sha256"]=old.base.V5_SHA
    path=folder/"fair-summary.json"
    if path.exists() or path.is_symlink():
        raise ValueError("Never overwrite diagnostic results")
    with path.open("x",encoding="utf8") as f:
        json.dump(report,f,sort_keys=True,indent=2)
        f.write("\n")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--mode",choices=("prepare","run","combine"),required=True)
    parser.add_argument("--sample",choices=_ALLOWED_MODES,required=True)
    parser.add_argument("--folder",type=Path,required=True)
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--base",type=Path)
    parser.add_argument("--v5",type=Path)
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--receipt",type=Path)
    parser.add_argument("--authorize",default="")
    args=parser.parse_args(argv)
    if args.mode=="prepare":
        summary=prepare(args.folder,args.sample)
    elif args.mode=="run":
        if not all([args.arm,args.base,args.v5,args.candidate,args.receipt]):
            parser.error("Pinned paths and model arm required")
        summary=run_arm(args.folder,args.sample,args.arm,args.base,args.v5,
                        args.candidate,args.receipt,args.authorize)
    else:
        if not args.receipt or not args.candidate:
            parser.error("Original candidate receipt and weights required")
        summary=combine(args.folder,args.sample,args.receipt,args.candidate)
    print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    main()
