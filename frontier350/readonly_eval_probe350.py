"""Arctus350 first native, bounded, GREEDY model-quality smoke benchmark.

Uses 8 original synthetic prompts, no tools/retrieval, no training and no
weight writing. Never prints model generations, input prompts, private file
paths or checkpoint tensors. This is a *baseline probe*, not a blinded
independent benchmark or proof of improved native intelligence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

CASES = [
    {"id":"arith-a","category":"arithmetic","prefix":"Rechne 37 + 48. Antworte nur mit der Zahl.\nAntwort:","expected_exact":"85","max_new_tokens":24},
    {"id":"arith-b","category":"arithmetic","prefix":"Was ist 12 mal 9? Nur die Zahl.\nAntwort:","expected_exact":"108","max_new_tokens":24},
    {"id":"arith-c","category":"arithmetic","prefix":"Berechne 144 - 29. Nur die Zahl.\nAntwort:","expected_exact":"115","max_new_tokens":24},
    {"id":"instr-a","category":"instruction","prefix":"Schreibe exakt das Wort Komet.\nAntwort:","expected_exact":"Komet","max_new_tokens":24},
    {"id":"instr-b","category":"instruction","prefix":"Antworte ausschließlich mit dem Wort Regen.\nAntwort:","expected_exact":"Regen","max_new_tokens":24},
    {"id":"engl-a","category":"english","prefix":"Translate the German word Haus to English. One word only.\nAnswer:","expected_exact":"house","max_new_tokens":24},
    {"id":"engl-b","category":"english","prefix":"Fill the blank with one English word: The Sun is a ____.\nAnswer:","expected_exact":"star","max_new_tokens":24},
    {"id":"germ-a","category":"german","prefix":"Welche Farbe hat eine reife Zitrone? Antworte mit nur einem Wort.\nAntwort:","expected_exact":"gelb","max_new_tokens":24},
]


def validate_cases(cases):
    if not isinstance(cases,list) or not 1<=len(cases)<=12:
        raise ValueError("Invalid synthetic smoke case count")
    seen=set()
    for row in cases:
        if (not isinstance(row,dict) or
            not isinstance(row.get("id"),str) or not row["id"] or row["id"] in seen or
            not isinstance(row.get("prefix"),str) or not 5<=len(row["prefix"])<=240 or
            not isinstance(row.get("category"),str) or not row["category"] or
            not isinstance(row.get("expected_exact"),str) or not row["expected_exact"] or
            type(row.get("max_new_tokens")) is not int or not 1<=row["max_new_tokens"]<=24):
            raise ValueError("Invalid bounded synthetic generation case")
        seen.add(row["id"])


def aggregate_results(cases, rows, eos_id):
    validate_cases(cases)
    if type(eos_id) is not int:
        raise ValueError("EOS token must be integer")
    if len(cases)!=len(rows):
        raise ValueError("Missing model generation result")
    expected={x["id"]:x for x in cases}
    seen=set();counts={};totals={"exact":0,"tokens":0,"seconds":0.0,"eos":0,"repeat":0.0}
    for row in rows:
        if not isinstance(row,dict) or row.get("id") not in expected or row["id"] in seen:
            raise ValueError("Invalid or duplicate result row")
        seen.add(row["id"])
        if (not isinstance(row.get("text"),str) or
            not isinstance(row.get("generated_ids"),list) or
            not all(type(n) is int and n>=0 for n in row["generated_ids"]) or
            type(row.get("elapsed_seconds")) not in (int,float) or
            not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"]<=0):
            raise ValueError("Invalid or non-finite generation result")
        if row.get("category") != expected[row["id"]]["category"]:
            raise ValueError("Mismatched result category")
        case=expected[row["id"]]
        if len(row["generated_ids"]) > case["max_new_tokens"]:
            raise ValueError("Unbounded generation")
        exact = int(row["text"].strip().casefold() == case["expected_exact"].strip().casefold())
        cat=counts.setdefault(case["category"],{"cases":0,"exact":0})
        cat["cases"]+=1;cat["exact"]+=exact
        totals["exact"]+=exact
        totals["tokens"]+=len(row["generated_ids"])
        totals["seconds"]+=row["elapsed_seconds"]
        totals["eos"]+=int(eos_id in row["generated_ids"])
    if seen != set(expected):
        raise ValueError("Cases and results differ")
    return {
        "schema":1, "kind":"native_arctus350_greedy_smoke_not_final_holdout",
        "cases":len(cases),
        "exact_match_pct":round(100*totals["exact"]/len(cases),2),
        "categories":{k:{"cases":v["cases"],"exact_match_pct":round(100*v["exact"]/v["cases"],2)}
                      for k,v in sorted(counts.items())},
        "eos_fraction":round(totals["eos"]/len(cases),4),
        "generated_tokens":totals["tokens"],
        "seconds":round(totals["seconds"],3),
        "tokens_per_second":round(totals["tokens"]/totals["seconds"],3),
        "weights_modified":False,
        "private_data_uploaded":False,
        "tools_enabled":False,
        "model_outputs_logged":False,
        "ready_for_training":False,
    }


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--expected-checkpoint-sha256",required=True)
    args=ap.parse_args(argv)
    import torch
    from tokenizers import Tokenizer
    from chat350_contract import EXPECTED_TOKENIZER_SHA256
    from secure_checkpoint350 import load_checkpoint
    from model350 import FRONTIER_350,Neral350M
    from eval_chat350 import generate, sha256_file

    validate_cases(CASES)
    if not isinstance(args.expected_checkpoint_sha256,str) or len(args.expected_checkpoint_sha256)!=64:
        raise ValueError("Expected SHA256 is required")
    if args.checkpoint.is_symlink() or args.tokenizer.is_symlink():
        raise ValueError("Original files must not be symlinks")
    if sha256_file(args.checkpoint)!=args.expected_checkpoint_sha256:
        raise ValueError("Original checkpoint SHA mismatch")
    if sha256_file(args.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Tokenizer SHA mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 hardware not ready")

    tok=Tokenizer.from_file(str(args.tokenizer))
    ck=load_checkpoint(args.checkpoint)
    if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Model/tokenizer identity mismatch")
    if not isinstance(ck.get("model"),dict):
        raise ValueError("Model tensors missing")
    device=torch.device("cuda")
    model=Neral350M().to(device=device,dtype=torch.bfloat16)
    model.load_state_dict(ck["model"],strict=True)
    model.eval()
    eos=int(tok.token_to_id("<|eos|>"))
    rows=[]
    for case in CASES:
        start=time.monotonic()
        ids,response=generate(model,tok,case["prefix"],case["max_new_tokens"],device)
        torch.cuda.synchronize()
        rows.append({"id":case["id"],"category":case["category"],
                    "generated_ids":ids,"text":response,
                    "elapsed_seconds":max(time.monotonic()-start,0.000001)})
        # No source, private output text, or raw generation is printed or saved.
    summary=aggregate_results(CASES,rows,eos)
    summary["case_sha256"]=hashlib.sha256(
        json.dumps(CASES,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()
    summary["checkpoint_sha256_verified"]=True
    summary["tokenizer_sha256_verified"]=True
    summary["device_kind"]="cuda_bf16"
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
