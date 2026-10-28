"""Read-only RTX4070 free-generation audit on 76 withheld synthetic VAL rows.

No prompt/answer/ID/weights leaves the local PC. Only aggregate exact-match,
EOS and repetition by source task + language are stored. Compares immutable
native Arctus350 64-step and 96-step experimental models, never trains.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

BASE_SHA="57bbd1f70fcca6aee5e557b132d6b1b3f7b5e3378d0248923e5478bae36caf58"
CAND_SHA="9c13b315465b74312c347effee8ef9c9e520ce95749ce9844a0a1fd2b66f48b2"
MANIFEST_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
N=76


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def validate_eval_partition(splits):
    if not isinstance(splits,dict) or set(splits)!={"train","val","test"}:
        raise ValueError("expected exactly train/val/test")
    if not all(isinstance(splits[k],list) and splits[k] for k in splits):
        raise ValueError("empty or malformed split")
    ids={s:set() for s in splits}
    for s,rows in splits.items():
        for row in rows:
            if not isinstance(row,dict) or not isinstance(row.get("id"),str):
                raise ValueError("invalid source row")
            if row["id"] in ids[s]:raise ValueError("duplicate split id")
            ids[s].add(row["id"])
    if ids["val"]&(ids["train"]|ids["test"]):raise ValueError("leaking heldout rows")
    return splits["val"]


def record_aggregate(rows):
    """Privacy-conserving statistics, never input/output text or record IDs."""
    from .chat350_generation_metrics import repeated_ngram_fraction
    if not isinstance(rows,list) or not rows:raise ValueError("empty evaluation")
    aggregate=Counter()
    groups={}
    total_rep=0.0
    for row in rows:
        if not isinstance(row,dict) or row.get("domain") not in ("math","format"):
            raise ValueError("invalid domain")
        if row.get("language") not in ("de","en"):
            raise ValueError("invalid language")
        if type(row.get("eos")) is not bool or not isinstance(row.get("expected"),str) or not isinstance(row.get("generated"),str):
            raise ValueError("invalid generated/expected label")
        tokens=row.get("tokens")
        if not isinstance(tokens,list) or not all(type(x) is int and x>=0 for x in tokens):
            raise ValueError("invalid generated token IDs")
        correct=int(row["generated"].strip()==row["expected"])
        eos=int(row["eos"])
        complete=int(correct and eos)
        repeated=repeated_ngram_fraction(tokens,3)
        total_rep+=repeated
        group=row["domain"]+"/"+row["language"]
        g=groups.setdefault(group,Counter())
        for x in (aggregate,g):
            x["rows"]+=1
            x["exact_correct"]+=correct
            x["eos_count"]+=eos
            x["completed_exact"]+=complete
    out={
        "total":aggregate["rows"],
        "exact_correct":aggregate["exact_correct"],
        "eos_count":aggregate["eos_count"],
        "completed_exact":aggregate["completed_exact"],
        "repeated_trigram_fraction":total_rep/aggregate["rows"],
        "by_family_language":{
            k:dict(sorted(v.items())) for k,v in sorted(groups.items())
        },
    }
    return out


def improvement_gate(b,c):
    for x in (b,c):
        if (not isinstance(x,dict) or
            any(type(x.get(k)) is not int for k in ("exact_correct","eos_count","total")) or
            type(x.get("repeated_trigram_fraction")) not in (int,float) or
            not math.isfinite(x["repeated_trigram_fraction"]) or
            not 0<=x["repeated_trigram_fraction"]<=1 or
            not 0<=x["exact_correct"]<=x["total"] or not 0<=x["eos_count"]<=x["total"]):
            raise ValueError("invalid cohort measurement")
    if b["total"]!=c["total"]:raise ValueError("different validation sample counts")
    return (c["exact_correct"]>b["exact_correct"] and
            c["eos_count"]>=b["eos_count"] and
            c["repeated_trigram_fraction"]<=b["repeated_trigram_fraction"]+1e-12)


def _gpu_temp():
    try:
        proc=subprocess.run(
            ["nvidia-smi","--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,text=True,check=False,timeout=6)
        lines=proc.stdout.strip().splitlines()
        return int(lines[0]) if proc.returncode==0 and len(lines)==1 else None
    except (OSError,ValueError,subprocess.TimeoutExpired):
        return None


def measure_one(model_path,tok,rows):
    import torch
    from .model350 import Neral350M,FRONTIER_350
    from .secure_checkpoint350 import load_checkpoint
    from eval_chat350 import generate
    expected_sha=BASE_SHA if "64step" in model_path.name else CAND_SHA
    if sha256(model_path)!=expected_sha:
        raise RuntimeError("model SHA drift or unsupported model")
    check=load_checkpoint(model_path)
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    if (check.get("config")!=FRONTIER_350.to_dict()
        or check.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256
        or check.get("experimental_non_promotable") is not True):
        raise ValueError("invalid native Arctus model lineage")
    model=Neral350M().to(device="cuda",dtype=torch.bfloat16)
    model.load_state_dict(check["model"],strict=True)
    del check
    model.eval()
    out=[]
    for idx,row in enumerate(rows):
        if idx%8==0:
            temp=_gpu_temp()
            if temp is None or temp>=80:
                raise RuntimeError("missing/hot GPU telemetry")
            if torch.cuda.mem_get_info(0)[0]//2**20<2048:
                raise RuntimeError("CUDA free memory below 2GiB")
        prompt=row["messages"][0]["content"]
        answer=row["messages"][1]["content"]
        seq,text=generate(model,tok,"User: "+prompt+"\nAssistant:",16,"cuda")
        eos=int(tok.token_to_id("<|eos|>"))
        tokens=seq[:seq.index(eos)] if eos in seq else seq
        out.append({
            "id":row["id"],"domain":row["domain"],"language":row["language"],
            "expected":answer,"generated":text,
            "eos":eos in seq,"tokens":tokens,
        })
    report=record_aggregate(out)
    del model
    torch.cuda.empty_cache()
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline",required=True,type=Path)
    ap.add_argument("--candidate",required=True,type=Path)
    ap.add_argument("--tokenizer",required=True,type=Path)
    ap.add_argument("--data",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError("refuse overwrite")
    from zoneinfo import ZoneInfo
    now=datetime.now(ZoneInfo("Europe/Berlin"))
    if now.date().isoformat()!="2026-10-28" or now.hour>=21:
        raise RuntimeError("read-only GPU comparison is permitted today before 21:00 only")
    import torch
    from tokenizers import Tokenizer
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .arctus350_exact_math_curriculum import independently_verify_row
    if not torch.cuda.is_available() or "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected own RTX4070 GPU")
    if torch.cuda.mem_get_info(0)[0]//2**20<9216:
        raise RuntimeError("GPU is occupied")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    if sha256(a.baseline)!=BASE_SHA or sha256(a.candidate)!=CAND_SHA:
        raise ValueError("pinned local source/candidate checkpoint digest mismatch")
    if sha256(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("locked BPE digest mismatch")
    manifest,source=validate_dataset_root(a.data)
    if manifest["manifest_sha256"]!=MANIFEST_SHA:
        raise ValueError("supplemental dataset SHA mismatch")
    validate_prompt_holdouts(source)
    rows=validate_eval_partition(source)
    if len(rows)!=N or any(not independently_verify_row(r) for r in rows):
        raise ValueError("unexpected val count or wrong reference label")
    tokenizer=Tokenizer.from_file(str(a.tokenizer))
    base=measure_one(a.baseline,tokenizer,rows)
    candidate=measure_one(a.candidate,tokenizer,rows)
    verdict=improvement_gate(base,candidate)
    report={
        "schema":"arctus350-heldout-76-64-vs-96-free-generation-v1",
        "status":"completed_read_only",
        "subset":"synthetic validation only; no train and no test",
        "manifest_sha256":MANIFEST_SHA,
        "baseline_checkpoint_sha256":BASE_SHA,
        "candidate_checkpoint_sha256":CAND_SHA,
        "baseline":base,"candidate":candidate,
        "strict_no_regression_gate":verdict,
        "automatic_model_promotion":False,
        "public_release_allowed":False,
        "privacy":"aggregate counts only; no prompts, generated answers, row IDs or model weights uploaded",
        "cloud_cost_eur":0,
        "training_started":False,
        "limitation":"synthetic task val, not independent natural-language benchmark; instruction correctness is exact comparison only",
    }
    a.out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
