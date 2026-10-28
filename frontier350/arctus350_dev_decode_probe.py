"""Read-only A/B on six PUBLIC developer cases: does source-trigram blocking help?

Never a frozen-suite or release score. Greedy baseline versus deterministic
prompt-source trigram blocking. Checkpoints and tokenizer remain immutable.
Not a replacement for native 350M evaluation. Model capability may stay poor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

CASE_IDS=("de_chat_01","en_chat_01","multi_01",
          "exact_ok_de","math_01","unknown_01")
MODELS={
    "2b": "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7",
    "historical_1b_sft": "408998c0d0816527341253fe37c2d9e2f1216c5cad8500dee247d11c72cc8d7b",
    "2b_sixstep": "05eb63767613458bdf6a1c5849b40b73d953729f274864aa345fe69c6c781c52",
}


def forbidden_source_ngrams(ids):
    return {tuple(ids[i:i+3]) for i in range(len(ids)-2)}


def disallows_source_trigram(already, candidate, source):
    return len(already)>=2 and tuple([*already[-2:],candidate]) in source


def normalize_text(text):
    if not isinstance(text,str):
        raise ValueError("text must be Unicode")
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())


def echo_flags(prompt,answer):
    p=normalize_text(prompt)
    a=normalize_text(answer)
    exact=bool(p and a==p)
    prefix=bool(p and (a==p or (a.startswith(p) and len(a)>len(p)
                      and not (a[len(p)].isalnum() or a[len(p)] in "_-'"))))
    return {"exact":exact,"prefix":prefix}


def validate_case_selector(selected,available):
    if (not isinstance(selected,tuple) or not 1<=len(selected)<=6
        or any(type(x) is not str for x in selected)
        or len(selected)!=len(set(selected))
        or not set(selected).issubset(set(available))):
        raise ValueError("invalid/missing developer-case selector")
    return selected


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for z in iter(lambda:f.read(4*1024*1024),b""):
            h.update(z)
    return h.hexdigest()


def _generate(model,tok,prefix,last_user,mode,max_new):
    import torch
    from .chat350_eval_contract import case_prefix_ids,generation_forbidden_ids
    from .chat350_generation_metrics import repeated_ngram_fraction
    from .chat350_eval_contract import trim_hallucinated_user_turn
    from .model350 import FRONTIER_350

    eos=tok.token_to_id("<|eos|>")
    prompt_tokens=case_prefix_ids(tok,prefix)
    ids=list(prompt_tokens)
    source=forbidden_source_ngrams(prompt_tokens)
    for variant in (last_user," "+last_user):
        source.update(forbidden_source_ngrams(tok.encode(variant).ids))
    forbidden=generation_forbidden_ids(tok)
    generated=[]
    blocked=0
    for _ in range(max_new):
        if len(ids)>=FRONTIER_350.max_seq_len:break
        if torch.cuda.mem_get_info(0)[0]//2**20 < 4096:
            raise RuntimeError("insufficient shared CUDA free memory")
        x=torch.tensor([ids],device="cuda",dtype=torch.long)
        z=model(x)[0][0,-1].clone()
        z[list(forbidden)]=-float("inf")
        if mode=="source_trigram_block":
            for _retry in range(32):
                candidate=int(z.argmax().item())
                if not disallows_source_trigram(generated,candidate,source):break
                z[candidate]=-float("inf")
                blocked+=1
            else:
                raise ValueError("source trigram guard exhausted 32 options")
        nxt=int(z.argmax().item())
        generated.append(nxt)
        if nxt==eos:break
        ids.append(nxt)
    ids_before_eos=generated[:-1] if generated and generated[-1]==eos else generated
    decoded=tok.decode(ids_before_eos,skip_special_tokens=True)
    display,_=trim_hallucinated_user_turn(decoded)
    flags=echo_flags(last_user,display)
    return {
        "text":display[:220],
        "eos_generated":bool(generated and generated[-1]==eos),
        "generated_token_count":len(ids_before_eos),
        "repeated_trigram_fraction":repeated_ngram_fraction(ids_before_eos),
        "source_trigram_suppressions":blocked,
        "prompt_echo_exact":flags["exact"],
        "prompt_echo_prefix":flags["prefix"],
        "truncated_output":len(display)>220,
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--cases",type=Path,required=True)
    ap.add_argument("--two-b",type=Path,required=True)
    ap.add_argument("--historical-sft",type=Path,required=True)
    ap.add_argument("--sixstep",type=Path,required=True)
    ap.add_argument("--windows-free-mib",type=int,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("output cannot be overwritten")
    if args.cases.name!="chat350_dev_cases.json":raise ValueError("only dev cases, no frozen suite")
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .chat350_eval_contract import parse_chat_prefix
    from .model350 import FRONTIER_350,Neral350M
    from .secure_checkpoint350 import load_checkpoint
    import torch
    from tokenizers import Tokenizer

    if args.windows_free_mib<8192:raise RuntimeError("Windows RTX4070 free VRAM below guard")
    if digest(args.tokenizer)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("tokenizer digest mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 unavailable")
    if "RTX 4070" not in torch.cuda.get_device_name(0):raise RuntimeError("wrong GPU")
    if torch.cuda.mem_get_info(0)[0]//2**20<8192:raise RuntimeError("CUDA free VRAM below guard")
    torch.cuda.set_per_process_memory_fraction(.35,device=0)
    case_data=json.loads(args.cases.read_text(encoding="utf8"))
    if not isinstance(case_data,list):raise ValueError("invalid case list")
    lookup={x["id"]:x for x in case_data}
    if len(lookup)!=len(case_data):raise ValueError("duplicate dev case ID")
    validate_case_selector(CASE_IDS,tuple(lookup))
    selected=[lookup[x] for x in CASE_IDS]
    for c in selected:
        if not 1<=c["max_new_tokens"]<=64:raise ValueError("invalid token budget")
        parse_chat_prefix(c["prefix"])
    tok=Tokenizer.from_file(str(args.tokenizer))
    if tok.get_vocab_size()!=FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    paths={"2b":args.two_b,"historical_1b_sft":args.historical_sft,"2b_sixstep":args.sixstep}
    result={
        "schema":"arctus350-dev-source-trigram-ab-v1",
        "status":"measured",
        "purpose":"decoding hypothesis only; NOT an intelligence or sealed benchmark",
        "case_ids":list(CASE_IDS),
        "max_new_cap":32,"models":{},
        "dev_suite_sha256":digest(args.cases),
        "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
        "training_started":False,"weights_modified":False,
        "release_allowed":False,
    }
    for label,path in paths.items():
        if digest(path)!=MODELS[label]:raise ValueError("pinned checkpoint mismatch: "+label)
        ck=load_checkpoint(path)
        if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:
            raise ValueError("checkpoint model/tokenizer mismatch")
        model=Neral350M().to(device="cuda",dtype=torch.bfloat16)
        model.load_state_dict(ck["model"],strict=True)
        del ck
        model.eval()
        rows=[]
        with torch.inference_mode():
            for case in selected:
                hist=parse_chat_prefix(case["prefix"])
                user=hist[-1][1]
                for mode in ("greedy","source_trigram_block"):
                    out=_generate(model,tok,case["prefix"],user,mode,min(case["max_new_tokens"],32))
                    rows.append({"id":case["id"],"mode":mode,**out})
        result["models"][label]={"checkpoint_sha256":MODELS[label],
                                 "rows":rows,"comparability":"same_checkpoint_both_modes"}
        del model
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info(0)[0]//2**20<8192:
            raise RuntimeError("shared GPU pressure between model runs")
    for label,p in paths.items():
        if digest(p)!=MODELS[label]:raise ValueError("checkpoint changed during decoding")
    result["max_cuda_reserved_mib"]=torch.cuda.max_memory_reserved(0)//2**20
    result["caveat"]="Suppressing copied source trigrams may force nonsense, not knowledge."
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
