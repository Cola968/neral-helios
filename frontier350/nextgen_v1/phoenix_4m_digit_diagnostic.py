"""Public-development latent digit diagnostic for the exact Phoenix 4Mi foundation checkpoint.

No training. Measures where the correct digit ranks among 0..9 under four public prompt
formats for all 24 published domain cases. This distinguishes latent answer preference from
free-generation formatting failure. Per-case details stay local; Actions logs show aggregates.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))

from helios350_preview import TOKENIZER_SHA,trusted_asset
from helios_domain_dev_probe import PUBLIC_DEV_CASES,prompt_styles,validate_public_cases
from phoenix_auto_sft_diagnostic import sha256_file
from phoenix_foundation_refresh_public_dev_eval import load_candidate

FOUNDATION_SHA="d54d3f752b22be09257164459c3bf56b0dd5c430db023882c4546430988a4963"
DIGITS=tuple(str(i) for i in range(10))


def ranked_digits_device(model,tokenizer,bos,torch,prefix,expected):
    device=next(model.parameters()).device
    context=[bos,*tokenizer.encode(prefix,add_special_tokens=False).ids]
    candidates=[tokenizer.encode(" "+digit,add_special_tokens=False).ids for digit in DIGITS]
    lengths={len(ids) for ids in candidates}
    if not context or len(context)>384 or len(lengths)!=1 or 0 in lengths:raise ValueError("unsupported digit candidates")
    scores=[]
    with torch.inference_mode():
        if next(iter(lengths))==1:
            logits,_=model(torch.tensor([context],dtype=torch.long,device=device))
            logp=torch.log_softmax(logits[0,-1].float(),dim=-1)
            scores=[float(logp[ids[0]].item()) for ids in candidates]
        else:
            for ids in candidates:
                x=torch.tensor([context+ids[:-1]],dtype=torch.long,device=device)
                logits,_=model(x)
                logp=torch.log_softmax(logits[0,len(context)-1:].float(),dim=-1)
                target=torch.tensor(ids,dtype=torch.long,device=device)
                scores.append(float(logp.gather(-1,target[:,None]).sum().item()))
    expected_index=int(expected);ranking=sorted(range(10),key=lambda i:scores[i],reverse=True)
    return {"top_digit":DIGITS[ranking[0]],"correct_rank":ranking.index(expected_index)+1,
            "correct_top1":ranking[0]==expected_index,
            "correct_vs_best_wrong_logprob_margin":scores[expected_index]-max(scores[i] for i in range(10) if i!=expected_index)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoint",action="store_true")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoint:p.error("explicit own-checkpoint opt-in required")
    if sha256_file(a.checkpoint)!=FOUNDATION_SHA:raise ValueError("exact 4Mi checkpoint SHA mismatch")
    if a.out.exists() or not a.out.is_absolute():p.error("new absolute output file required")
    validate_public_cases()

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    guard.require_cuda_ready(torch)

    root=a.assets_root.resolve(strict=True)
    tok_path=trusted_asset(root,root/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)
    tok=Tokenizer.from_file(str(tok_path));bos=tok.token_to_id("<|bos|>")
    if bos is None or tok.get_vocab_size()!=FRONTIER_350.vocab_size:raise ValueError("tokenizer contract mismatch")
    state=load_candidate(torch,a.checkpoint,FOUNDATION_SHA,FRONTIER_350.to_dict())
    model=Neral350M().to(device="cuda",dtype=torch.bfloat16)
    model.load_state_dict(state["model"],strict=True);model.eval();del state

    details=[]
    for case_id,domain,language,question,expected,equation in PUBLIC_DEV_CASES:
        rankings={name:ranked_digits_device(model,tok,int(bos),torch,prefix,expected)
                  for name,prefix in prompt_styles(question,language,equation).items()}
        details.append({"id":case_id,"domain":domain,"language":language,"expected":expected,"rankings":rankings})
    formats=sorted({name for row in details for name in row["rankings"]})
    def aggregate(rows):
        result={}
        for name in formats:
            vals=[row["rankings"][name] for row in rows if name in row["rankings"]]
            if not vals:continue
            result[name]={"cases":len(vals),"correct_digit_top1":sum(v["correct_top1"] for v in vals),
                          "top1_pct":100.0*sum(v["correct_top1"] for v in vals)/len(vals),
                          "mean_correct_digit_rank":round(sum(v["correct_rank"] for v in vals)/len(vals),3)}
        return result
    domains={d:aggregate([row for row in details if row["domain"]==d])
             for d in sorted({row["domain"] for row in details})}
    languages={lang:aggregate([row for row in details if row["language"]==lang]) for lang in ("de","en")}
    report={"classification":"PUBLIC_DEVELOPMENT_LATENT_DIGIT_DIAGNOSTIC_NOT_HOLDOUT",
            "checkpoint_sha256":FOUNDATION_SHA,"cases":len(details),"overall":aggregate(details),
            "domains":domains,"languages":languages,"details":details,
            "training_started":False,"weights_modified":False,"independent_holdout":False,
            "interpretation":"Forced-choice digit ranking only; compare with free-generation 0/24 to diagnose latent knowledge vs answer-format failure."}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"event":"phoenix_4m_digit_diagnostic_complete","overall":report["overall"],
                      "domains":domains,"languages":languages,"training_started":False},sort_keys=True),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
