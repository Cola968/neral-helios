"""Direct public-development A/B between two Phoenix FP32 foundation-v2 checkpoints.

Intended for controlled same-token source-mix experiments. Both checkpoints must be complete
4Mi development refresh states. Measures the existing NIS2/free-answer/knowledge-choice vector
plus correct-digit ranking under the published 24-case prompt formats. Detailed per-case
outputs remain local. This is not an independent holdout or production promotion.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))

from helios350_preview import TOKENIZER_SHA,trusted_asset
from helios_domain_dev_probe import PUBLIC_DEV_CASES,prompt_styles,validate_public_cases
from phoenix_4m_digit_diagnostic import ranked_digits_device
from phoenix_auto_sft_diagnostic import sha256_file
from phoenix_foundation_refresh_public_dev_eval import load_candidate,run_model,knowledge_choice_eval

REFERENCE_SHA="d54d3f752b22be09257164459c3bf56b0dd5c430db023882c4546430988a4963"
EXPECTED_NEW_TOKENS=4_194_304


def digit_eval(model,tok,bos,torch):
    validate_public_cases();rows=[]
    for case_id,domain,language,question,expected,equation in PUBLIC_DEV_CASES:
        rankings={name:ranked_digits_device(model,tok,bos,torch,prefix,expected)
                  for name,prefix in prompt_styles(question,language,equation).items()}
        rows.append({"id":case_id,"domain":domain,"language":language,"rankings":rankings})
    formats=sorted({name for r in rows for name in r["rankings"]})
    def agg(selected):
        out={}
        for name in formats:
            vals=[r["rankings"][name] for r in selected if name in r["rankings"]]
            if vals:out[name]={"cases":len(vals),"top1":sum(v["correct_top1"] for v in vals),
                               "top1_pct":100.0*sum(v["correct_top1"] for v in vals)/len(vals),
                               "mean_rank":round(sum(v["correct_rank"] for v in vals)/len(vals),3)}
        return out
    return {"overall":agg(rows),
            "domains":{d:agg([r for r in rows if r["domain"]==d]) for d in sorted({r["domain"] for r in rows})},
            "languages":{lang:agg([r for r in rows if r["language"]==lang]) for lang in ("de","en")},
            "rows":rows}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--trusted-own-checkpoints",action="store_true")
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--reference-sha256",required=True)
    p.add_argument("--candidate",type=Path,required=True);p.add_argument("--candidate-sha256",required=True)
    p.add_argument("--assets-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoints:p.error("explicit own-checkpoint opt-in required")
    if a.reference_sha256!=REFERENCE_SHA or sha256_file(a.reference)!=REFERENCE_SHA:raise ValueError("reference must be exact existing 4Mi leader")
    if len(a.candidate_sha256)!=64 or sha256_file(a.candidate)!=a.candidate_sha256:raise ValueError("candidate SHA mismatch")
    if a.out.exists() or not a.out.is_absolute():p.error("new absolute output required")

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    guard.require_cuda_ready(torch)

    root=a.assets_root.resolve(strict=True);tok_path=trusted_asset(root,root/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)
    tok=Tokenizer.from_file(str(tok_path));bos,eos=tok.token_to_id("<|bos|>"),tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos==eos:raise ValueError("invalid tokenizer")

    def evaluate(path,sha):
        state=load_candidate(torch,path,sha,FRONTIER_350.to_dict())
        if state.get("processed_tokens")!=EXPECTED_NEW_TOKENS:raise ValueError("same 4Mi boundary required")
        meta={"sha256":sha,"manifest_sha256":state.get("refresh_manifest_sha256"),
              "lineage":state.get("lineage"),"best_validation":state.get("best_validation")}
        m=Neral350M().to(device="cuda",dtype=torch.bfloat16);m.load_state_dict(state["model"],strict=True);m.eval();del state
        model_result=run_model(torch,m,tok,int(bos),int(eos))
        knowledge=knowledge_choice_eval(torch,m,tok,int(bos))
        digits=digit_eval(m,tok,int(bos),torch)
        del m;torch.cuda.empty_cache();return meta,model_result,knowledge,digits

    ref_meta,ref,ref_k,ref_d=evaluate(a.reference,a.reference_sha256)
    cand_meta,cand,cand_k,cand_d=evaluate(a.candidate,a.candidate_sha256)
    guard.require_safe_runtime_temperature()
    rs,cs=ref["summary"],cand["summary"]
    delta={k:cs[k]-rs[k] for k in rs if isinstance(rs[k],(int,float)) and isinstance(cs.get(k),(int,float))}
    knowledge_delta={k:cand_k[k]["accuracy_pct"]-ref_k[k]["accuracy_pct"] for k in ref_k}
    digit_delta={}
    for fmt,row in ref_d["overall"].items():
        if fmt in cand_d["overall"]:
            digit_delta[fmt]={"top1":cand_d["overall"][fmt]["top1"]-row["top1"],
                              "top1_pct":cand_d["overall"][fmt]["top1_pct"]-row["top1_pct"],
                              "mean_rank":cand_d["overall"][fmt]["mean_rank"]-row["mean_rank"]}
    report={"classification":"PUBLIC_DEVELOPMENT_4MI_FOUNDATION_MIX_AB_NOT_HOLDOUT",
            "reference_meta":ref_meta,"candidate_meta":cand_meta,
            "reference":ref,"candidate":cand,"delta_candidate_minus_reference":delta,
            "reference_knowledge":ref_k,"candidate_knowledge":cand_k,"knowledge_delta_pct":knowledge_delta,
            "reference_digits":ref_d,"candidate_digits":cand_d,"digit_delta":digit_delta,
            "same_new_token_boundary":True,"independent_holdout":False,"production_eligible":False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"event":"phoenix_foundation_mix_ab_complete",
                      "reference":rs,"candidate":cs,"delta":delta,
                      "knowledge_reference":{k:v["accuracy_pct"] for k,v in ref_k.items()},
                      "knowledge_candidate":{k:v["accuracy_pct"] for k,v in cand_k.items()},
                      "knowledge_delta":knowledge_delta,
                      "digits_reference":ref_d["overall"],"digits_candidate":cand_d["overall"],
                      "digit_delta":digit_delta,"candidate_sha256":a.candidate_sha256,
                      "production_eligible":False},sort_keys=True),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
