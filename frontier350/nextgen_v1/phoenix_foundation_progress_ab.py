"""Direct public-development A/B for sequential Phoenix foundation checkpoints.

Development-only progression gate. Unlike the same-token source-mix A/B, this compares a
4Mi checkpoint with its exact 8Mi resume descendant from the same manifest/lineage. It does
not create an independent holdout or production-promotion claim.
"""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))

from helios350_preview import TOKENIZER_SHA,trusted_asset
from phoenix_auto_sft_diagnostic import sha256_file
from phoenix_foundation_refresh_public_dev_eval import load_candidate,run_model,knowledge_choice_eval
from phoenix_foundation_mix_ab import digit_eval

SHA64=re.compile(r"^[0-9a-f]{64}$")
REFERENCE_TOKENS=4_194_304
CANDIDATE_TOKENS=8_388_608

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoints",action="store_true")
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--reference-sha256",required=True)
    p.add_argument("--candidate",type=Path,required=True);p.add_argument("--candidate-sha256",required=True)
    p.add_argument("--assets-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoints:p.error("explicit own-checkpoint opt-in required")
    for path,digest in ((a.reference,a.reference_sha256),(a.candidate,a.candidate_sha256)):
        if not SHA64.fullmatch(digest) or sha256_file(path)!=digest:raise ValueError("checkpoint SHA mismatch")
    if a.out.exists() or not a.out.is_absolute():p.error("new absolute output required")

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    guard.require_cuda_ready(torch)

    root=a.assets_root.resolve(strict=True)
    tok_path=trusted_asset(root,root/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)
    tok=Tokenizer.from_file(str(tok_path));bos,eos=tok.token_to_id("<|bos|>"),tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos==eos:raise ValueError("invalid tokenizer")

    def evaluate(path,sha,expected_tokens):
        state=load_candidate(torch,path,sha,FRONTIER_350.to_dict())
        if state.get("processed_tokens")!=expected_tokens:raise ValueError("unexpected progression boundary")
        meta={"sha256":sha,"processed_tokens":expected_tokens,
              "manifest_sha256":state.get("refresh_manifest_sha256"),"lineage":state.get("lineage"),
              "best_validation":state.get("best_validation")}
        m=Neral350M().to(device="cuda",dtype=torch.bfloat16);m.load_state_dict(state["model"],strict=True);m.eval();del state
        result=run_model(torch,m,tok,int(bos),int(eos));knowledge=knowledge_choice_eval(torch,m,tok,int(bos))
        digits=digit_eval(m,tok,int(bos),torch)
        del m;torch.cuda.empty_cache()
        return meta,result,knowledge,digits

    rm,ref,rk,rd=evaluate(a.reference,a.reference_sha256,REFERENCE_TOKENS)
    cm,cand,ck,cd=evaluate(a.candidate,a.candidate_sha256,CANDIDATE_TOKENS)
    if rm["manifest_sha256"]!=cm["manifest_sha256"]:raise ValueError("progression checkpoints use different corpus manifests")
    guard.require_safe_runtime_temperature()
    rs,cs=ref["summary"],cand["summary"]
    delta={k:cs[k]-rs[k] for k in rs if isinstance(rs[k],(int,float)) and isinstance(cs.get(k),(int,float))}
    kd={k:ck[k]["accuracy_pct"]-rk[k]["accuracy_pct"] for k in rk}
    dd={}
    for fmt,row in rd["overall"].items():
        if fmt in cd["overall"]:
            dd[fmt]={"top1":cd["overall"][fmt]["top1"]-row["top1"],
                     "top1_pct":cd["overall"][fmt]["top1_pct"]-row["top1_pct"],
                     "mean_rank":cd["overall"][fmt]["mean_rank"]-row["mean_rank"]}
    report={"classification":"PUBLIC_DEVELOPMENT_FOUNDATION_PROGRESS_AB_NOT_HOLDOUT",
            "reference_meta":rm,"candidate_meta":cm,"reference":ref,"candidate":cand,
            "delta_candidate_minus_reference":delta,"reference_knowledge":rk,"candidate_knowledge":ck,
            "knowledge_delta_pct":kd,"reference_digits":rd,"candidate_digits":cd,"digit_delta":dd,
            "same_lineage_manifest":True,"independent_holdout":False,"production_eligible":False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"event":"phoenix_foundation_progress_ab_complete","reference":rs,"candidate":cs,
                      "delta":delta,"knowledge_delta":kd,"digit_delta":dd,
                      "reference_sha256":a.reference_sha256,"candidate_sha256":a.candidate_sha256,
                      "production_eligible":False},sort_keys=True),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
