"""Evaluate the hard-gated tuned auto-SFT Neral 350M candidate.

Public development only. Reuses the same NIS2 and eight DE/EN free-answer suites
as the earlier baseline/candidate comparison. Detailed generations remain private.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helios350_preview import TOKENIZER_SHA, trusted_asset
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_weight_only_pilot import _checked_historical_weights
from phoenix_auto_sft_public_dev_eval import (
    BASE_SHA, run_model, sha256_file,
)

SCHEMA = "phoenix_auto_sft_tuned_v2_fp32_master"
CLASSIFICATION = "auto_sft_tuned_public_dev_selected_not_promotable"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoints", action="store_true")
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--candidate-checkpoint", type=Path, required=True)
    p.add_argument("--candidate-sha256", required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoints:
        p.error("explicit trust required")
    if a.out.exists() or not a.out.is_absolute():
        p.error("new absolute output required")

    base_path=require_pinned_checkpoint(a.base_checkpoint)
    root=a.assets_root.resolve(strict=True)
    tok_path=trusted_asset(root, root/"frontier350_assets"/"tokenizer.json", TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    guard.require_cuda_ready(torch)

    tok=Tokenizer.from_file(str(tok_path))
    bos,eos=tok.token_to_id("<|bos|>"),tok.token_to_id("<|eos|>")
    if tok.get_vocab_size()!=FRONTIER_350.vocab_size or bos is None or eos is None or bos==eos:
        raise ValueError("tokenizer contract mismatch")
    device=torch.device("cuda")

    base=_checked_historical_weights(base_path,torch,Neral350M,FRONTIER_350.to_dict())
    m=Neral350M().to(device=device,dtype=torch.bfloat16)
    m.load_state_dict(base["model"],strict=True)
    del base
    baseline=run_model(torch,m,tok,int(bos),int(eos))
    del m
    torch.cuda.empty_cache()

    if sha256_file(a.candidate_checkpoint)!=a.candidate_sha256:
        raise ValueError("candidate SHA mismatch")
    state=torch.load(a.candidate_checkpoint,map_location="cpu",weights_only=False)
    if (state.get("checkpoint_schema")!=SCHEMA or state.get("classification")!=CLASSIFICATION
        or state.get("base_checkpoint_sha256")!=BASE_SHA
        or state.get("config")!=FRONTIER_350.to_dict()
        or state.get("tokenizer_sha256")!=TOKENIZER_SHA
        or state.get("production_eligible") is not False
        or state.get("master_parameter_dtype")!="float32"
        or state.get("adamw_moment_dtype")!="float32"
        or state.get("compute_autocast_dtype")!="bfloat16"
        or state.get("human_review_complete") is not False
        or state.get("independent_holdout_checked") is not False
        or not isinstance(state.get("model"),dict)):
        raise ValueError("tuned candidate contract mismatch")
    selected=state.get("selected")
    m=Neral350M().to(device=device,dtype=torch.bfloat16)
    m.load_state_dict(state["model"],strict=True)
    del state
    candidate=run_model(torch,m,tok,int(bos),int(eos))
    del m
    torch.cuda.empty_cache()
    guard.require_safe_runtime_temperature()

    b,c=baseline["summary"],candidate["summary"]
    delta={k:c[k]-b[k] for k in b if isinstance(b[k],(int,float)) and isinstance(c.get(k),(int,float))}
    report={
      "classification":"PUBLIC_DEVELOPMENT_TUNED_SFT_AB_NOT_INDEPENDENT_HOLDOUT",
      "base_sha256":BASE_SHA,"candidate_sha256":a.candidate_sha256,
      "selected_training_config":selected,
      "baseline":baseline,"candidate":candidate,"delta_candidate_minus_base":delta,
      "independent_holdout":False,"human_review_complete":False,
      "quality_gain_verified":False,"production_eligible":False
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"phoenix_tuned_sft_public_dev_ab_complete",
      "base":b,"candidate":c,"delta":delta,"candidate_sha256":a.candidate_sha256,
      "production_eligible":False},sort_keys=True),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
