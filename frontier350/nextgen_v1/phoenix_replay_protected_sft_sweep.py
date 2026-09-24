"""Replay-protected auto-SFT sweep for the OWN Neral 350M checkpoint.

Combines response-only OASST2 SFT with a small verified foundation-replay loss.
The replay mixture follows the existing Neral V9 diagnostic pattern: 9/12 main
foundation, then education/general/math replay. This is a DEVELOPMENT experiment:
public NIS2 prompts are excluded from SFT selection and used only as non-regression
gates. No human source approval or independent sealed holdout is claimed.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path

from helios350_preview import TOKENIZER_SHA, trusted_asset
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_auto_sft_diagnostic import auto_prepare, _cpu_tree, atomic_save, sha256_file
from phoenix_auto_sft_public_dev_eval import pair_eval, exact_generation_eval
from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices
from real_gpu_preflight import json_manifest, verify_shards
from phoenix_master_precision import require_fp32_trainable_subset

HISTORICAL_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
PARAMETERS=349_825_152
SCHEMA="phoenix_replay_protected_auto_sft_v2"
CLASSIFICATION="replay_protected_fp32_master_auto_sft_public_dev_selected_not_promotable"
DOMAINS=("education","general","math_reasoning")
REPLAY_SEQUENCE=("main",)*9+DOMAINS
PAIR_CATEGORIES=("instruction_following","dialogue_state","truthfulness","robustness","compositional_reasoning")
CONFIGS=(
    {"blocks":4,"lr":2e-5,"steps":50,"replay_weight":0.10},
    {"blocks":4,"lr":2e-5,"steps":50,"replay_weight":0.25},
    {"blocks":4,"lr":2e-5,"steps":50,"replay_weight":0.50},
    {"blocks":4,"lr":1.5e-5,"steps":50,"replay_weight":0.15},
    {"blocks":4,"lr":1.5e-5,"steps":75,"replay_weight":0.25},
)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoint",action="store_true")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--private-root",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--main-data",type=Path,required=True)
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoint:p.error("explicit own-checkpoint opt-in required")
    if a.out.exists() or a.out.is_symlink() or not a.out.is_absolute():p.error("new absolute output required")

    source=require_pinned_checkpoint(a.checkpoint)
    if sha256_file(source)!=HISTORICAL_SHA: raise ValueError("historical source SHA changed")
    assets=a.assets_root.resolve(strict=True); private=a.private_root.resolve(strict=True)
    tokenizer_path=trusted_asset(assets,assets/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)
    repo_root=Path(__file__).resolve().parents[2]
    tokenizer,bos,eos,splits,data_report=auto_prepare(repo_root,a.archive,private,tokenizer_path)

    # Verify all replay bytes before Torch/CUDA.
    main_manifest=json_manifest(a.main_data.parent/"manifest.json")
    if main_manifest.get("benchmark_content_included") is not False:
        raise ValueError("main foundation manifest is not benchmark-separated")
    if main_manifest.get("tokenizer",{}).get("tokenizer_sha256")!=TOKENIZER_SHA:
        raise ValueError("main foundation tokenizer mismatch")
    main_shards=verify_shards(a.main_data,main_manifest.get("train_files"))
    replay_manifest=json_manifest(a.replay/"manifest.json")
    if replay_manifest.get("tokenizer_sha256")!=TOKENIZER_SHA or set(replay_manifest.get("domains",{}))!=set(DOMAINS):
        raise ValueError("replay manifest/tokenizer/domain mismatch")
    replay_shards=sum(verify_shards(a.replay/d,replay_manifest["domains"][d].get("files")) for d in DOMAINS)
    print(json.dumps({"event":"replay_bytes_verified","main_shards":main_shards,
      "replay_shards":replay_shards,"training_started":False},sort_keys=True),flush=True)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np, torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from protection_v9 import configure_trainable
    from train350_compiled import PersistentMMapPool
    guard.require_cuda_ready(torch)

    historical=_checked_historical_weights(source,torch,Neral350M,FRONTIER_350.to_dict())
    base_weights=historical["model"]; historical_tokens=int(historical.get("processed_tokens",-1))
    device=torch.device("cuda"); seq=512
    val_indices=validation_indices(len(splits["validation"]))
    pools={"main":PersistentMMapPool(a.main_data,FRONTIER_350.vocab_size)}
    pools.update({d:PersistentMMapPool(a.replay/d,FRONTIER_350.vocab_size) for d in DOMAINS})

    def sft_batch(rows,picks):
        selected=[rows[i] for i in picks]; width=dynamic_batch_width(selected,seq)
        x=torch.full((len(selected),width),eos,dtype=torch.long)
        y=torch.full((len(selected),width),-100,dtype=torch.long)
        for i,row in enumerate(selected):
            n=len(row["input_ids"]);x[i,:n]=torch.tensor(row["input_ids"],dtype=torch.long);y[i,:n]=torch.tensor(row["labels"],dtype=torch.long)
        return x.to(device),y.to(device)

    def fresh_model():
        m=Neral350M().to(device=device,dtype=torch.float32);m.load_state_dict(base_weights,strict=True)
        if m.parameter_report()["total"]!=PARAMETERS:raise RuntimeError("parameter contract changed")
        return m

    def eval_clone(m):
        q=Neral350M().to(device=device,dtype=torch.bfloat16);q.load_state_dict(m.state_dict(),strict=True);return q.eval()

    def validation(m):
        m.eval();vals=[]
        with torch.no_grad():
            for idx in val_indices:
                x,y=sft_batch(splits["validation"],[idx])
                with torch.autocast("cuda",dtype=torch.bfloat16):_,loss=m(x,y)
                if loss is None or not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite validation")
                vals.append((float(loss),int((y!=-100).sum().item())))
        m.train();return token_weighted_loss(vals)

    base_model=fresh_model().eval()
    baseline_val=validation(base_model)
    base_pair=pair_eval(torch,base_model,tokenizer,bos)
    base_exact=exact_generation_eval(base_model,tokenizer,bos,eos,torch)
    baseline={"validation_loss":baseline_val,"pair_overall_pct":base_pair["overall_pct"],
      "exact_generation_pct":base_exact["accuracy_pct"],
      "categories":{k:base_pair["categories"][k]["accuracy_pct"] for k in PAIR_CATEGORIES}}
    del base_model;torch.cuda.empty_cache()

    results=[];best_key=None;best_payload=None
    for index,cfg in enumerate(CONFIGS):
        guard.require_safe_runtime_temperature()
        torch.manual_seed(20260923);torch.cuda.manual_seed_all(20260923);random.seed(20260923);np.random.seed(20260923)
        torch.cuda.reset_peak_memory_stats()
        m=fresh_model();named=configure_trainable(m,cfg["blocks"]);params=[p for _,p in named]
        opt=torch.optim.AdamW(params,lr=cfg["lr"],betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=True)
        require_fp32_trainable_subset(m,opt,torch,require_moments=False)
        sft_rng=random.Random(20260923);replay_rng=random.Random(20260924);supervised=0;replay_tokens=0;start=time.monotonic()
        m.train()
        for step in range(1,cfg["steps"]+1):
            picks=[sft_rng.randrange(len(splits["train"])) for _ in range(4)]
            sx,sy=sft_batch(splits["train"],picks)
            source_name=REPLAY_SEQUENCE[(step-1)%len(REPLAY_SEQUENCE)]
            rx,ry=pools[source_name].batch(1,512,device,replay_rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                _,sloss=m(sx,sy);_,rloss=m(rx,ry)
            if sloss is None or rloss is None or not bool(torch.isfinite(sloss)) or not bool(torch.isfinite(rloss)):
                raise FloatingPointError("nonfinite SFT/replay loss")
            w=cfg["replay_weight"];den=1.0+w
            (sloss/den).backward();(rloss*(w/den)).backward()
            grad=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            if not math.isfinite(float(grad)):raise FloatingPointError("nonfinite combined gradient")
            opt.step()
            if step==1: require_fp32_trainable_subset(m,opt,torch,require_moments=True)
            supervised+=int((sy!=-100).sum().item());replay_tokens+=512
            if step%25==0 or step==cfg["steps"]:
                torch.cuda.synchronize();guard.require_safe_runtime_temperature()
        elapsed=time.monotonic()-start
        val=validation(m);q=eval_clone(m)
        pair=pair_eval(torch,q,tokenizer,bos);exact=exact_generation_eval(q,tokenizer,bos,eos,torch)
        del q;torch.cuda.empty_cache()
        cats={k:pair["categories"][k]["accuracy_pct"] for k in PAIR_CATEGORIES}
        no_reg=all(cats[k]>=baseline["categories"][k] for k in PAIR_CATEGORIES)
        qualifies=(val<baseline_val and pair["overall_pct"]>=baseline["pair_overall_pct"] and no_reg
                   and exact["accuracy_pct"]>baseline["exact_generation_pct"])
        row={"index":index,**cfg,"validation_loss":val,"validation_delta":val-baseline_val,
          "pair_overall_pct":pair["overall_pct"],"exact_generation_pct":exact["accuracy_pct"],
          "categories":cats,"no_pair_category_regression":no_reg,"qualifies":qualifies,
          "supervised_tokens":supervised,"foundation_replay_tokens":replay_tokens,
          "training_wall_seconds":elapsed,"peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1)}
        results.append(row);print(json.dumps({"event":"replay_sft_candidate",**row},sort_keys=True),flush=True)
        key=(exact["accuracy_pct"],pair["overall_pct"],-val)
        if qualifies and (best_key is None or key>best_key):
            best_key=key;best_payload={"selected":row,
              "model":{n:t.detach().to("cpu",dtype=torch.float32) for n,t in m.state_dict().items()},
              "optimizer":_cpu_tree(torch,opt.state_dict()),"sft_rng":sft_rng.getstate(),"replay_rng":replay_rng.getstate(),
              "torch_rng":torch.get_rng_state(),"cuda_rng":[v.cpu() for v in torch.cuda.get_rng_state_all()]}
        del opt,params,named,m;torch.cuda.empty_cache()

    a.out.mkdir(parents=True,exist_ok=False);candidate_sha=None;selected=None
    if best_payload is not None:
        selected=best_payload["selected"]
        state={"checkpoint_schema":SCHEMA,"classification":CLASSIFICATION,"model":best_payload["model"],
          "optimizer":best_payload["optimizer"],"sft_rng":best_payload["sft_rng"],"replay_rng":best_payload["replay_rng"],
          "torch_rng":best_payload["torch_rng"],"cuda_rng":best_payload["cuda_rng"],"config":FRONTIER_350.to_dict(),
          "tokenizer_sha256":TOKENIZER_SHA,"base_checkpoint_sha256":HISTORICAL_SHA,
          "historical_pretraining_tokens":historical_tokens,"source_archive_sha256":data_report["source_archive_sha256"],
          "foundation_main_manifest_sha256":sha256_file(a.main_data.parent/"manifest.json"),
          "foundation_replay_manifest_sha256":sha256_file(a.replay/"manifest.json"),"selected":selected,
          "master_parameter_dtype":"float32","adamw_moment_dtype":"float32",
          "compute_autocast_dtype":"bfloat16",
          "human_review_complete":False,"independent_holdout_checked":False,"production_eligible":False}
        cp=a.out/"candidate_replay_protected_auto_sft.pt";atomic_save(torch,state,cp);candidate_sha=sha256_file(cp)

    report={"event":"phoenix_replay_protected_sft_sweep_finished","classification":CLASSIFICATION,
      "baseline":baseline,"configs":results,"qualified_count":sum(r["qualifies"] for r in results),
      "selected":selected,"candidate_checkpoint_written":candidate_sha is not None,"candidate_sha256":candidate_sha,
      "master_parameter_dtype":"float32","adamw_moment_dtype":"float32","compute_autocast_dtype":"bfloat16",
      "human_review_complete":False,"independent_holdout_checked":False,"production_eligible":False}
    (a.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"event":report["event"],"baseline":baseline,"qualified_count":report["qualified_count"],
      "selected":selected,"candidate_sha256":candidate_sha,"production_eligible":False},sort_keys=True),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
