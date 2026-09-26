"""Short full-model learning-rate probe for Neral 350M continued pretraining.

Development only. Every LR starts from the identical immutable historical weights,
sees the same deterministic 262,144-token sample order, and uses a fresh AdamW.
Selection uses refresh validation plus the public NIS2 pair suite only; no candidate
checkpoint is written or promoted.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helios350_preview import TOKENIZER_SHA, trusted_asset
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_foundation_refresh_dev import verify_refresh_manifest
from phoenix_auto_sft_public_dev_eval import pair_eval

SEQ = 256
MICROBATCH = 4
GRAD_ACCUM = 1
TOKENS_PER_UPDATE = SEQ * MICROBATCH
PROBE_TOKENS = 262_144
UPDATES = PROBE_TOKENS // TOKENS_PER_UPDATE
LRS = (5e-6, 1e-5, 2e-5, 3e-5)
PAIR_CATEGORIES = (
    "instruction_following","dialogue_state","truthfulness",
    "robustness","compositional_reasoning",
)


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoint",action="store_true")
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoint:
        p.error("explicit own-checkpoint opt-in required")
    if not a.out.is_absolute() or a.out.exists():
        p.error("new absolute report required")

    source=require_pinned_checkpoint(a.source_checkpoint)
    assets=a.assets_root.resolve(strict=True)
    tok_path=trusted_asset(
        assets, assets/"frontier350_assets"/"tokenizer.json", TOKENIZER_SHA
    )
    refresh=verify_refresh_manifest(a.corpus_root,a.manifest_sha256,TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from model350 import FRONTIER_350,Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from train350_compiled import PersistentMMapPool

    guard.require_cuda_ready(torch)
    base=_checked_historical_weights(source,torch,Neral350M,FRONTIER_350.to_dict())
    base_weights=base["model"]
    del base
    train_pool=PersistentMMapPool(refresh["root"]/"train",FRONTIER_350.vocab_size)
    valid_pool=PersistentMMapPool(refresh["root"]/"valid",FRONTIER_350.vocab_size)
    tokenizer=Tokenizer.from_file(str(tok_path))
    bos=tokenizer.token_to_id("<|bos|>")
    if bos is None:
        raise ValueError("missing BOS")
    device=torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision("high")

    def validation(model):
        rng=random.Random(20261999)
        model.eval()
        vals=[]
        with torch.no_grad():
            for _ in range(16):
                x,y=valid_pool.batch(4,SEQ,device,rng)
                _,loss=model(x,y)
                if loss is None or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite LR-probe validation")
                vals.append(float(loss))
        model.train()
        return sum(vals)/len(vals)

    def fresh_model():
        model=Neral350M().to(device=device,dtype=torch.bfloat16)
        model.set_gradient_checkpointing(False)
        model.load_state_dict(base_weights,strict=True)
        return model

    baseline_model=fresh_model()
    baseline_validation=validation(baseline_model)
    baseline_pair=pair_eval(torch,baseline_model,tokenizer,int(bos))
    baseline_categories={k:baseline_pair["categories"][k]["accuracy_pct"]
                         for k in PAIR_CATEGORIES}
    baseline_summary={
        "validation_loss":baseline_validation,
        "pair_overall_pct":baseline_pair["overall_pct"],
        "categories":baseline_categories,
    }
    del baseline_model
    torch.cuda.empty_cache()

    results=[]
    for lr in LRS:
        guard.require_safe_runtime_temperature()
        torch.manual_seed(20260923)
        torch.cuda.manual_seed_all(20260923)
        random.seed(20260923)
        np.random.seed(20260923)
        rng=random.Random(20260924)
        torch.cuda.reset_peak_memory_stats()
        model=fresh_model()
        optimizer=torch.optim.AdamW(
            model.parameters(),lr=lr,betas=(0.9,0.95),eps=1e-8,
            weight_decay=0.1,fused=True
        )
        model.train()
        started=time.perf_counter()
        last_loss=None
        for step in range(UPDATES):
            optimizer.zero_grad(set_to_none=True)
            x,y=train_pool.batch(MICROBATCH,SEQ,device,rng)
            _,loss=model(x,y)
            if loss is None or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite LR-probe train loss")
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(
                model.parameters(),1.0,error_if_nonfinite=True
            )
            if not math.isfinite(float(grad)):
                raise FloatingPointError("nonfinite LR-probe grad")
            optimizer.step()
            last_loss=float(loss.detach())
            if (step+1)%64==0:
                torch.cuda.synchronize()
                guard.require_safe_runtime_temperature()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        val=validation(model)
        pair=pair_eval(torch,model,tokenizer,int(bos))
        categories={k:pair["categories"][k]["accuracy_pct"] for k in PAIR_CATEGORIES}
        no_pair_regression=all(
            categories[k]>=baseline_categories[k] for k in PAIR_CATEGORIES
        )
        row={
            "lr":lr,
            "new_tokens":PROBE_TOKENS,
            "optimizer_steps":UPDATES,
            "last_train_loss":last_loss,
            "validation_loss":val,
            "validation_delta":val-baseline_validation,
            "pair_overall_pct":pair["overall_pct"],
            "categories":categories,
            "no_pair_category_regression":no_pair_regression,
            "wall_seconds":elapsed,
            "tokens_per_second":PROBE_TOKENS/elapsed,
            "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
        }
        print(json.dumps({"event":"foundation_lr_candidate",**row},sort_keys=True),flush=True)
        results.append(row)
        del optimizer,model
        gc.collect()
        torch.cuda.empty_cache()

    eligible=[r for r in results
              if r["validation_loss"]<baseline_validation
              and r["no_pair_category_regression"]
              and r["pair_overall_pct"]>=baseline_pair["overall_pct"]]
    selected=min(eligible,key=lambda r:r["validation_loss"]) if eligible else None
    report={
        "event":"phoenix_foundation_lr_probe_finished",
        "classification":"PUBLIC_DEVELOPMENT_HYPERPARAMETER_PROBE_NOT_HOLDOUT",
        "parameters":349_825_152,
        "layers":24,
        "probe_tokens_per_lr":PROBE_TOKENS,
        "baseline":baseline_summary,
        "results":results,
        "selected":selected,
        "checkpoint_written":False,
        "production_eligible":False,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
