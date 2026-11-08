"""Manual local QLoRA: NERAL v8 self-verification SFT, never a release.

Only the exact user-authorized, isolated 240-step experiment can train. The
v5 base adapter and original Qwen foundation are read-only SHA verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.neral_v8_selfverify_data import (
    V8_AUTH,V8_STATUS,build_selfverify_experiment,
)

MODEL_STATUS="NERAL_V8_SELFVERIFY_R16_240_UNRELEASED"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",required=True,choices=("cpu-contract","train-local-gpu"))
    p.add_argument("--base",type=Path)
    p.add_argument("--v5-adapter",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--authorize",default="")
    a=p.parse_args(argv)
    if a.mode=="cpu-contract":
        train,val,meta=build_selfverify_experiment()
        print(json.dumps({
            "status":meta["status"],
            "train_rows":len(train),
            "validation_rows":len(val),
            "train_sha256":meta["train_sha256"],
            "validation_sha256":meta["validation_sha256"],
            "source_counts":meta["source_counts"],
            "wrong_claims_verified":meta["self_verification_wrong_proposal_examples"],
            "training_performed":False,
            "release_validated":False,
        },sort_keys=True))
        return
    if a.authorize!=V8_AUTH or not all((a.base,a.v5_adapter,a.out)):
        p.error("Exact authorization and pinned private paths required")
    from scripts.neral_v6_r16_gpu import _gpu_train
    result=_gpu_train(a,data_factory=build_selfverify_experiment,
                      steps=240,lr=4e-6,authorize=V8_AUTH,
                      status=MODEL_STATUS,checkpoint_interval=40,
                      max_steps_limit=240)
    print(json.dumps({
        "status":result["status"],
        "steps_completed":result["steps_completed"],
        "selected_checkpoint":result["checkpoint_selected_step"],
        "predecessor_adapter_sha256":result["predecessor_adapter_sha256"],
        "candidate_adapter_sha256":result["candidate_adapter_sha256"],
        "train_sha256":result["train_sha256"],
        "validation_sha256":result["validation_sha256"],
        "validation_loss":result["validation_loss"],
        "peak_reserved_vram_mib":result["peak_reserved_vram_mib"],
        "training_performed":True,
        "release_validated":False,"adapter_promoted":False,
        "independent_transfer_proven":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
