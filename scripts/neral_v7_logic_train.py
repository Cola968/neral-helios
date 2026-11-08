"""Explicit manual NERAL v7 logic QLoRA research pilot (never promotes a model)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts.neral_v7_logic_pilot import AUTH,STATUS,build_logic_experiment


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",choices=("cpu-contract","train-local-gpu"),required=True)
    p.add_argument("--base",type=Path)
    p.add_argument("--v5-adapter",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--authorize",default="")
    a=p.parse_args(argv)
    if a.mode=="cpu-contract":
        _,_,meta=build_logic_experiment()
        print(json.dumps({
            "status":meta["status"],"train_rows":meta["train_rows"],
            "validation_rows":meta["validation_rows"],
            "source_counts":meta["source_counts"],"train_sha256":meta["train_sha256"],
            "release_validated":False,"training_performed":False,
        },sort_keys=True))
        return
    if a.authorize!=AUTH or not a.base or not a.v5_adapter or not a.out:
        p.error("Explicit experimental GPU permit and pinned model paths required")
    from scripts.neral_v6_r16_gpu import _gpu_train
    result=_gpu_train(a,data_factory=build_logic_experiment,steps=120,
                      lr=3e-6,authorize=AUTH,
                      status="NERAL_V7_LOGIC_R16_120_STEPS_UNRELEASED",
                      checkpoint_interval=20)
    print(json.dumps({
        "status":result["status"],"steps_completed":result["steps_completed"],
        "candidate_adapter_sha256":result["candidate_adapter_sha256"],
        "checkpoint_selected_step":result["checkpoint_selected_step"],
        "checkpoint_mode":result["checkpoint_mode"],
        "peak_reserved_vram_mib":result["peak_reserved_vram_mib"],
        "release_validated":False,"adapter_promoted":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
