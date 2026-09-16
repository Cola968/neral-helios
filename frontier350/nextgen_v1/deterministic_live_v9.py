"""Short GPU reproducibility diagnostic, not a production training entrypoint.

Set cuBLAS determinism BEFORE CUDA is initialized. The existing V9 trainer sets
its own math-precision policy; this wrapper does not claim exact equality until
two independent real checkpoints are compared. All arguments go to V9 unchanged.
"""
import json
import os

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before Python starts")

import torch
import live_v9


def main():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic algorithms not enabled")
    print(json.dumps({
        "event": "determinism_setup",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "note": "V9 manages its own TF32 policy; equality is established ONLY by comparing checkpoints",
    }, sort_keys=True), flush=True)
    live_v9.trainer.append_jsonl = live_v9.append_and_stream
    live_v9.trainer.main()


if __name__ == "__main__":
    main()
