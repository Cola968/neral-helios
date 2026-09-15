from __future__ import annotations

"""Runtime-only V7.1 fixes discovered by the first RTX 4070 smoke run.

This patch is deliberately additive. It does not change model weights, sampled-token
semantics, optimizer math, LR/replay recipes, V5/V6 execution profiles, health gates or
V7 recovery cadence.

Fixes:
- guard the legacy 5-step print with ``metric_now`` so deferred fused telemetry never
  references ``row`` before that row is created;
- mark a new CUDA-Graph step before the manual fused max-autotune forward. PyTorch 2.8
  explicitly requires this when graph-owned outputs/cached tensors would otherwise be
  treated as overwritten on the next invocation. Fast2's no-cudagraph fused objective is
  unaffected.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"


def patch_trainer(text: str) -> str:
    old_print = "        if step%5==0 or done==args.new_tokens:print(json.dumps(row),flush=True)\n"
    new_print = "        if metric_now and (step%5==0 or done==args.new_tokens):print(json.dumps(row),flush=True)\n"
    if new_print not in text:
        if text.count(old_print) != 1:
            raise RuntimeError(f"V7.1 hotfix print anchor mismatch: {text.count(old_print)}")
        text = text.replace(old_print, new_print, 1)

    old_forward = (
        "            with torch.autocast('cuda',dtype=torch.bfloat16):\n"
        "                logits,_=run(x_all,None)\n"
    )
    new_forward = (
        "            if args.compile_mode=='max-autotune' and hasattr(torch.compiler,'cudagraph_mark_step_begin'):\n"
        "                torch.compiler.cudagraph_mark_step_begin()\n"
        "            with torch.autocast('cuda',dtype=torch.bfloat16):\n"
        "                logits,_=run(x_all,None)\n"
    )
    if new_forward not in text and old_forward in text:
        text = text.replace(old_forward, new_forward, 1)
    elif new_forward not in text and old_forward not in text:
        # Fast2 replaces the manual fused forward with a no-cudagraph compiled objective.
        # In that case no CUDA-Graph marker is necessary.
        if "fused_loss=fused_run(x_all,y_all)" not in text:
            raise RuntimeError("V7.1 hotfix could not identify manual or Fast2 fused objective")

    return text


def main() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    patched = patch_trainer(trainer)
    if patched != trainer:
        TRAINER.write_text(patched, encoding="utf-8")
    print("V7.1 runtime hotfix applied: deferred-row guard + CUDA-Graph step marker")


if __name__ == "__main__":
    main()
