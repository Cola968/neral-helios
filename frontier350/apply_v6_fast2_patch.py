from __future__ import annotations

"""Experimental V6 Fast2 execution patch.

Fast2 keeps model architecture, sampled tokens, optimizer, LR/replay recipe, 5M health
checks, SAFE gates and rollback semantics unchanged. It only moves the existing exact
weighted replay+main CE objective inside one compiled graph for fused replay-only stages.

PyTorch 2.8 CUDA Graph replay remains disabled for this fused objective.  RoPE is
prewarmed on the model's actual CUDA device (normally cuda:0), matching the cache key
used by forward execution.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f"V6 Fast2 patch anchor mismatch for {label}: {text.count(old)}")
    return text.replace(old, new, 1)


def patch_trainer(text: str) -> str:
    text = once(
        text,
        "from model350 import Neral350M,FRONTIER_350\n",
        "from model350 import Neral350M,FRONTIER_350,_rope_table\n",
        "RoPE cache helper import",
    )

    old_run = "    run=model if args.compile_mode=='none' else torch.compile(model,fullgraph=True,dynamic=False,mode=args.compile_mode)\n"
    new_run = """    # Populate the immutable RoPE table before compilation on the exact device used
    # by model tensors. torch.device('cuda') has index=None and is a different cache key
    # from cuda:0, so using the actual model device is required for true prewarming.
    if args.compile_mode!='none':
        _rope_table(FRONTIER_350.max_seq_len,FRONTIER_350.head_dim,
                    FRONTIER_350.rope_theta,next(model.parameters()).device,torch.bfloat16)

    # Fast2 compiles the exact fused objective with TorchInductor/autotune but explicitly
    # without CUDA Graphs. This path measured consistently faster on the RTX 4070 while
    # avoiding graph-lifetime coupling to process-global cached tensors.
    if args.fused_replay_main:
        replay_rows_const=batch*3
        def _v6_fused_objective(x_all,y_all):
            logits,_=model(x_all,None)
            rtargets=y_all[:replay_rows_const]
            mtargets=y_all[replay_rows_const:]
            rloss=torch.nn.functional.cross_entropy(
                logits[:replay_rows_const].reshape(-1,logits.size(-1)),
                rtargets.reshape(-1),ignore_index=-100,reduction='sum')
            mloss=torch.nn.functional.cross_entropy(
                logits[replay_rows_const:].reshape(-1,logits.size(-1)),
                mtargets.reshape(-1),ignore_index=-100,reduction='sum')
            replay_valid=rtargets.ne(-100).sum(dtype=torch.float32)
            main_valid=mtargets.ne(-100).sum(dtype=torch.float32)
            denom=main_valid+args.replay_boost*replay_valid
            return (rloss*args.replay_boost+mloss)/denom
        fused_run=_v6_fused_objective if args.compile_mode=='none' else torch.compile(
            _v6_fused_objective,fullgraph=True,dynamic=False,mode='max-autotune-no-cudagraphs')
        run=model
    else:
        fused_run=None
        run=model if args.compile_mode=='none' else torch.compile(model,fullgraph=True,dynamic=False,mode=args.compile_mode)
"""
    text = once(text, old_run, new_run, "compiled fused objective declaration")

    old_loss = """            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,_=run(x_all,None)
                rloss=torch.nn.functional.cross_entropy(
                    logits[:replay_rows].reshape(-1,logits.size(-1)),
                    y_all[:replay_rows].reshape(-1),ignore_index=-100,reduction='sum')
                if main_counts:
                    mloss=torch.nn.functional.cross_entropy(
                        logits[replay_rows:].reshape(-1,logits.size(-1)),
                        y_all[replay_rows:].reshape(-1),ignore_index=-100,reduction='sum')
                else:
                    mloss=rloss.new_zeros(())
                fused_loss=(rloss*args.replay_boost+mloss)/weight_denominator
"""
    new_loss = """            with torch.autocast('cuda',dtype=torch.bfloat16):
                fused_loss=fused_run(x_all,y_all)
"""
    text = once(text, old_loss, new_loss, "compiled fused objective use")

    return text


def main() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    patched = patch_trainer(trainer)
    if patched != trainer:
        TRAINER.write_text(patched, encoding="utf-8")
    print("V6 Fast2 patch prepared: exact-device RoPE prewarm + no-cudagraph fused objective; semantic gates unchanged")


if __name__ == "__main__":
    main()
