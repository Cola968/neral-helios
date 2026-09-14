from __future__ import annotations

"""Experimental V6 Fast2 execution patch.

Prepared while the current guarded V6 run continues. It is intentionally NOT wired into
an already-running job: GitHub Actions checks out a fixed commit, so hot-patching that
process would be unsafe.

Fast2 keeps model architecture, sampled tokens, optimizer, LR/replay recipe, 5M health
checks, SAFE gates and rollback semantics unchanged. It only moves the existing exact
weighted replay+main CE objective inside the compiled graph for fused replay-only stages.
The launcher can still fall back to the non-fused safe profile if compilation/runtime
fails.
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
    old_run = "    run=model if args.compile_mode=='none' else torch.compile(model,fullgraph=True,dynamic=False,mode=args.compile_mode)\n"
    new_run = """    # Fast2: for fused replay-only stages compile model+weighted CE together so the
    # head, slice/reduction and CE stay in one graph. Non-fused/projected paths retain the
    # existing compiled-model behavior.
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
            # Derive the exact denominator from the same -100 mask used by CE. This is
            # algebraically identical to main_tokens + replay_boost*replay_tokens and
            # keeps the final partial update shape-static.
            replay_valid=rtargets.ne(-100).sum(dtype=torch.float32)
            main_valid=mtargets.ne(-100).sum(dtype=torch.float32)
            denom=main_valid+args.replay_boost*replay_valid
            return (rloss*args.replay_boost+mloss)/denom
        fused_run=_v6_fused_objective if args.compile_mode=='none' else torch.compile(
            _v6_fused_objective,fullgraph=True,dynamic=False,mode=args.compile_mode)
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

    # Fast2 only changes graph placement. Keep the existing telemetry/finite-scan cadence
    # supplied by apply_v6_fast_patch.py and all continuity gates unchanged.
    return text


def main() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    patched = patch_trainer(trainer)
    if patched != trainer:
        TRAINER.write_text(patched, encoding="utf-8")
    print("V6 Fast2 patch prepared: compiled exact fused objective; gates/objective unchanged")


if __name__ == "__main__":
    main()
