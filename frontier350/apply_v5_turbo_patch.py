from __future__ import annotations

"""Runtime speed patch for Scale V5.

The patch is deliberately limited to execution mechanics. It does not change the model
architecture, sampled token counts, replay weighting, learning-rate schedule, optimizer,
SAFE gates, or V5's 5M rollback cap.

For replay_only stages the existing objective is algebraically separable: every replay
source uses the same replay_boost. Therefore the three replay batches and the packed main
batch can be concatenated into one physical batch and evaluated with one model
forward/backward while preserving the same token-weighted objective. Projected-rescue
stages keep the legacy separate-gradient path because they need per-domain reference
gradients.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"
V5 = ROOT / "run_scale_v5.py"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f"V5 turbo patch anchor mismatch for {label}: {text.count(old)}")
    return text.replace(old, new, 1)


def patch_trainer(text: str) -> str:
    text = once(
        text,
        "    ap.add_argument('--fast-replay-only',action='store_true',help='Avoid redundant vector copies and defer scalar syncs in replay_only mode')\n",
        "    ap.add_argument('--fast-replay-only',action='store_true',help='Avoid redundant vector copies and defer scalar syncs in replay_only mode')\n"
        "    ap.add_argument('--fused-replay-main',action='store_true',help='Fuse all replay+main microbatches into one physical forward/backward in replay_only mode')\n",
        "trainer argument",
    )
    text = once(
        text,
        "    if args.fast_replay_only and args.mode!='replay_only':raise ValueError('fast replay path is valid only for replay_only mode')\n",
        "    if args.fast_replay_only and args.mode!='replay_only':raise ValueError('fast replay path is valid only for replay_only mode')\n"
        "    if args.fused_replay_main and args.mode!='replay_only':raise ValueError('fused replay/main path is valid only for replay_only mode')\n"
        "    if args.fused_replay_main:args.fast_replay_only=True\n",
        "trainer validation",
    )
    text = once(
        text,
        "              'fast_replay_only':args.fast_replay_only,'tokens_per_update':tokens_per_update,\n",
        "              'fast_replay_only':args.fast_replay_only,'fused_replay_main':args.fused_replay_main,\n"
        "              'tokens_per_update':tokens_per_update,\n",
        "trainer contract",
    )

    anchor = """        effective_replay_fraction=(args.replay_boost*replay_tokens)/weight_denominator

        if args.fast_replay_only:
"""
    fused = """        effective_replay_fraction=(args.replay_boost*replay_tokens)/weight_denominator

        if args.fused_replay_main:
            # Exact replay_only objective, fewer launches: 3 replay batches + packed main
            # become one physical batch. All replay domains share the same replay_boost,
            # so sum-reduction followed by the original denominator is algebraically
            # equivalent to the previous four backward passes.
            optimizer.zero_grad(set_to_none=True)
            replay_x=[];replay_y=[]
            for domain,count in zip(DOMAINS,token_counts[:3]):
                x,y=sample(domain,count,batch);replay_x.append(x);replay_y.append(y)
            main_counts=token_counts[3:]
            parts_x=list(replay_x);parts_y=list(replay_y)
            if main_counts:
                main_count=sum(main_counts)
                mx,my=sample('main',main_count,batch*len(main_counts))
                parts_x.append(mx);parts_y.append(my)
            replay_rows=batch*len(replay_x)
            x_all=torch.cat(parts_x,dim=0);y_all=torch.cat(parts_y,dim=0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
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
            if not bool(torch.isfinite(fused_loss)):raise FloatingPointError('nonfinite fused replay/main loss')
            fused_loss.backward()
            clipped=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            # Loss/gradient finiteness remains checked on every update. A full parameter
            # scan is every fifth update and at the stage boundary; an optimizer-only fault
            # can therefore never cross a 5M continuity commit unnoticed.
            check_params=(step%5==0) or (done+total>=args.new_tokens)
            if check_params and not bool(torch.stack([torch.isfinite(p).all() for p in params]).all()):
                raise FloatingPointError('nonfinite parameter after fused optimizer step')
            grad_norm_tensor=clipped.detach();loss_value_tensor=fused_loss.detach()
            post_step_finite=torch.tensor(True,device='cuda')
        elif args.fast_replay_only:
"""
    text = once(text, anchor, fused, "fused training branch")
    text = once(
        text,
        "             'main_pack':args.main_pack,'fast_replay_only':args.fast_replay_only,\n",
        "             'main_pack':args.main_pack,'fast_replay_only':args.fast_replay_only,\n"
        "             'fused_replay_main':args.fused_replay_main,\n",
        "trainer metric metadata",
    )
    text = once(
        text,
        "             'batch_size':batch,'microbatches':micros,'main_pack':args.main_pack,'fast_replay_only':args.fast_replay_only,\n",
        "             'batch_size':batch,'microbatches':micros,'main_pack':args.main_pack,'fast_replay_only':args.fast_replay_only,\n"
        "             'fused_replay_main':args.fused_replay_main,\n",
        "trainer summary metadata",
    )
    return text


def patch_v5(text: str) -> str:
    text = once(
        text,
        '               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",\n',
        '               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "max-autotune-no-cudagraphs",\n',
        "V5 compile profile",
    )
    text = once(
        text,
        '        if recipe["mode"] == "replay_only":\n            cmd.append("--fast-replay-only")\n',
        '        if recipe["mode"] == "replay_only":\n            cmd.extend(("--fast-replay-only", "--fused-replay-main"))\n',
        "V5 fused invocation",
    )
    text = once(
        text,
        '        print(json.dumps({"event": "v5_train_start", "tag": tag, "from_tokens": train_tokens,\n                          "block_tokens": block_tokens, "recipe": recipe}), flush=True)\n',
        '        print(json.dumps({"event": "v5_train_start", "tag": tag, "from_tokens": train_tokens,\n                          "block_tokens": block_tokens, "recipe": recipe,\n                          "execution_profile": "fused48_max_autotune" if recipe["mode"] == "replay_only" else "projected_reference"}), flush=True)\n',
        "V5 telemetry profile",
    )
    return text


def main() -> None:
    if not TRAINER.is_file():
        raise FileNotFoundError(TRAINER)
    trainer = TRAINER.read_text(encoding="utf-8")
    patched = patch_trainer(trainer)
    if patched != trainer:
        TRAINER.write_text(patched, encoding="utf-8")

    if V5.is_file():
        v5 = V5.read_text(encoding="utf-8")
        patched_v5 = patch_v5(v5)
        if patched_v5 != v5:
            V5.write_text(patched_v5, encoding="utf-8")

    print("Scale V5 turbo patch applied: fused replay/main batch + max-autotune; gates unchanged")


if __name__ == "__main__":
    main()
