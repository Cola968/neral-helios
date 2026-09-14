from __future__ import annotations

"""Runtime speed patch for Scale V5.

Execution-only optimizations. The model architecture, sampled token counts, replay
weighting, LR schedule, optimizer, SAFE gates and V5's 5M rollback cap stay unchanged.

Turbo profiles:
- turbo60: fused replay+main physical batch + TorchInductor max-autotune/cudagraph path.
- turbo50: same exact fused objective with max-autotune-no-cudagraphs.
- safe: legacy fast replay_only path with default compile.

The launcher automatically falls back when a faster profile is unsupported/OOM. Fused
replay/main is algebraically equivalent for replay_only because all replay domains share
the same replay_boost. Projected rescue keeps the reference-gradient path.
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
        "trainer fused argument",
    )
    text = once(
        text,
        "    ap.add_argument('--compile-mode',choices=('none','default','reduce-overhead','max-autotune-no-cudagraphs'),default='default')\n",
        "    ap.add_argument('--compile-mode',choices=('none','default','reduce-overhead','max-autotune-no-cudagraphs','max-autotune'),default='default')\n",
        "trainer max-autotune mode",
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
            # Exact replay_only objective, fewer launches: three replay batches plus the
            # packed main batch become one physical batch. The loss keeps the original
            # token weighting and replay_boost denominator.
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
            # Full parameter scan only at the telemetry boundary and stage endpoint.
            # Loss/gradient checks remain per update; the 5M commit gate remains intact.
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
        "    start=time.perf_counter();start_done=done;start_step=step\n",
        "    start=time.perf_counter();start_done=done;start_step=step\n"
        "    metric_tick=start;metric_done=done\n",
        "deferred metric timer",
    )

    old_metrics = """        done+=total;step+=1
        for i,count in enumerate(token_counts):counts[DOMAINS[i] if i<3 else 'main']+=count
        torch.cuda.synchronize()
        if args.fast_replay_only:
            if not bool(post_step_finite):raise FloatingPointError('nonfinite parameter after optimizer step')
            loss_value=float(loss_value_tensor);grad_norm=float(grad_norm_tensor)
        elapsed=time.perf_counter()-tick
        row={'event':'train','step':step,'processed_tokens':int(source['processed_tokens'])+done,
             'new_tokens':done,'loss':loss_value,'grad_norm':grad_norm,'lr':current_lr,'seconds':elapsed,
             'tokens_per_second':total/elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'batch_size':batch,'microbatches':micros,
             'main_pack':args.main_pack,'fast_replay_only':args.fast_replay_only,
             'replay_boost':args.replay_boost,'effective_replay_fraction':effective_replay_fraction,
             'max_main_physical_batch':batch*min(args.main_pack,max(1,micros-3)),
             'tokens_per_update':total,'batch_chain':chain,**diagnostics}
        append_jsonl(out/'metrics.jsonl',row)
"""
    new_metrics = """        done+=total;step+=1
        for i,count in enumerate(token_counts):counts[DOMAINS[i] if i<3 else 'main']+=count
        # Fused mode only forces a host/GPU synchronization at the existing five-step
        # telemetry cadence (or endpoint). This removes four scalar-sync/file-write stalls
        # out of five without reducing numerical checks at a continuity boundary.
        metric_now=(not args.fused_replay_main) or step%5==0 or done==args.new_tokens
        if metric_now:
            torch.cuda.synchronize()
            if args.fast_replay_only:
                if not bool(post_step_finite):raise FloatingPointError('nonfinite parameter after optimizer step')
                loss_value=float(loss_value_tensor);grad_norm=float(grad_norm_tensor)
            if args.fused_replay_main:
                elapsed=time.perf_counter()-metric_tick
                measured_tokens=done-metric_done
                metric_tick=time.perf_counter();metric_done=done
            else:
                elapsed=time.perf_counter()-tick
                measured_tokens=total
            row={'event':'train','step':step,'processed_tokens':int(source['processed_tokens'])+done,
                 'new_tokens':done,'loss':loss_value,'grad_norm':grad_norm,'lr':current_lr,'seconds':elapsed,
                 'tokens_per_second':measured_tokens/elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'batch_size':batch,'microbatches':micros,
                 'main_pack':args.main_pack,'fast_replay_only':args.fast_replay_only,
                 'fused_replay_main':args.fused_replay_main,
                 'fused_physical_batch':batch*micros if args.fused_replay_main else None,
                 'replay_boost':args.replay_boost,'effective_replay_fraction':effective_replay_fraction,
                 'max_main_physical_batch':batch*min(args.main_pack,max(1,micros-3)),
                 'tokens_per_update':total,'batch_chain':chain,**diagnostics}
            append_jsonl(out/'metrics.jsonl',row)
"""
    text = once(text, old_metrics, new_metrics, "deferred fused telemetry")

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
        '        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),\n',
        '        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_v5_stage_launcher.py"),\n',
        "V5 reliable launcher",
    )
    text = once(
        text,
        '               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",\n',
        '               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "max-autotune",\n',
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
        '        print(json.dumps({"event": "v5_train_start", "tag": tag, "from_tokens": train_tokens,\n                          "block_tokens": block_tokens, "recipe": recipe,\n                          "execution_profile": "auto_turbo60_turbo50_safe" if recipe["mode"] == "replay_only" else "projected_reference"}), flush=True)\n',
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

    print("Scale V5 turbo patch applied: auto turbo60 -> turbo50 -> safe; gates unchanged")


if __name__ == "__main__":
    main()
