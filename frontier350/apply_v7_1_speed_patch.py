from __future__ import annotations

"""Apply Scale V7.1 execution-only speedups after V7 guardrails.

Invariants:
- model architecture/state-dict unchanged
- sampled RNG order and token values unchanged
- replay/main weighting and optimizer math unchanged
- all pre-step finite checks retained before parameter mutation
- V7 recovery, health gates and GPU guardrails unchanged

V7.1 adds an optional coalesced replay+main input path and replaces two host syncs in the
fused pre-step numerical gate with one combined boolean decision.  The input path is only
selected by the RTX benchmark gate; otherwise V7 behavior remains active.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"
CONTROLLER = ROOT / "run_scale_v5.py"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V7.1 patch anchor mismatch for {label}: {count}")
    return text.replace(old, new, 1)


def patch_trainer(text: str) -> str:
    text = once(
        text,
        "from v7_recovery import load_exact_recovery,restore_exact_recovery,save_exact_recovery\n",
        "from v7_recovery import load_exact_recovery,restore_exact_recovery,save_exact_recovery\n"
        "from v7_1_policy import update_batch_chain\n",
        "V7.1 policy import",
    )

    text = once(
        text,
        "    ap.add_argument('--v7-temp-every-updates',type=int,default=50)\n",
        "    ap.add_argument('--v7-temp-every-updates',type=int,default=50)\n"
        "    ap.add_argument('--v71-coalesced-input',action='store_true',help='Execution-only: sample fused replay+main into one host batch')\n"
        "    ap.add_argument('--v71-input-transfer',choices=('legacy-pin','direct-pageable'),default='legacy-pin')\n",
        "V7.1 trainer arguments",
    )

    text = once(
        text,
        "              'fast_replay_only':args.fast_replay_only,'fused_replay_main':args.fused_replay_main,\n",
        "              'fast_replay_only':args.fast_replay_only,'fused_replay_main':args.fused_replay_main,\n"
        "              'v71_coalesced_input':args.v71_coalesced_input,'v71_input_transfer':args.v71_input_transfer,\n",
        "V7.1 execution contract fields",
    )

    old_sample = """    def sample(domain,count,physical_batch):
        nonlocal chain
        arrays=pools[domain].arrays
        rows=np.empty((physical_batch,seq+1),dtype=np.int64)
        for i in range(physical_batch):
            array=rng.choice(arrays);offset=rng.randrange(len(array)-seq)
            rows[i]=array[offset:offset+seq+1]
        if rows.min()<0 or rows.max()>=FRONTIER_350.vocab_size:raise ValueError('invalid token id')
        chain=hashlib.sha256(bytes.fromhex(chain)+domain.encode()+count.to_bytes(8,'little')+rows.tobytes()).hexdigest()
        whole=torch.from_numpy(rows).pin_memory().to('cuda',non_blocking=True)
        return whole[:,:-1],mask_budget_targets(whole[:,1:],count)
"""
    new_sample = """    def _v71_transfer_rows(rows):
        host=torch.from_numpy(rows)
        if args.v71_input_transfer=='legacy-pin':host=host.pin_memory()
        return host.to('cuda',non_blocking=True)

    def _v71_fill_rows(domain,count,rows):
        nonlocal chain
        arrays=pools[domain].arrays
        for i in range(rows.shape[0]):
            array=rng.choice(arrays);offset=rng.randrange(len(array)-seq)
            rows[i]=array[offset:offset+seq+1]
        if rows.min()<0 or rows.max()>=FRONTIER_350.vocab_size:raise ValueError('invalid token id')
        chain=update_batch_chain(chain,domain,count,rows)

    def sample(domain,count,physical_batch):
        rows=np.empty((physical_batch,seq+1),dtype=np.int64)
        _v71_fill_rows(domain,count,rows)
        whole=_v71_transfer_rows(rows)
        return whole[:,:-1],mask_budget_targets(whole[:,1:],count)

    def v71_sample_fused(token_counts):
        segments=[]
        for domain,count in zip(DOMAINS,token_counts[:3]):segments.append((domain,count,batch))
        main_counts=token_counts[3:]
        if main_counts:segments.append(('main',sum(main_counts),batch*len(main_counts)))
        rows=np.empty((sum(n for _,_,n in segments),seq+1),dtype=np.int64)
        spans=[];cursor=0
        for domain,count,nrows in segments:
            view=rows[cursor:cursor+nrows]
            _v71_fill_rows(domain,count,view)
            spans.append((cursor,nrows,count));cursor+=nrows
        whole=_v71_transfer_rows(rows)
        x_all=whole[:,:-1].contiguous();y_all=whole[:,1:].contiguous()
        for start,nrows,count in spans:
            capacity=nrows*seq
            if count<capacity:y_all[start:start+nrows].reshape(-1)[count:]=-100
        return x_all,y_all,batch*3
"""
    text = once(text, old_sample, new_sample, "exact sampling/coalescing path")

    old_fused_input = """            replay_x=[];replay_y=[]
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
"""
    new_fused_input = """            main_counts=token_counts[3:]
            if args.v71_coalesced_input:
                x_all,y_all,replay_rows=v71_sample_fused(token_counts)
            else:
                replay_x=[];replay_y=[]
                for domain,count in zip(DOMAINS,token_counts[:3]):
                    x,y=sample(domain,count,batch);replay_x.append(x);replay_y.append(y)
                parts_x=list(replay_x);parts_y=list(replay_y)
                if main_counts:
                    main_count=sum(main_counts)
                    mx,my=sample('main',main_count,batch*len(main_counts))
                    parts_x.append(mx);parts_y.append(my)
                replay_rows=batch*len(replay_x)
                x_all=torch.cat(parts_x,dim=0);y_all=torch.cat(parts_y,dim=0)
"""
    text = once(text, old_fused_input, new_fused_input, "coalesced fused input")

    old_checks = """            if not bool(torch.isfinite(fused_loss)):raise FloatingPointError('nonfinite fused replay/main loss')
            fused_loss.backward()
            clipped=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
"""
    new_checks = """            # Preserve both finite conditions before optimizer mutation, but make one
            # host decision instead of synchronizing once for loss and again for grad norm.
            fused_loss.backward()
            clipped=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=False)
            pre_step_ok=torch.stack((torch.isfinite(fused_loss),torch.isfinite(clipped))).all()
            if not bool(pre_step_ok):raise FloatingPointError('nonfinite fused loss/gradient before optimizer step')
            optimizer.step()
"""
    text = once(text, old_checks, new_checks, "single pre-step host synchronization")
    return text


def patch_controller(text: str) -> str:
    return once(
        text,
        'str(Path(__file__).parent / "train_v7_stage_launcher.py")',
        'str(Path(__file__).parent / "train_v7_1_stage_launcher.py")',
        "V7.1 stage launcher",
    )


def main() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    TRAINER.write_text(patch_trainer(trainer), encoding="utf-8")
    CONTROLLER.write_text(patch_controller(controller), encoding="utf-8")
    print("Scale V7.1 applied: exact coalesced input option + one-sync fused numerical gate")


if __name__ == "__main__":
    main()
