from __future__ import annotations

"""Apply Scale V7 on top of the prepared V6 Fast/Fast2 runtime.

The patch is execution/recovery-only.  It does not alter model architecture, sampled-token
semantics, optimizer math, LR/replay recipes, health thresholds or SAFE release gates.
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
        raise RuntimeError(f"V7 patch anchor mismatch for {label}: {count}")
    return text.replace(old, new, 1)


def patch_trainer(text: str) -> str:
    text = once(
        text,
        "import argparse,hashlib,json,math,random,time\n",
        "import argparse,hashlib,json,math,random,time,subprocess\n",
        "trainer subprocess import",
    )
    text = once(
        text,
        "from recovery_identity import file_sha\n",
        "from recovery_identity import file_sha\n"
        "from v7_policy import semantic_resume_contract,gpu_guard_reason\n"
        "from v7_recovery import load_exact_recovery,restore_exact_recovery,save_exact_recovery\n",
        "V7 imports",
    )
    text = once(
        text,
        "    ap.add_argument('--resume-save-interval-tokens',type=int,default=0,help='Optional full resume-state save interval; 0 preserves legacy 20-update cadence')\n",
        "    ap.add_argument('--resume-save-interval-tokens',type=int,default=0,help='Optional full resume-state save interval; 0 preserves legacy 20-update cadence')\n"
        "    ap.add_argument('--v7-resume-state',default='',help='Compact exact V7 recovery state; execution profile may change')\n"
        "    ap.add_argument('--v7-recovery-interval-tokens',type=int,default=1000000,help='Compact exact recovery cadence; endpoint excluded')\n"
        "    ap.add_argument('--v7-max-reserved-mib',type=float,default=9216.0)\n"
        "    ap.add_argument('--v7-min-free-mib',type=float,default=1280.0)\n"
        "    ap.add_argument('--v7-max-temp-c',type=float,default=82.0)\n"
        "    ap.add_argument('--v7-temp-every-updates',type=int,default=50)\n",
        "V7 trainer arguments",
    )
    text = once(
        text,
        "    if args.snapshot_interval_tokens<0 or args.resume_save_interval_tokens<0:raise ValueError('checkpoint intervals must be >=0')\n",
        "    if args.snapshot_interval_tokens<0 or args.resume_save_interval_tokens<0:raise ValueError('checkpoint intervals must be >=0')\n"
        "    if args.v7_recovery_interval_tokens<=0:raise ValueError('V7 recovery interval must be positive')\n"
        "    if args.v7_temp_every_updates<=0:raise ValueError('V7 temperature cadence must be positive')\n"
        "    if args.resume and args.v7_resume_state:raise ValueError('choose one resume source')\n",
        "V7 argument validation",
    )
    text = once(
        text,
        "    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('use explicit resume')\n",
        "    if out.exists() and any(out.iterdir()) and not (args.resume or args.v7_resume_state):raise FileExistsError('use explicit resume')\n",
        "V7 resumable output directory",
    )
    text = once(
        text,
        "        if state['contract']!=contract:raise ValueError('resume contract differs')\n",
        "        if semantic_resume_contract(state['contract'])!=semantic_resume_contract(contract):raise ValueError('resume semantic contract differs')\n",
        "execution-profile-neutral full resume",
    )

    restore_anchor = "    total_parameters=sum(p.numel() for p in model.parameters())\n"
    restore_block = """    if args.v7_resume_state:
        recovery=load_exact_recovery(Path(args.v7_resume_state),expected_anchor_sha256=args.anchor_sha,contract=contract)
        done,step,chain,counts=restore_exact_recovery(recovery,named=named,optimizer=optimizer,rng=rng)
        if done>=args.new_tokens:raise RuntimeError('V7 recovery already at stage endpoint')
        print(json.dumps({'event':'v7_exact_resume','path':args.v7_resume_state,
                          'new_tokens':done,'processed_tokens':int(source['processed_tokens'])+done,
                          'updates':step,'compile_mode':args.compile_mode,
                          'fused_replay_main':args.fused_replay_main}),flush=True)

    total_parameters=sum(p.numel() for p in model.parameters())
"""
    text = once(text, restore_anchor, restore_block, "V7 compact resume restore")

    scheduled_anchor = "    def scheduled_lr(tokens_done):\n"
    recovery_setup = """    next_v7_recovery=((done//args.v7_recovery_interval_tokens)+1)*args.v7_recovery_interval_tokens

    def v7_save_recovery(reason):
        if done<=0 or done>=args.new_tokens:return None
        path=out/f'v7_recovery_{done:012d}.pt'
        if not path.exists():
            save_exact_recovery(path,named=named,optimizer=optimizer,rng=rng,contract=contract,
                anchor_sha256=args.anchor_sha,source_processed_tokens=int(source['processed_tokens']),
                done=done,step=step,chain=chain,counts=counts)
        print(json.dumps({'event':'v7_recovery_snapshot','reason':reason,'path':str(path),
                          'new_tokens':done,'processed_tokens':int(source['processed_tokens'])+done}),flush=True)
        return path

    def v7_gpu_temp():
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],
                                        text=True,timeout=2).splitlines()[0].strip()
            return float(raw)
        except Exception:
            return None

    def scheduled_lr(tokens_done):
"""
    text = once(text, scheduled_anchor, recovery_setup, "V7 recovery helpers")

    progress_anchor = "        for i,count in enumerate(token_counts):counts[DOMAINS[i] if i<3 else 'main']+=count\n"
    progress_block = """        for i,count in enumerate(token_counts):counts[DOMAINS[i] if i<3 else 'main']+=count
        if done<args.new_tokens and done>=next_v7_recovery:
            v7_save_recovery('interval')
            while next_v7_recovery<=done:next_v7_recovery+=args.v7_recovery_interval_tokens
"""
    text = once(text, progress_anchor, progress_block, "V7 interval recovery")

    metric_anchor = "        if metric_now:\n            torch.cuda.synchronize()\n"
    metric_block = """        if metric_now:
            torch.cuda.synchronize()
            if done<args.new_tokens:
                free_bytes,_total_bytes=torch.cuda.mem_get_info()
                reserved_mib=torch.cuda.memory_reserved()/1048576.0
                free_mib=free_bytes/1048576.0
                temp_c=v7_gpu_temp() if (step%args.v7_temp_every_updates==0) else None
                guard=gpu_guard_reason(reserved_mib=reserved_mib,free_mib=free_mib,temp_c=temp_c,
                                       max_reserved_mib=args.v7_max_reserved_mib,
                                       min_free_mib=args.v7_min_free_mib,max_temp_c=args.v7_max_temp_c)
                if guard:
                    recovery_path=v7_save_recovery('gpu_guard_'+guard)
                    print(json.dumps({'event':'v7_gpu_guard','reason':guard,'reserved_mib':reserved_mib,
                                      'free_mib':free_mib,'temp_c':temp_c,
                                      'recovery':str(recovery_path) if recovery_path else None}),flush=True)
                    raise RuntimeError('V7_GPU_GUARD:'+guard)
"""
    text = once(text, metric_anchor, metric_block, "V7 runtime GPU guard")
    return text


def patch_controller(text: str) -> str:
    text = once(
        text,
        "from recovery_identity import file_sha, validate_pair\n",
        "from recovery_identity import file_sha, validate_pair\nfrom v7_recovery import list_recovery_states,materialize_checkpoint\n",
        "controller V7 imports",
    )
    text = once(
        text,
        "STAGE_TOKENS = 5_000_000\nFULL_AUDIT_INTERVAL = 50_000_000\n",
        "STAGE_TOKENS = 5_000_000\nV7_RECOVERY_INTERVAL_TOKENS = 1_000_000\nFULL_AUDIT_INTERVAL = 50_000_000\n",
        "V7 recovery cadence",
    )
    text = once(
        text,
        'str(Path(__file__).parent / "train_v5_stage_launcher.py")',
        'str(Path(__file__).parent / "train_v7_stage_launcher.py")',
        "V7 stage launcher",
    )
    text = once(
        text,
        '               "--snapshot-interval-tokens", str(STAGE_TOKENS),\n               "--resume-save-interval-tokens", str(STAGE_TOKENS)]\n',
        '               "--snapshot-interval-tokens", "0",\n               "--resume-save-interval-tokens", "0",\n               "--v7-recovery-interval-tokens", str(V7_RECOVERY_INTERVAL_TOKENS)]\n',
        "compact recovery replaces full-model interval saves",
    )
    text = once(
        text,
        "    def train_stage(recipe: dict, tag: str, block_tokens: int) -> tuple[Path, dict]:\n",
        "    def train_stage(recipe: dict, tag: str, block_tokens: int) -> tuple[Path, list[Path], dict]:\n",
        "controller recovery return type",
    )
    text = once(
        text,
        "        return raw, perf\n\n    def make_interp",
        "        recovery_states=list_recovery_states(out)\n        return raw, recovery_states, perf\n\n    def make_interp",
        "controller recovery state discovery",
    )
    text = once(
        text,
        "        raw, perf = train_stage(recipe, tag, block)\n",
        "        raw, recovery_states, perf = train_stage(recipe, tag, block)\n",
        "controller train result",
    )

    reject_anchor = """        if accepted is None:
            # Only this 5M stage is discarded. The previous continuity anchor remains the
"""
    salvage = """        if accepted is None and recovery_states:
            # V7 probes compact exact states only after an endpoint rejection. Successful
            # stages pay no extra evaluator cost. Newest-first maximizes retained work.
            for recovery_no,recovery_state in enumerate(recovery_states,1):
                probe=scratch/f'{tag}_v7_recovery_{recovery_no:02d}.pt'
                probe_tokens=materialize_checkpoint(train_anchor,recovery_state,probe)
                print(json.dumps({'event':'v7_recovery_probe','path':str(recovery_state),
                                  'materialized':str(probe),'tokens':probe_tokens,
                                  'index':recovery_no,'count':len(recovery_states)}),flush=True)
                for idx,alpha in enumerate(dict.fromkeys(round(x,8) for x in alphas),1):
                    row=health_screen(probe,alpha,f'{tag}_v7r{recovery_no}_a{idx}',rotation,base_rows,base_val)
                    row['throughput']=perf;row['v7_recovery_salvage']=True
                    row['recovery_state']=str(recovery_state)
                    screened.append(row)
                    if row['healthy']:
                        accepted=row
                        break
                if accepted is not None:break

        if accepted is None:
            # No endpoint or intermediate state passed the unchanged health gate. Only in
"""
    text = once(text, reject_anchor, salvage, "V7 newest-healthy salvage")

    text = once(
        text,
        '            "max_software_rollback_tokens": STAGE_TOKENS,\n',
        '            "max_software_rollback_tokens": STAGE_TOKENS,\n            "v7_recovery_interval_tokens": V7_RECOVERY_INTERVAL_TOKENS,\n            "v7_blind_5m_rewind": False,\n',
        "V7 state metadata",
    )
    text = once(
        text,
        '        "max_software_rollback_tokens": STAGE_TOKENS,\n        "full_audit_interval_tokens": FULL_AUDIT_INTERVAL,\n',
        '        "max_software_rollback_tokens": STAGE_TOKENS,\n        "v7_recovery_interval_tokens": V7_RECOVERY_INTERVAL_TOKENS,\n        "v7_blind_5m_rewind": False,\n        "full_audit_interval_tokens": FULL_AUDIT_INTERVAL,\n',
        "V7 result metadata",
    )
    return text


def main() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    TRAINER.write_text(patch_trainer(trainer), encoding="utf-8")
    CONTROLLER.write_text(patch_controller(controller), encoding="utf-8")
    print("Scale V7 applied: 1M compact exact recovery + newest-healthy salvage + GPU guardrails")


if __name__ == "__main__":
    main()
