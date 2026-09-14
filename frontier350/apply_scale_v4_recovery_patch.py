from __future__ import annotations

"""Apply the Scale V4 recovery/speed patch inside a checked-out workflow tree.

This is intentionally source-only and idempotent. It does not change promotion gates,
recipes, data, replay weighting, model math, or the active workflow run that is already
checked out on an older commit.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"
CONTROLLER = ROOT / "run_scale_v4.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"patch anchor missing: {label}")
    if text.count(old) != 1:
        raise RuntimeError(f"patch anchor not unique: {label}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, repl: str, label: str) -> str:
    out, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"regex patch failed ({count}): {label}")
    return out


def patch_trainer(text: str) -> str:
    if "--snapshot-interval-tokens" in text and "recovery_checkpoints" in text:
        return text

    text = replace_once(
        text,
        "    ap.add_argument('--snapshot-fraction',type=float,default=0.0,help='Optional one-shot in-process snapshot fraction in (0,1), written as midpoint.pt')\n",
        "    ap.add_argument('--snapshot-fraction',type=float,default=0.0,help='Optional legacy one-shot recovery snapshot fraction in (0,1), written as midpoint.pt')\n"
        "    ap.add_argument('--snapshot-interval-tokens',type=int,default=0,help='Optional lightweight recovery snapshot interval; endpoint is excluded')\n"
        "    ap.add_argument('--resume-save-interval-tokens',type=int,default=0,help='Optional full resume-state save interval; 0 preserves legacy 20-update cadence')\n",
        "trainer args",
    )
    text = replace_once(
        text,
        "    if args.snapshot_fraction and not (0.0 < args.snapshot_fraction < 1.0):raise ValueError('snapshot fraction must be 0 or in (0,1)')\n"
        "    if args.fast_replay_only and args.mode!='replay_only':raise ValueError('fast replay path is valid only for replay_only mode')\n"
        "    tokens_per_update=seq*batch*micros\n",
        "    if args.snapshot_fraction and not (0.0 < args.snapshot_fraction < 1.0):raise ValueError('snapshot fraction must be 0 or in (0,1)')\n"
        "    if args.snapshot_interval_tokens<0 or args.resume_save_interval_tokens<0:raise ValueError('checkpoint intervals must be >=0')\n"
        "    if args.fast_replay_only and args.mode!='replay_only':raise ValueError('fast replay path is valid only for replay_only mode')\n"
        "    tokens_per_update=seq*batch*micros\n"
        "    if args.snapshot_interval_tokens and args.snapshot_interval_tokens<tokens_per_update:raise ValueError('snapshot interval must be >= one update')\n"
        "    if args.resume_save_interval_tokens and args.resume_save_interval_tokens<tokens_per_update:raise ValueError('resume save interval must be >= one update')\n",
        "trainer validation",
    )
    text = replace_once(
        text,
        "              'snapshot_fraction':args.snapshot_fraction,'trainable_names':names,'data_sha256':hashes,\n",
        "              'snapshot_fraction':args.snapshot_fraction,'snapshot_interval_tokens':args.snapshot_interval_tokens,\n"
        "              'resume_save_interval_tokens':args.resume_save_interval_tokens,'trainable_names':names,'data_sha256':hashes,\n",
        "trainer contract",
    )
    text = replace_once(
        text,
        "    snapshot_target=math.ceil(args.new_tokens*args.snapshot_fraction) if args.snapshot_fraction else 0\n"
        "    snapshot_path=out/'midpoint.pt'\n",
        "    snapshot_target=math.ceil(args.new_tokens*args.snapshot_fraction) if args.snapshot_fraction else 0\n"
        "    snapshot_path=out/'midpoint.pt'\n"
        "    recovery_targets=[]\n"
        "    if args.snapshot_interval_tokens:\n"
        "        target=args.snapshot_interval_tokens\n"
        "        while target<args.new_tokens:\n"
        "            recovery_targets.append(target);target+=args.snapshot_interval_tokens\n"
        "    recovery_index=0\n"
        "    while recovery_index<len(recovery_targets) and recovery_targets[recovery_index]<=done:recovery_index+=1\n"
        "    recovery_records=[]\n"
        "    next_resume_save=(math.floor(done/args.resume_save_interval_tokens)+1)*args.resume_save_interval_tokens if args.resume_save_interval_tokens else 0\n",
        "trainer targets",
    )

    save_replacement = '''    def save(path=None,include_training_state=True):
        state={k:v.detach().to(device='cpu',dtype=torch.bfloat16) for k,v in model.state_dict().items()}
        if any(not torch.equal(state[k],source['model'][k]) for k in frozen_names):
            raise RuntimeError('frozen weights changed')
        ck=dict(source);ck.pop('v9_training_state',None)
        ck.update(model=state,processed_tokens=int(source['processed_tokens'])+done,
                  step=int(source['step'])+step,best_validation=float('inf'),validation_identity='')
        ck['optimizer']={**source['optimizer'],'state':{}}
        if include_training_state:
            ck['v9_training_state']={'contract':contract,'masters':{n:p.detach().cpu() for n,p in named},
                'optimizer':optimizer.state_dict(),'sampler_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'new_tokens':done,'updates':step,'batch_chain':chain,'source_tokens':counts}
        ck['v9_provenance']={'anchor_sha256':args.anchor_sha,'mode':args.mode,'frozen_weights_unchanged':True,
                             'data_sha256':hashes,'new_training_tokens':done,'batch_chain':chain,
                             'lr_start':args.lr,'lr_end':args.lr_end,'replay_boost':args.replay_boost,
                             'recovery_only':not include_training_state}
        atomic_save(ck,path or out/'resume.pt')

    if not args.resume:save()'''
    text = sub_once(
        text,
        r"    def save\(path=None\):.*?\n\n    if not args\.resume:save\(\)",
        save_replacement,
        "trainer save function",
    )

    loop_replacement = '''        if step%5==0 or done==args.new_tokens:print(json.dumps(row),flush=True)
        if snapshot_target and not snapshot_path.exists() and done>=snapshot_target:
            save(snapshot_path,False);print(json.dumps({'event':'snapshot','kind':'legacy_midpoint','path':str(snapshot_path),'new_tokens':done,'processed_tokens':int(source['processed_tokens'])+done,'sha256':file_sha(snapshot_path)}),flush=True)
        while recovery_index<len(recovery_targets) and done>=recovery_targets[recovery_index]:
            target=recovery_targets[recovery_index];path=out/f'recovery_{target:012d}.pt'
            save(path,False)
            record={'path':str(path),'target_new_tokens':target,'new_tokens':done,
                    'processed_tokens':int(source['processed_tokens'])+done,'sha256':file_sha(path)}
            recovery_records.append(record);print(json.dumps({'event':'snapshot','kind':'recovery_interval',**record}),flush=True)
            recovery_index+=1
        if args.resume_save_interval_tokens:
            if done==args.new_tokens or done>=next_resume_save:
                save()
                while next_resume_save and next_resume_save<=done:next_resume_save+=args.resume_save_interval_tokens
        elif step%20==0 or done==args.new_tokens:
            save()
        if args.max_updates and step-start_step>=args.max_updates:save();break'''
    text = sub_once(
        text,
        r"        if step%5==0 or done==args\.new_tokens:print\(json\.dumps\(row\),flush=True\).*?        if args\.max_updates and step-start_step>=args\.max_updates:save\(\);break",
        loop_replacement,
        "trainer checkpoint loop",
    )
    text = replace_once(
        text,
        "             'midpoint_sha256':file_sha(snapshot_path) if snapshot_path.exists() else None,\n"
        "             'checkpoint_sha256':file_sha(out/'resume.pt'),'frozen_weights_unchanged':True,'promoted':False}\n",
        "             'midpoint_sha256':file_sha(snapshot_path) if snapshot_path.exists() else None,\n"
        "             'recovery_checkpoint_interval_tokens':args.snapshot_interval_tokens,\n"
        "             'recovery_checkpoints':recovery_records,\n"
        "             'resume_save_interval_tokens':args.resume_save_interval_tokens,\n"
        "             'checkpoint_sha256':file_sha(out/'resume.pt'),'frozen_weights_unchanged':True,'promoted':False}\n",
        "trainer summary",
    )
    return text


def patch_controller(text: str) -> str:
    if "RECOVERY_INTERVAL_TOKENS = 50_000_000" in text and "recovery_points" in text:
        return text

    text = replace_once(
        text,
        'MIN_REPLAY_TAIL = 3 * SEQ * BATCH\nSTATE_FILE = "frontier350_scale_v4_state.json"\n',
        'MIN_REPLAY_TAIL = 3 * SEQ * BATCH\nRECOVERY_INTERVAL_TOKENS = 50_000_000\nRESUME_SAVE_INTERVAL_TOKENS = 10_000_000\nSTATE_FILE = "frontier350_scale_v4_state.json"\n',
        "controller constants",
    )

    train_replacement = '''    def train(block_tokens: int, recipe: dict, tag: str) -> tuple[Path, list[Path], dict]:
        out = scratch / tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(train_anchor), "--anchor-sha", train_sha,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", recipe["mode"], "--new-tokens", str(block_tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",
               "--lr", str(recipe["lr"]), "--lr-end", str(recipe["lr_end"]),
               "--replay-boost", str(recipe["replay_boost"]),
               "--snapshot-interval-tokens", str(RECOVERY_INTERVAL_TOKENS),
               "--resume-save-interval-tokens", str(RESUME_SAVE_INTERVAL_TOKENS)]
        if recipe["mode"] == "replay_only":
            cmd.append("--fast-replay-only")
        print(json.dumps({"event": "train_start", "tag": tag, "from_tokens": train_tokens,
                          "block_tokens": block_tokens, "pack": PACK, "recipe": recipe,
                          "recovery_interval_tokens": RECOVERY_INTERVAL_TOKENS,
                          "resume_save_interval_tokens": RESUME_SAVE_INTERVAL_TOKENS}), flush=True)
        subprocess.run(cmd, check=True)
        summary = read_json(out / "summary.json")
        if not summary.get("complete") or not summary.get("frozen_weights_unchanged"):
            raise RuntimeError("training contract failed")
        rates = [float(r["tokens_per_second"]) for r in train_rows(out / "metrics.jsonl")[5:]
                 if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
        perf = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates),
                "p10_tps": p10(rates), "peak_tps": max(rates)} if rates else {}
        recovery_points = []
        for item in summary.get("recovery_checkpoints", []):
            p = Path(item["path"])
            expected_tokens = int(item["processed_tokens"])
            expected_sha = item["sha256"]
            if not valid_checkpoint(p, expected_sha, expected_tokens):
                raise RuntimeError("invalid recovery checkpoint identity")
            recovery_points.append(p)
        # Backward-compatible fallback if a trainer only produced the legacy midpoint.
        mid = out / "midpoint.pt"
        if not recovery_points and mid.is_file():
            recovery_points.append(mid)
        recovery_points.sort(key=ck_tokens, reverse=True)
        return out / "resume.pt", recovery_points, perf

    def make_interp'''
    text = sub_once(
        text,
        r"    def train\(block_tokens: int, recipe: dict, tag: str\).*?\n    def make_interp",
        train_replacement,
        "controller train function",
    )

    recovery_replacement = '''            raw, recovery_points, perf = train(block, recipe, tag)
            winner, shadow_try, tries = search_checkpoint(raw, recipe, tag + "_end", perf,
                                                           rotation, base_smoke_rows, base_smoke_val)
            attempts_all.extend(tries)
            if winner is not None:
                selected = winner
                conservative = max(0, ri - 1)
                break
            if shadow_try is not None:
                shadow_candidate = shadow_try
                # If the shadow used the one-case bilingual tolerance, force retention
                # next stage; otherwise return toward the faster balanced recipe.
                conservative = 1 if shadow_try.get("shadow_tolerance_only") else max(0, ri - 1)
                break
            # Endpoint hard-failed. Walk recovery checkpoints newest -> oldest. They are
            # only evaluated now, so normal successful stages do not pay benchmark cost.
            for recovery_no, recovery_point in enumerate(recovery_points, 1):
                recovery_tokens = ck_tokens(recovery_point)
                print(json.dumps({"event": "recovery_probe", "tag": tag,
                                  "checkpoint": str(recovery_point), "tokens": recovery_tokens,
                                  "index": recovery_no, "count": len(recovery_points)}), flush=True)
                winner, shadow_try, tries = search_checkpoint(
                    recovery_point, recipe, f"{tag}_recovery_{recovery_tokens}", perf,
                    rotation, base_smoke_rows, base_smoke_val)
                attempts_all.extend(tries)
                if winner is not None:
                    selected = winner
                    conservative = max(0, ri - 1)
                    break
                if shadow_try is not None:
                    shadow_candidate = shadow_try
                    conservative = 1 if shadow_try.get("shadow_tolerance_only") else max(0, ri - 1)
                    break
            if selected is not None or shadow_candidate is not None:
                break
'''
    text = sub_once(
        text,
        r"            raw, midpoint, perf = train\(block, recipe, tag\).*?(?=        if selected is not None:)",
        recovery_replacement,
        "controller recovery loop",
    )
    text = replace_once(
        text,
        '        "block_ladder": list(LADDER),\n        "pack": PACK,\n',
        '        "block_ladder": list(LADDER),\n        "recovery_interval_tokens": RECOVERY_INTERVAL_TOKENS,\n        "resume_save_interval_tokens": RESUME_SAVE_INTERVAL_TOKENS,\n        "recovery_newest_first_on_endpoint_failure": True,\n        "pack": PACK,\n',
        "controller result metadata",
    )
    return text


def main() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    patched_trainer = patch_trainer(trainer)
    patched_controller = patch_controller(controller)
    TRAINER.write_text(patched_trainer, encoding="utf-8")
    CONTROLLER.write_text(patched_controller, encoding="utf-8")
    print("Scale V4 recovery patch applied: 50M lightweight recovery checkpoints; 10M full resume saves")


if __name__ == "__main__":
    main()
