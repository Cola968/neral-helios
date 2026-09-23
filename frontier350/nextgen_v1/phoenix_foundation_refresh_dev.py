from __future__ import annotations

"""Development-only full-model foundation refresh for the current Neral 350M anchor.

Starts a NEW AdamW lineage from the exact 2.073B-token weight checkpoint because the
historical optimizer moments are unavailable. This is not an exact historical resume
and is never production eligible. It trains all 24 layers on a SHA-verified, current-
anchor BPE refresh corpus, writes complete optimizer/RNG/sampler state and uses the
rolling checkpoint journal so later stages can resume this NEW lineage exactly.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint_journal import CheckpointJournal, safe_save_every
from helios350_preview import CONTINUITY_SHA, TOKENIZER_SHA, trusted_asset
from phoenix_continuity_cpu_probe import EXPECTED_TOKENS, require_pinned_checkpoint
from source_lock350 import LOCK_PATH

SCHEMA = "phoenix_foundation_refresh_dev_v1"
CLASSIFICATION = "DEVELOPMENT_ONLY_FRESH_ADAMW_FOUNDATION_REFRESH"
PARAMETERS = 349_825_152
MAX_TARGET = 20_000_000
SHA64 = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(torch, state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError("stale checkpoint temporary file")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(state, tmp)
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def verify_refresh_manifest(root: Path, expected_sha: str, tokenizer_sha: str) -> dict:
    if not SHA64.fullmatch(expected_sha):
        raise ValueError("full refresh manifest SHA required")
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("regular refresh manifest required")
    if sha256_file(manifest_path) != expected_sha:
        raise ValueError("refresh manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != 1
        or manifest.get("architecture") != "frontier350-v1"
        or manifest.get("corpus_role") != "foundation-pretraining-refresh-development"
        or manifest.get("classification") != "DEVELOPMENT_ONLY_NOT_PRODUCTION_APPROVED"
        or manifest.get("continuation_from_tokens") != EXPECTED_TOKENS
        or manifest.get("continuation_from_checkpoint_sha256") != CONTINUITY_SHA
        or manifest.get("new_train_tokens") != MAX_TARGET
        or manifest.get("production_eligible") is not False
        or manifest.get("historical_2b_document_overlap_checked") is not False
        or manifest.get("train_validation_exact_document_disjoint") is not True
    ):
        raise ValueError("refresh manifest lineage/classification contract mismatch")
    if manifest.get("source_lock_sha256") != sha256_file(LOCK_PATH):
        raise ValueError("refresh corpus source lock differs from current repository lock")
    token = manifest.get("tokenizer", {})
    if token.get("vocab_size") != 16_384 or token.get("tokenizer_sha256") != tokenizer_sha:
        raise ValueError("refresh manifest tokenizer contract mismatch")

    def checked_files(role: str, subdir: str):
        rows = manifest.get(role)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing {role}")
        paths = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "tokens", "sha256"}:
                raise ValueError(f"invalid {role} row")
            name = row["path"]
            digest = row["sha256"]
            tokens = row["tokens"]
            if (
                not isinstance(name, str) or Path(name).name != name
                or not SHA64.fullmatch(str(digest))
                or type(tokens) is not int or tokens <= 0
                or name in seen
            ):
                raise ValueError(f"invalid {role} identity")
            path = root / subdir / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size != tokens * 2:
                raise ValueError(f"{role} shard size/path mismatch")
            if sha256_file(path) != digest:
                raise ValueError(f"{role} shard SHA mismatch")
            seen.add(name)
            paths.append(path)
        return paths

    train = checked_files("train_files", "train")
    valid = checked_files("valid_files", "valid")
    if {sha256_file(p) for p in train} & {sha256_file(p) for p in valid}:
        raise ValueError("identical train/validation shard bytes")
    return {"manifest": manifest, "train": train, "valid": valid, "root": root}


def lineage(manifest_sha: str, target: int, seq: int, micro: int, accum: int,
            lr: float, seed: int) -> dict:
    return {
        "kind": "phoenix_foundation_refresh_fresh_adamw_v1",
        "historical_source_sha256": CONTINUITY_SHA,
        "historical_processed_tokens": EXPECTED_TOKENS,
        "refresh_manifest_sha256": manifest_sha,
        "target_new_tokens": target,
        "seq": seq,
        "microbatch": micro,
        "grad_accum": accum,
        "peak_lr": lr,
        "schedule": "cosine",
        "warmup_frac": 0.01,
        "weight_decay": 0.1,
        "seed": seed,
        "gradient_checkpointing": True,
    }


def validate_resume(torch, state: dict, expected_lineage: dict, manifest_sha: str) -> None:
    if (
        not isinstance(state, dict)
        or state.get("checkpoint_schema") != SCHEMA
        or state.get("classification") != CLASSIFICATION
        or state.get("lineage") != expected_lineage
        or state.get("refresh_manifest_sha256") != manifest_sha
        or state.get("tokenizer_sha256") != TOKENIZER_SHA
        or state.get("config") is None
        or not isinstance(state.get("model"), dict)
        or not isinstance(state.get("optimizer"), dict)
        or not state["optimizer"].get("state")
        or type(state.get("processed_tokens")) is not int
        or type(state.get("step")) is not int
        or not all(k in state for k in (
            "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "train_sampler_rng"
        ))
        or state.get("production_eligible") is not False
    ):
        raise ValueError("refresh resume checkpoint is incomplete or wrong lineage")
    if not all(torch.is_tensor(v) for v in state["model"].values()):
        raise ValueError("refresh resume contains non-Torch model values")
    processed = state["processed_tokens"]
    step = state["step"]
    update = expected_lineage["seq"] * expected_lineage["microbatch"] * expected_lineage["grad_accum"]
    if processed <= 0 or processed > expected_lineage["target_new_tokens"] or processed % update:
        raise ValueError("resume processed-token boundary is invalid")
    if step != processed // update:
        raise ValueError("resume optimizer-step/token counters disagree")
    if not math.isfinite(float(state.get("best_validation", float("nan")))):
        raise ValueError("resume best validation is nonfinite")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--corpus-root", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target-new-tokens", type=int, default=MAX_TARGET)
    p.add_argument("--stop-after-new-tokens", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256", default="")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=35023)
    args = p.parse_args(argv)

    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit opt-in to the owner's checkpoint is required")
    if args.target_new_tokens != MAX_TARGET:
        p.error(f"this development lineage is locked to {MAX_TARGET} new target tokens")
    update = args.seq * args.microbatch * args.grad_accum
    if (args.seq != 256 or args.microbatch != 1 or args.grad_accum != 4
            or args.lr != 1e-5 or update != 1024):
        p.error("development run dimensions/LR are intentionally locked")
    if not 0 < args.stop_after_new_tokens <= args.target_new_tokens:
        p.error("invalid stop")
    if args.stop_after_new_tokens < args.target_new_tokens and args.stop_after_new_tokens % update:
        p.error("intermediate stop must be an exact optimizer boundary")

    source = require_pinned_checkpoint(args.source_checkpoint)
    assets = args.assets_root.resolve(strict=True)
    tokenizer_path = trusted_asset(
        assets, assets / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA
    )
    refresh = verify_refresh_manifest(args.corpus_root, args.manifest_sha256, TOKENIZER_SHA)
    out = args.out.absolute()
    expected_lineage = lineage(
        args.manifest_sha256, args.target_new_tokens, args.seq,
        args.microbatch, args.grad_accum, args.lr, args.seed
    )

    fresh = args.resume is None
    if fresh:
        if args.resume_sha256:
            p.error("resume SHA without resume file")
        if out.exists():
            p.error("fresh refresh output must not exist")
    else:
        if not SHA64.fullmatch(args.resume_sha256):
            p.error("exact resume SHA required")
        if not out.is_dir() or args.resume.resolve() != (out / "resume.pt").resolve():
            p.error("resume must be output/resume.pt")
        if not args.resume.is_file() or args.resume.is_symlink():
            p.error("regular refresh resume required")
        if sha256_file(args.resume) != args.resume_sha256:
            p.error("refresh resume SHA mismatch")

    import phoenix_cuda_diagnostic as gpu_guard
    gpu_guard.require_stably_idle_gpu()

    import numpy as np
    import torch
    from model350 import FRONTIER_350, Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from train350 import checkpoint_state, cosine_lr, mask_budget_targets, update_token_counts
    from train350_compiled import PersistentMMapPool, evaluate

    gpu_guard.require_cuda_ready(torch)
    if FRONTIER_350.n_layers != 24 or FRONTIER_350.vocab_size != 16_384:
        raise RuntimeError("Neral architecture contract changed")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()

    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(True)
    if model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("locked 350M parameter count changed")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.1, fused=True
    )
    rng = random.Random(args.seed + 1)
    step = processed = 0
    best = float("inf")

    if fresh:
        base = _checked_historical_weights(source, torch, Neral350M, FRONTIER_350.to_dict())
        model.load_state_dict(base["model"], strict=True)
        del base
        if optimizer.state_dict()["state"]:
            raise RuntimeError("fresh refresh optimizer unexpectedly has inherited moments")
        out.mkdir(parents=True, exist_ok=False)
    else:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(torch, state, expected_lineage, args.manifest_sha256)
        if state.get("config") != FRONTIER_350.to_dict():
            raise ValueError("resume architecture mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        processed = state["processed_tokens"]
        best = float(state["best_validation"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        rng.setstate(state["train_sampler_rng"])
        del state
        if processed >= args.stop_after_new_tokens:
            raise ValueError("resume is already at or beyond requested stop")

    train = PersistentMMapPool(refresh["root"] / "train", FRONTIER_350.vocab_size)
    valid = PersistentMMapPool(refresh["root"] / "valid", FRONTIER_350.vocab_size)
    total_updates = math.ceil(args.target_new_tokens / update)
    warmup = max(1, int(total_updates * 0.01))
    validation_seed = args.seed + 999

    # no_grad (not inference_mode) keeps RoPE cache reusable by later backward.
    baseline_val = evaluate(
        model, valid, seq=args.seq, batch=args.microbatch, batches=8,
        device=device, seed=validation_seed
    )
    best = min(best, baseline_val)

    metrics = out / "metrics.jsonl"
    if fresh:
        with metrics.open("x", encoding="utf-8") as log:
            log.write(json.dumps({
                "event": "refresh_start",
                "classification": CLASSIFICATION,
                "historical_processed_tokens": EXPECTED_TOKENS,
                "new_processed_tokens": 0,
                "target_new_tokens": args.target_new_tokens,
                "stop_after_new_tokens": args.stop_after_new_tokens,
                "baseline_validation_loss": baseline_val,
                "lineage": expected_lineage,
                "production_eligible": False,
            }, sort_keys=True) + "\n")

    def append(row: dict) -> None:
        with metrics.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()

    def state_dict() -> dict:
        state = checkpoint_state(
            model, optimizer, step, processed, best, args.target_new_tokens,
            TOKENIZER_SHA, train_sampler_rng=rng,
            validation_identity=args.manifest_sha256
        )
        state.update({
            "checkpoint_schema": SCHEMA,
            "classification": CLASSIFICATION,
            "historical_processed_tokens": EXPECTED_TOKENS,
            "refresh_manifest_sha256": args.manifest_sha256,
            "lineage": expected_lineage,
            "production_eligible": False,
            "historical_overlap_checked": False,
        })
        return state

    journal = None
    if not fresh:
        journal = CheckpointJournal(
            out / "resume.pt", initial_sha256=args.resume_sha256,
            initial_tokens=processed, writer=lambda state, path: atomic_save(torch, state, path),
            max_rollback_tokens=262_144, retain=3
        )

    save_every = safe_save_every(
        seq=args.seq, microbatch=args.microbatch, accumulation=args.grad_accum,
        max_rollback_tokens=262_144, requested=256
    )
    eval_every = 256
    log_every = 32
    invocation_start = processed
    invocation_wall = time.monotonic()
    last_log_time = invocation_wall
    last_log_tokens = processed
    model.train()

    while processed < args.stop_after_new_tokens:
        lr = cosine_lr(step, warmup, total_updates, args.lr, min_ratio=0.10)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        counts = update_token_counts(
            args.stop_after_new_tokens - processed,
            args.seq * args.microbatch, args.grad_accum
        )
        this_update = sum(counts)
        weighted = torch.zeros((), device=device, dtype=torch.float32)
        for count in counts:
            x, y = train.batch(args.microbatch, args.seq, device, rng)
            y = mask_budget_targets(y, count)
            _, loss = model(x, y)
            if loss is None or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite refresh training loss")
            (loss * (count / this_update)).backward()
            weighted.add_(loss.detach().float() * (count / this_update))
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if not math.isfinite(float(grad)):
            raise FloatingPointError("nonfinite refresh gradient")
        optimizer.step()
        processed += this_update
        step += 1

        if step % 32 == 0 or processed >= args.stop_after_new_tokens:
            torch.cuda.synchronize()
            gpu_guard.require_safe_runtime_temperature()

        if step % log_every == 0 or processed >= args.stop_after_new_tokens:
            now = time.monotonic()
            append({
                "event": "refresh_train",
                "step": step,
                "new_processed_tokens": processed,
                "cumulative_reference_tokens": EXPECTED_TOKENS + processed,
                "loss": float(weighted),
                "lr": lr,
                "grad_norm": float(grad),
                "tokens_per_second_recent": (processed - last_log_tokens) / max(1e-9, now - last_log_time),
                "tokens_per_second_invocation": (processed - invocation_start) / max(1e-9, now - invocation_wall),
                "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 1),
            })
            last_log_time, last_log_tokens = now, processed

        if step % eval_every == 0 or processed >= args.stop_after_new_tokens:
            val = evaluate(
                model, valid, seq=args.seq, batch=args.microbatch, batches=8,
                device=device, seed=validation_seed
            )
            best = min(best, val)
            append({
                "event": "refresh_validation",
                "step": step,
                "new_processed_tokens": processed,
                "validation_loss": val,
                "best_validation_loss": best,
            })

        if step % save_every == 0 or processed >= args.stop_after_new_tokens:
            state = state_dict()
            resume_path = out / "resume.pt"
            if journal is None:
                atomic_save(torch, state, resume_path)
                digest = sha256_file(resume_path)
                journal = CheckpointJournal(
                    resume_path, initial_sha256=digest, initial_tokens=processed,
                    writer=lambda state, path: atomic_save(torch, state, path),
                    max_rollback_tokens=262_144, retain=3
                )
            else:
                journal.commit(state, resume_path)

    if journal is None:
        raise RuntimeError("refresh run ended without a full resumable checkpoint")
    receipt = journal.verify()
    source_unchanged = sha256_file(source) == CONTINUITY_SHA
    summary = {
        "event": "phoenix_foundation_refresh_dev_finished",
        "classification": CLASSIFICATION,
        "parameters": PARAMETERS,
        "layers_trained": FRONTIER_350.n_layers,
        "historical_processed_tokens": EXPECTED_TOKENS,
        "new_processed_tokens": processed,
        "cumulative_reference_tokens": EXPECTED_TOKENS + processed,
        "target_new_tokens": args.target_new_tokens,
        "requested_stop_new_tokens": args.stop_after_new_tokens,
        "optimizer_steps": step,
        "baseline_validation_loss_for_invocation": baseline_val,
        "best_validation_loss": best,
        "resume_sha256": receipt["sha256"],
        "checkpoint_kind": receipt["kind"],
        "refresh_manifest_sha256": args.manifest_sha256,
        "source_unchanged": source_unchanged,
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "training_wall_seconds": time.monotonic() - invocation_wall,
        "historical_overlap_checked": False,
        "quality_gain_verified": False,
        "production_eligible": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not source_unchanged:
        raise RuntimeError("historical anchor checkpoint changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
