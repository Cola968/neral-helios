"""Bounded auto-screened OASST2 SFT diagnostic for the OWN Neral 350M weights.

Research fast lane only. This deliberately does NOT claim human review, independent
holdout clearance, production eligibility, or final quality. It uses the exact
SHA-pinned public OASST2 archive, reviewed-human DE/EN pairs, strict automatic
PII/URL/role filtering, tree-disjoint train/validation, response-only EOS loss,
and a separate create-only local checkpoint.

Purpose: answer one question quickly and honestly — does supervised chat-format
training move the real 350M checkpoint in the right direction at all?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from audit_sft_stage import EMAIL, PHONE_LIKE, IPV4, URL, ROLE
from helios350_preview import TOKENIZER_SHA, trusted_asset
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_oasst2_bridge import load_archive, local_input, select_tree_disjoint
from fetch_oasst2_sft50 import UPSTREAM_REVISION, UPSTREAM_SHA256, choose_pairs
from sft_contract import encode_row
from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices
from stage_oasst1_sft import frozen_chat_hashes

PARAMETERS = 349_825_152
DEFAULT_STEPS = 50
MAX_STEPS = 100
MAX_RESERVED = 9 * 1024**3
SOURCE_CLASS = "auto_screened_public_oasst2_diagnostic_not_human_approved"
CHECKPOINT_SCHEMA = "phoenix_auto_sft_diagnostic_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rejected(text: str) -> str | None:
    if EMAIL.search(text) or PHONE_LIKE.search(text) or IPV4.search(text):
        return "possible_identifier"
    if URL.search(text):
        return "url"
    if ROLE.search(text):
        return "embedded_role"
    return None


def auto_prepare(repo_root: Path, archive: Path, private_root: Path, tokenizer_path: Path):
    benchmark = repo_root / "benchmark_nis2_chat50.py"
    frozen = frozen_chat_hashes(benchmark)
    messages = load_archive(local_input(archive, private_root))
    pairs = choose_pairs(messages, max_en=800, max_de=400)
    raw_splits, source_summary = select_tree_disjoint(pairs, messages, frozen)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    if tok.get_vocab_size() != 16_384:
        raise ValueError("locked tokenizer vocabulary changed")
    bos = tok.token_to_id("<|bos|>")
    eos = tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos == eos:
        raise ValueError("invalid BOS/EOS")

    packed = {"train": [], "validation": []}
    rejects: dict[str, int] = {}
    for split, rows in raw_splits.items():
        for row in rows:
            reason = _rejected(row["user"]) or _rejected(row["assistant"])
            if reason is not None:
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            try:
                encoded = encode_row(row, tok, int(bos), int(eos), 16_384, 512)
            except ValueError:
                rejects["sequence_or_token_contract"] = rejects.get("sequence_or_token_contract", 0) + 1
                continue
            packed[split].append(encoded)

    if len(packed["train"]) < 500 or len(packed["validation"]) < 50:
        raise RuntimeError("auto-screening left too little DE/EN SFT data")
    train_ids = {row["id"] for row in packed["train"]}
    valid_ids = {row["id"] for row in packed["validation"]}
    if train_ids & valid_ids:
        raise RuntimeError("train/validation identity leakage")
    report = {
        "classification": SOURCE_CLASS,
        "source_revision": UPSTREAM_REVISION,
        "source_archive_sha256": UPSTREAM_SHA256,
        "nis2_exact_prompt_hashes_excluded": len(frozen),
        "selector": source_summary,
        "packed_rows": {k: len(v) for k, v in packed.items()},
        "auto_rejected_rows": sum(rejects.values()),
        "auto_reject_reasons": rejects,
        "human_review_complete": False,
        "independent_holdout_checked": False,
        "semantic_overlap_checked": False,
        "production_eligible": False,
    }
    return tok, int(bos), int(eos), packed, report


def _cpu_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(torch, v) for v in value)
    return value


def atomic_save(torch, state: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError("refusing checkpoint overwrite")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError("stale temporary checkpoint exists")
    try:
        torch.save(state, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--batch", type=int, choices=(1, 2, 4), default=4)
    p.add_argument("--train-blocks", type=int, choices=(4, 8), default=4)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit execution and trust in the owner's checkpoint are required")
    if not 1 <= args.steps <= MAX_STEPS:
        p.error(f"steps must be 1..{MAX_STEPS}")
    if args.out.exists() or args.out.is_symlink() or not args.out.is_absolute():
        p.error("output must be a new absolute directory")

    source = require_pinned_checkpoint(args.checkpoint)
    assets = args.assets_root.resolve(strict=True)
    private = args.private_root.resolve(strict=True)
    tokenizer_path = trusted_asset(
        assets, assets / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA
    )
    repo_root = Path(__file__).resolve().parents[2]
    tok, bos, eos, splits, data_report = auto_prepare(
        repo_root, args.archive, private, tokenizer_path
    )

    # Stable idle telemetry happens before Torch/CUDA model allocation.
    import phoenix_cuda_diagnostic as gpu_guard
    gpu_guard.require_stably_idle_gpu()

    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from protection_v9 import configure_trainable

    gpu_guard.require_cuda_ready(torch)
    base = _checked_historical_weights(source, torch, Neral350M, FRONTIER_350.to_dict())
    device = torch.device("cuda")
    torch.manual_seed(20260923)
    torch.cuda.manual_seed_all(20260923)
    random.seed(20260923)
    np.random.seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()

    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(base["model"], strict=True)
    if model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("Neral architecture changed")
    historical_tokens = int(base.get("processed_tokens", -1))
    del base

    named = configure_trainable(model, args.train_blocks)
    params = [p for _, p in named]
    if not params:
        raise RuntimeError("no trainable parameters")
    trainable_parameters = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(
        params, lr=2e-5, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.01, fused=True
    )
    rng = random.Random(20260923)
    seq = 512
    val_indices = validation_indices(len(splits["validation"]))

    def batch(rows: list[dict], picks: list[int]):
        selected = [rows[i] for i in picks]
        width = dynamic_batch_width(selected, seq)
        x = torch.full((len(selected), width), eos, dtype=torch.long)
        y = torch.full((len(selected), width), -100, dtype=torch.long)
        for i, row in enumerate(selected):
            n = len(row["input_ids"])
            x[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
            y[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
        return x.to(device), y.to(device)

    def validation_loss() -> tuple[float, int]:
        model.eval()
        values = []
        supervised = 0
        # no_grad, NOT inference_mode: RoPE cache may be reused by subsequent backward.
        with torch.no_grad():
            for index in val_indices:
                x, y = batch(splits["validation"], [index])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                if loss is None or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite SFT validation")
                count = int((y != -100).sum().item())
                values.append((float(loss), count))
                supervised += count
        model.train()
        return token_weighted_loss(values), supervised

    before, validation_tokens = validation_loss()
    if torch.cuda.memory_reserved() > MAX_RESERVED:
        raise RuntimeError("baseline exceeds experimental VRAM ceiling")

    stats = []
    supervised_total = 0
    started = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        gpu_guard.require_safe_runtime_temperature()
        picks = [rng.randrange(len(splits["train"])) for _ in range(args.batch)]
        x, y = batch(splits["train"], picks)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        if loss is None or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite SFT train loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not math.isfinite(float(grad)):
            raise FloatingPointError("nonfinite SFT gradient")
        optimizer.step()
        torch.cuda.synchronize()
        gpu_guard.require_safe_runtime_temperature()
        if torch.cuda.memory_reserved() > MAX_RESERVED:
            raise RuntimeError("SFT diagnostic exceeds 9 GiB reserved VRAM")
        supervised = int((y != -100).sum().item())
        supervised_total += supervised
        row = {
            "step": step,
            "train_loss": float(loss),
            "grad_norm": float(grad),
            "supervised_answer_tokens": supervised,
            "cumulative_supervised_answer_tokens": supervised_total,
        }
        stats.append(row)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps({"event": "auto_sft_step", **row}, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    after, validation_tokens_after = validation_loss()
    if validation_tokens_after != validation_tokens:
        raise RuntimeError("validation token denominator changed")
    improved = after < before

    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.out / "candidate_auto_sft.pt"
    candidate_sha = None
    if improved:
        state = {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "classification": SOURCE_CLASS,
            "model": {name: tensor.detach().to("cpu", dtype=torch.bfloat16)
                      for name, tensor in model.state_dict().items()},
            "config": FRONTIER_350.to_dict(),
            "tokenizer_sha256": TOKENIZER_SHA,
            "base_checkpoint_sha256": sha256_file(source),
            "source_archive_sha256": UPSTREAM_SHA256,
            "source_revision": UPSTREAM_REVISION,
            "historical_pretraining_tokens": historical_tokens,
            "sft_supervised_tokens": supervised_total,
            "sft_steps": args.steps,
            "train_blocks": args.train_blocks,
            "trainable_parameters": trainable_parameters,
            "optimizer": _cpu_tree(torch, optimizer.state_dict()),
            "sampler_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [v.cpu() for v in torch.cuda.get_rng_state_all()],
            "data_report": data_report,
            "production_eligible": False,
            "human_review_complete": False,
            "independent_holdout_checked": False,
        }
        atomic_save(torch, state, checkpoint)
        candidate_sha = sha256_file(checkpoint)

    source_unchanged = sha256_file(source) == "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
    report = {
        "event": "phoenix_auto_sft_diagnostic_finished",
        "classification": SOURCE_CLASS,
        "parameters": PARAMETERS,
        "trainable_parameters": trainable_parameters,
        "train_blocks": args.train_blocks,
        "steps": args.steps,
        "batch": args.batch,
        "packed_rows": data_report["packed_rows"],
        "auto_rejected_rows": data_report["auto_rejected_rows"],
        "validation_rows_evaluated": len(val_indices),
        "validation_supervised_tokens": validation_tokens,
        "validation_loss_before": before,
        "validation_loss_after": after,
        "validation_loss_improved": improved,
        "supervised_answer_tokens_trained": supervised_total,
        "training_wall_seconds": elapsed,
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "candidate_checkpoint_written": bool(candidate_sha),
        "candidate_sha256": candidate_sha,
        "historical_source_unchanged": source_unchanged,
        "human_review_complete": False,
        "independent_holdout_checked": False,
        "quality_gain_verified": False,
        "production_eligible": False,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.out / "metrics.jsonl").open("x", encoding="utf-8") as f:
        for row in stats:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not source_unchanged:
        raise RuntimeError("historical checkpoint changed")
    if not improved:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
