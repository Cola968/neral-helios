"""Development-only OASST2 SFT sweep on a trusted Phoenix foundation-v2 checkpoint.

This is a diagnostic, not a reviewed SFT release path. It uses the pinned human-only
auto-screened OASST2 preparation, preserves FP32 master parameters/AdamW moments, and
compares each candidate against the exact supplied foundation checkpoint on public
development suites. No independent holdout or production promotion is claimed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

from helios350_preview import TOKENIZER_SHA, trusted_asset
from phoenix_auto_sft_diagnostic import auto_prepare, _cpu_tree, atomic_save, sha256_file
from phoenix_foundation_refresh_public_dev_eval import (
    load_candidate as load_foundation_candidate,
    run_model,
    knowledge_choice_eval,
)
from phoenix_master_precision import require_fp32_trainable_subset
from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices
from train_sft_pilot import base_token_accounting

PARAMETERS = 349_825_152
SCHEMA = "phoenix_auto_oasst2_on_refresh_v1_fp32_master"
CLASSIFICATION = "AUTO_SCREENED_OASST2_ON_FOUNDATION_V2_PUBLIC_DEV_ONLY"
MAX_RESERVED = 9 * 1024**3
CONFIGS = (
    {"blocks": 4, "lr": 1.0e-5, "steps": 40},
    {"blocks": 8, "lr": 5.0e-6, "steps": 50},
    {"blocks": 8, "lr": 5.0e-6, "steps": 100},
)
PAIR_CATEGORIES = (
    "instruction_following", "dialogue_state", "truthfulness",
    "robustness", "compositional_reasoning",
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--foundation-checkpoint", type=Path, required=True)
    p.add_argument("--foundation-sha256", required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit trust in the owner's SHA-pinned foundation checkpoint required")
    if args.out.exists() or args.out.is_symlink() or not args.out.is_absolute():
        p.error("output must be a new absolute directory")
    if len(args.foundation_sha256) != 64:
        p.error("exact foundation SHA-256 required")

    assets = args.assets_root.resolve(strict=True)
    private = args.private_root.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    tokenizer_path = trusted_asset(
        assets, assets / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA
    )
    repo_root = Path(__file__).resolve().parents[2]
    tokenizer, bos, eos, splits, data_report = auto_prepare(
        repo_root, archive, private, tokenizer_path
    )

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    from protection_v9 import configure_trainable

    guard.require_cuda_ready(torch)
    foundation = load_foundation_candidate(
        torch, args.foundation_checkpoint, args.foundation_sha256, FRONTIER_350.to_dict()
    )
    accounting = base_token_accounting(foundation)
    base_weights = foundation["model"]
    foundation_manifest_sha = foundation["refresh_manifest_sha256"]
    foundation_processed = int(foundation["processed_tokens"])
    del foundation

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seq = 512
    val_indices = validation_indices(len(splits["validation"]))

    def make_batch(rows: list[dict], picks: list[int]):
        selected = [rows[i] for i in picks]
        width = dynamic_batch_width(selected, seq)
        x = torch.full((len(selected), width), eos, dtype=torch.long)
        y = torch.full((len(selected), width), -100, dtype=torch.long)
        for i, row in enumerate(selected):
            n = len(row["input_ids"])
            x[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
            y[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
        return x.to(device), y.to(device)

    def fresh_training_model():
        model = Neral350M().to(device=device, dtype=torch.float32)
        model.load_state_dict(base_weights, strict=True)
        if model.parameter_report()["total"] != PARAMETERS:
            raise RuntimeError("Neral parameter contract changed")
        return model

    def inference_clone(model):
        clone = Neral350M().to(device=device, dtype=torch.bfloat16)
        clone.load_state_dict(model.state_dict(), strict=True)
        return clone.eval()

    def validation(model) -> float:
        model.eval()
        pairs = []
        with torch.no_grad():
            for index in val_indices:
                x, y = make_batch(splits["validation"], [index])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                if loss is None or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite OASST2 validation loss")
                pairs.append((float(loss), int((y != -100).sum().item())))
        model.train()
        return token_weighted_loss(pairs)

    # Public development baseline for the exact supplied foundation checkpoint.
    base_train_model = fresh_training_model()
    baseline_sft_validation = validation(base_train_model)
    base_eval = inference_clone(base_train_model)
    baseline_public = run_model(torch, base_eval, tokenizer, bos, eos)
    baseline_knowledge = knowledge_choice_eval(torch, base_eval, tokenizer, bos)
    del base_eval, base_train_model
    torch.cuda.empty_cache()

    baseline_summary = baseline_public["summary"]
    baseline_categories = {
        k: baseline_public["pair"]["categories"][k]["accuracy_pct"]
        for k in PAIR_CATEGORIES
    }

    results = []
    best_key = None
    best_payload = None
    for index, cfg in enumerate(CONFIGS):
        guard.require_safe_runtime_temperature()
        torch.manual_seed(20260924)
        torch.cuda.manual_seed_all(20260924)
        random.seed(20260924)
        np.random.seed(20260924)
        torch.cuda.reset_peak_memory_stats()

        model = fresh_training_model()
        named = configure_trainable(model, cfg["blocks"])
        params = [param for _, param in named]
        trainable = sum(param.numel() for param in params)
        optimizer = torch.optim.AdamW(
            params, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.01, fused=True
        )
        require_fp32_trainable_subset(model, optimizer, torch, require_moments=False)
        rng = random.Random(20260924)
        supervised_tokens = 0
        started = time.monotonic()
        model.train()

        for step in range(1, cfg["steps"] + 1):
            picks = [rng.randrange(len(splits["train"])) for _ in range(4)]
            x, y = make_batch(splits["train"], picks)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            if loss is None or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite OASST2 SFT loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if not math.isfinite(float(grad)):
                raise FloatingPointError("nonfinite OASST2 SFT gradient")
            optimizer.step()
            if step == 1:
                require_fp32_trainable_subset(model, optimizer, torch, require_moments=True)
            supervised_tokens += int((y != -100).sum().item())
            if torch.cuda.memory_reserved() > MAX_RESERVED:
                raise RuntimeError("OASST2-on-refresh diagnostic exceeds 9 GiB VRAM ceiling")
            if step % 25 == 0 or step == cfg["steps"]:
                torch.cuda.synchronize()
                guard.require_safe_runtime_temperature()

        elapsed = time.monotonic() - started
        sft_val = validation(model)
        eval_model = inference_clone(model)
        public = run_model(torch, eval_model, tokenizer, bos, eos)
        knowledge = knowledge_choice_eval(torch, eval_model, tokenizer, bos)
        del eval_model
        torch.cuda.empty_cache()

        summary = public["summary"]
        categories = {
            k: public["pair"]["categories"][k]["accuracy_pct"]
            for k in PAIR_CATEGORIES
        }
        no_pair_regression = all(
            categories[k] >= baseline_categories[k] for k in PAIR_CATEGORIES
        )
        no_knowledge_regression = all(
            knowledge[v]["accuracy_pct"] >= baseline_knowledge[v]["accuracy_pct"]
            for v in baseline_knowledge
        )
        generation_gain = (
            summary["nis2_exact_generation_pct"] > baseline_summary["nis2_exact_generation_pct"]
            or summary["free_answer_strict"] > baseline_summary["free_answer_strict"]
            or summary["free_answer_digit_prefix"] > baseline_summary["free_answer_digit_prefix"]
        )
        qualifies = (
            sft_val < baseline_sft_validation
            and summary["nis2_pair_overall_pct"] >= baseline_summary["nis2_pair_overall_pct"]
            and no_pair_regression
            and no_knowledge_regression
            and generation_gain
        )
        row = {
            "index": index, **cfg,
            "trainable_parameters": trainable,
            "supervised_answer_tokens": supervised_tokens,
            "sft_validation_loss": sft_val,
            "sft_validation_delta": sft_val - baseline_sft_validation,
            "public_summary": summary,
            "pair_categories": categories,
            "knowledge_choice_pct": {v: knowledge[v]["accuracy_pct"] for v in knowledge},
            "no_pair_category_regression": no_pair_regression,
            "no_knowledge_choice_regression": no_knowledge_regression,
            "generation_gain": generation_gain,
            "qualifies": qualifies,
            "training_wall_seconds": elapsed,
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        }
        results.append(row)
        print(json.dumps({"event": "oasst2_on_refresh_candidate", **row}, sort_keys=True), flush=True)

        key = (
            summary["free_answer_strict"],
            summary["free_answer_digit_prefix"],
            summary["nis2_exact_generation_pct"],
            summary["nis2_pair_overall_pct"],
            -sft_val,
        )
        if qualifies and (best_key is None or key > best_key):
            best_key = key
            best_payload = {
                "row": row,
                "model": {
                    name: tensor.detach().to("cpu", dtype=torch.float32)
                    for name, tensor in model.state_dict().items()
                },
                "optimizer": _cpu_tree(torch, optimizer.state_dict()),
                "sampler_rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": [v.cpu() for v in torch.cuda.get_rng_state_all()],
            }

        del optimizer, params, named, model
        torch.cuda.empty_cache()

    args.out.mkdir(parents=True, exist_ok=False)
    candidate_sha = None
    selected = None
    if best_payload is not None:
        selected = best_payload["row"]
        state = {
            "checkpoint_schema": SCHEMA,
            "classification": CLASSIFICATION,
            "model": best_payload["model"],
            "optimizer": best_payload["optimizer"],
            "sampler_rng": best_payload["sampler_rng"],
            "torch_rng": best_payload["torch_rng"],
            "cuda_rng": best_payload["cuda_rng"],
            "config": FRONTIER_350.to_dict(),
            "tokenizer_sha256": TOKENIZER_SHA,
            "foundation_checkpoint_sha256": args.foundation_sha256,
            "foundation_manifest_sha256": foundation_manifest_sha,
            "foundation_processed_tokens": foundation_processed,
            **accounting,
            "oasst2_source_archive_sha256": data_report["source_archive_sha256"],
            "oasst2_source_revision": data_report["source_revision"],
            "selected": selected,
            "master_parameter_dtype": "float32",
            "adamw_moment_dtype": "float32",
            "compute_autocast_dtype": "bfloat16",
            "human_review_complete": False,
            "independent_holdout_checked": False,
            "quality_gain_verified": False,
            "production_eligible": False,
        }
        checkpoint = args.out / "candidate_auto_oasst2_on_refresh.pt"
        atomic_save(torch, state, checkpoint)
        candidate_sha = sha256_file(checkpoint)

    report = {
        "event": "phoenix_auto_oasst2_on_refresh_finished",
        "classification": CLASSIFICATION,
        "foundation_checkpoint_sha256": args.foundation_sha256,
        "foundation_manifest_sha256": foundation_manifest_sha,
        "foundation_processed_tokens": foundation_processed,
        **accounting,
        "data_report": data_report,
        "baseline_sft_validation_loss": baseline_sft_validation,
        "baseline_public_summary": baseline_summary,
        "baseline_knowledge_choice_pct": {
            v: baseline_knowledge[v]["accuracy_pct"] for v in baseline_knowledge
        },
        "configs": results,
        "qualified_count": sum(row["qualifies"] for row in results),
        "selected": selected,
        "candidate_checkpoint_written": candidate_sha is not None,
        "candidate_sha256": candidate_sha,
        "human_review_complete": False,
        "independent_holdout_checked": False,
        "quality_gain_verified": False,
        "production_eligible": False,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": report["event"],
        "foundation_processed_tokens": foundation_processed,
        "baseline_public_summary": baseline_summary,
        "qualified_count": report["qualified_count"],
        "selected": selected,
        "candidate_sha256": candidate_sha,
        "production_eligible": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
