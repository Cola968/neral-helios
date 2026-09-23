"""Small, predeclared SFT hyperparameter sweep for Neral 350M.\n\nThe current sweep is a temporal/block-depth sweet-spot search: a candidate must now
also improve exact instruction generation over the immutable baseline, not merely
lower validation loss without public development regression.

Uses the same auto-screened, tree-disjoint OASST2 diagnostic data as the first
fast-lane experiment. Public NIS2 chat-core is used only as a DEVELOPMENT
selection suite and its exact prompts were excluded from the SFT source selector.

Only a candidate that improves SFT validation AND does not regress any NIS2 pair
category vs the immutable base is saved. No sealed holdout, no human data review,
no production promotion.
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
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_auto_sft_diagnostic import auto_prepare, _cpu_tree, atomic_save, sha256_file
from phoenix_auto_sft_public_dev_eval import pair_eval, exact_generation_eval
from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices

BASE_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
PARAMETERS = 349_825_152
CLASSIFICATION = "auto_sft_tuned_public_dev_selected_not_promotable"
SCHEMA = "phoenix_auto_sft_tuned_v1"
CONFIGS = (
    {"blocks": 4, "lr": 2e-5, "steps": 10},
    {"blocks": 4, "lr": 2e-5, "steps": 20},
    {"blocks": 4, "lr": 2e-5, "steps": 30},
    {"blocks": 4, "lr": 2e-5, "steps": 40},
    {"blocks": 3, "lr": 2e-5, "steps": 30},
    {"blocks": 3, "lr": 2e-5, "steps": 40},
    {"blocks": 3, "lr": 2e-5, "steps": 50},
    {"blocks": 3, "lr": 1.5e-5, "steps": 40},
    {"blocks": 3, "lr": 1.5e-5, "steps": 50},
)
PAIR_CATEGORIES = (
    "instruction_following", "dialogue_state", "truthfulness",
    "robustness", "compositional_reasoning",
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit execution and trust in own checkpoint required")
    if args.out.exists() or args.out.is_symlink() or not args.out.is_absolute():
        p.error("output must be new absolute directory")

    source = require_pinned_checkpoint(args.checkpoint)
    assets = args.assets_root.resolve(strict=True)
    private = args.private_root.resolve(strict=True)
    tokenizer_path = trusted_asset(
        assets, assets / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA
    )
    repo_root = Path(__file__).resolve().parents[2]
    tokenizer, bos, eos, splits, data_report = auto_prepare(
        repo_root, args.archive, private, tokenizer_path
    )

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
    if sha256_file(source) != BASE_SHA:
        raise ValueError("immutable base SHA changed")
    base_weights = base["model"]
    historical_tokens = int(base.get("processed_tokens", -1))
    device = torch.device("cuda")
    val_indices = validation_indices(len(splits["validation"]))
    seq = 512

    def make_batch(rows, picks):
        selected = [rows[i] for i in picks]
        width = dynamic_batch_width(selected, seq)
        x = torch.full((len(selected), width), eos, dtype=torch.long)
        y = torch.full((len(selected), width), -100, dtype=torch.long)
        for i, row in enumerate(selected):
            n = len(row["input_ids"])
            x[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
            y[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
        return x.to(device), y.to(device)

    def validation(model):
        model.eval()
        pairs = []
        with torch.no_grad():
            for index in val_indices:
                x, y = make_batch(splits["validation"], [index])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                if loss is None or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite validation")
                pairs.append((float(loss), int((y != -100).sum().item())))
        model.train()
        return token_weighted_loss(pairs)

    def fresh_model():
        m = Neral350M().to(device=device, dtype=torch.bfloat16)
        m.load_state_dict(base_weights, strict=True)
        if m.parameter_report()["total"] != PARAMETERS:
            raise RuntimeError("parameter contract changed")
        return m

    def evaluation_clone(trained_model):
        """Mirror the eventual saved candidate: all inference weights in BF16."""
        clone = Neral350M().to(device=device, dtype=torch.bfloat16)
        clone.load_state_dict(trained_model.state_dict(), strict=True)
        return clone.eval()

    # Immutable baseline metrics.
    base_model = fresh_model().eval()
    baseline_validation = validation(base_model)
    base_pair = pair_eval(torch, base_model, tokenizer, bos)
    base_exact = exact_generation_eval(base_model, tokenizer, bos, eos, torch)
    base_summary = {
        "validation_loss": baseline_validation,
        "pair_overall_pct": base_pair["overall_pct"],
        "exact_generation_pct": base_exact["accuracy_pct"],
        "categories": {k: base_pair["categories"][k]["accuracy_pct"] for k in PAIR_CATEGORIES},
    }
    del base_model
    torch.cuda.empty_cache()

    results = []
    best_key = None
    best_payload = None

    for index, config in enumerate(CONFIGS):
        gpu_guard.require_safe_runtime_temperature()
        torch.manual_seed(20260923)
        torch.cuda.manual_seed_all(20260923)
        random.seed(20260923)
        np.random.seed(20260923)
        torch.cuda.reset_peak_memory_stats()

        model = fresh_model()
        named = configure_trainable(model, config["blocks"])
        params = [p for _, p in named]
        trainable = sum(p.numel() for p in params)
        optimizer = torch.optim.AdamW(
            params, lr=config["lr"], betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.01, fused=True
        )
        rng = random.Random(20260923)
        supervised_total = 0
        started = time.monotonic()
        model.train()
        for step in range(1, config["steps"] + 1):
            picks = [rng.randrange(len(splits["train"])) for _ in range(4)]
            x, y = make_batch(splits["train"], picks)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            if loss is None or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite sweep train loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if not math.isfinite(float(grad)):
                raise FloatingPointError("nonfinite sweep gradient")
            optimizer.step()
            supervised_total += int((y != -100).sum().item())
            if step % 25 == 0 or step == config["steps"]:
                torch.cuda.synchronize()
                gpu_guard.require_safe_runtime_temperature()

        elapsed = time.monotonic() - started
        val = validation(model)
        eval_model = evaluation_clone(model)
        pair = pair_eval(torch, eval_model, tokenizer, bos)
        exact = exact_generation_eval(eval_model, tokenizer, bos, eos, torch)
        del eval_model
        torch.cuda.empty_cache()
        categories = {k: pair["categories"][k]["accuracy_pct"] for k in PAIR_CATEGORIES}
        no_pair_category_regression = all(
            categories[k] >= base_summary["categories"][k] for k in PAIR_CATEGORIES
        )
        qualifies = (
            val < baseline_validation
            and pair["overall_pct"] >= base_summary["pair_overall_pct"]
            and no_pair_category_regression
            and exact["accuracy_pct"] > base_summary["exact_generation_pct"]
        )
        result = {
            "index": index, **config, "trainable_parameters": trainable,
            "supervised_tokens": supervised_total,
            "validation_loss": val,
            "validation_delta": val - baseline_validation,
            "pair_overall_pct": pair["overall_pct"],
            "exact_generation_pct": exact["accuracy_pct"],
            "categories": categories,
            "no_pair_category_regression": no_pair_category_regression,
            "qualifies": qualifies,
            "training_wall_seconds": elapsed,
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        }
        results.append(result)
        print(json.dumps({"event": "auto_sft_sweep_candidate", **result}, sort_keys=True), flush=True)

        # Selection is predeclared: among hard-gated candidates, maximize exact
        # instruction generation, then pair overall, then lower validation loss.
        key = (exact["accuracy_pct"], pair["overall_pct"], -val)
        if qualifies and (best_key is None or key > best_key):
            best_key = key
            best_payload = {
                "config": dict(config),
                "trainable_parameters": trainable,
                "supervised_tokens": supervised_total,
                "validation_loss": val,
                "pair_overall_pct": pair["overall_pct"],
                "exact_generation_pct": exact["accuracy_pct"],
                "categories": categories,
                "model": {n: t.detach().to("cpu", dtype=torch.bfloat16)
                          for n, t in model.state_dict().items()},
                "optimizer": _cpu_tree(torch, optimizer.state_dict()),
                "sampler_rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": [v.cpu() for v in torch.cuda.get_rng_state_all()],
            }

        del optimizer, params, named, model
        torch.cuda.empty_cache()

    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = None
    selected = None
    if best_payload is not None:
        selected = {k: v for k, v in best_payload.items()
                    if k not in {"model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng"}}
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
            "base_checkpoint_sha256": BASE_SHA,
            "historical_pretraining_tokens": historical_tokens,
            "source_archive_sha256": data_report["source_archive_sha256"],
            "source_revision": data_report["source_revision"],
            "selection_protocol": {
                "configs": list(CONFIGS),
                "requires_validation_improvement": True,
                "requires_pair_overall_non_regression": True,
                "requires_all_pair_categories_non_regression": True,
                "ranking": "exact_generation_pct_then_pair_overall_then_lower_validation",
            },
            "selected": selected,
            "human_review_complete": False,
            "independent_holdout_checked": False,
            "production_eligible": False,
        }
        checkpoint = args.out / "candidate_tuned_auto_sft.pt"
        atomic_save(torch, state, checkpoint)
        checkpoint_sha = sha256_file(checkpoint)

    report = {
        "event": "phoenix_auto_sft_sweep_finished",
        "classification": CLASSIFICATION,
        "baseline": base_summary,
        "configs": results,
        "qualified_count": sum(r["qualifies"] for r in results),
        "selected": selected,
        "candidate_checkpoint_written": checkpoint_sha is not None,
        "candidate_sha256": checkpoint_sha,
        "human_review_complete": False,
        "independent_holdout_checked": False,
        "production_eligible": False,
    }
    (args.out / "sweep_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": report["event"], "baseline": base_summary,
        "qualified_count": report["qualified_count"], "selected": selected,
        "candidate_sha256": checkpoint_sha, "production_eligible": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
