"""Apply the proven small OASST1 post-training recipe to a foundation-refresh checkpoint.

Development only. This creates a third lineage stage:
historical anchor -> fresh-AdamW foundation refresh -> OASST1 response-only SFT.
It never merges independent checkpoints. Candidate is written only when OASST1
validation improves and public NIS2 pair behavior does not regress vs the supplied
foundation-refresh base.
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
from phoenix_oasst1_sft_sweep import auto_prepare_oasst1
from phoenix_auto_sft_diagnostic import _cpu_tree, atomic_save, sha256_file
from phoenix_auto_sft_public_dev_eval import pair_eval, exact_generation_eval
from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = 349_825_152
FOUNDATION_SCHEMA = "phoenix_foundation_refresh_dev_v1"
FOUNDATION_CLASS = "DEVELOPMENT_ONLY_FRESH_ADAMW_FOUNDATION_REFRESH"
SFT_SCHEMA = "phoenix_refresh_oasst1_sft_dev_v1"
SFT_CLASS = "DEVELOPMENT_ONLY_FOUNDATION_REFRESH_PLUS_OASST1_SFT"
BLOCKS = 8
LR = 5e-6
STEPS = 100
PAIR_CATEGORIES = (
    "instruction_following", "dialogue_state", "truthfulness",
    "robustness", "compositional_reasoning",
)


def load_refresh(torch, path: Path, expected_sha: str, config: dict) -> dict:
    if sha256_file(path) != expected_sha:
        raise ValueError("foundation-refresh SHA mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    lineage = state.get("lineage")
    if (
        state.get("checkpoint_schema") != FOUNDATION_SCHEMA
        or state.get("classification") != FOUNDATION_CLASS
        or state.get("config") != config
        or state.get("tokenizer_sha256") != TOKENIZER_SHA
        or state.get("production_eligible") is not False
        or not isinstance(lineage, dict)
        or lineage.get("kind") != "phoenix_foundation_refresh_fresh_adamw_v1"
        or not isinstance(state.get("model"), dict)
        or not isinstance(state.get("optimizer"), dict)
        or not state["optimizer"].get("state")
        or type(state.get("processed_tokens")) is not int
        or state["processed_tokens"] <= 0
    ):
        raise ValueError("foundation-refresh checkpoint contract mismatch")
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--foundation-checkpoint", type=Path, required=True)
    p.add_argument("--foundation-sha256", required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit trust in own SHA-pinned checkpoint required")
    if not args.out.is_absolute() or args.out.exists() or args.out.is_symlink():
        p.error("output must be a new absolute directory")

    assets = args.assets_root.resolve(strict=True)
    tok_path = trusted_asset(
        assets, assets / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA
    )
    tokenizer, bos, eos, splits, data_report = auto_prepare_oasst1(
        REPO_ROOT, args.archive.resolve(strict=True), tok_path
    )

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    from protection_v9 import configure_trainable

    guard.require_cuda_ready(torch)
    foundation = load_refresh(
        torch, args.foundation_checkpoint, args.foundation_sha256, FRONTIER_350.to_dict()
    )
    foundation_tokens = int(foundation["processed_tokens"])
    foundation_manifest = foundation.get("refresh_manifest_sha256")
    foundation_model = foundation["model"]
    del foundation

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(20260923)
    torch.cuda.manual_seed_all(20260923)
    random.seed(20260923)
    np.random.seed(20260923)

    seq = 512
    val_indices = validation_indices(len(splits["validation"]))

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
                    raise FloatingPointError("nonfinite OASST1 validation")
                pairs.append((float(loss), int((y != -100).sum().item())))
        model.train()
        return token_weighted_loss(pairs)

    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(foundation_model, strict=True)
    if model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("parameter contract changed")
    del foundation_model

    model.eval()
    base_val = validation(model)
    base_pair = pair_eval(torch, model, tokenizer, bos)
    base_exact = exact_generation_eval(model, tokenizer, bos, eos, torch)
    base_categories = {
        k: base_pair["categories"][k]["accuracy_pct"] for k in PAIR_CATEGORIES
    }

    named = configure_trainable(model, BLOCKS)
    params = [param for _, param in named]
    trainable = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(
        params, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, fused=True
    )
    rng = random.Random(20260923)
    supervised = 0
    started = time.monotonic()
    model.train()

    for step in range(1, STEPS + 1):
        picks = [rng.randrange(len(splits["train"])) for _ in range(4)]
        x, y = make_batch(splits["train"], picks)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        if loss is None or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite OASST1 train loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not math.isfinite(float(grad)):
            raise FloatingPointError("nonfinite OASST1 gradient")
        optimizer.step()
        supervised += int((y != -100).sum().item())
        if step % 25 == 0:
            torch.cuda.synchronize()
            guard.require_safe_runtime_temperature()
            print(json.dumps({
                "event":"refresh_oasst1_step", "step":step,
                "supervised_answer_tokens":supervised, "loss":float(loss.detach()),
                "grad_norm":float(grad)
            }, sort_keys=True), flush=True)

    after_val = validation(model)
    model.eval()
    after_pair = pair_eval(torch, model, tokenizer, bos)
    after_exact = exact_generation_eval(model, tokenizer, bos, eos, torch)
    after_categories = {
        k: after_pair["categories"][k]["accuracy_pct"] for k in PAIR_CATEGORIES
    }
    no_category_regression = all(
        after_categories[k] >= base_categories[k] for k in PAIR_CATEGORIES
    )
    qualifies = (
        after_val < base_val
        and after_pair["overall_pct"] > base_pair["overall_pct"]
        and no_category_regression
        and after_exact["accuracy_pct"] >= base_exact["accuracy_pct"]
    )

    args.out.mkdir(parents=True, exist_ok=False)
    candidate_sha = None
    if qualifies:
        state = {
            "checkpoint_schema": SFT_SCHEMA,
            "classification": SFT_CLASS,
            "model": {n:t.detach().to("cpu", dtype=torch.bfloat16)
                      for n,t in model.state_dict().items()},
            "optimizer": _cpu_tree(torch, optimizer.state_dict()),
            "config": FRONTIER_350.to_dict(),
            "tokenizer_sha256": TOKENIZER_SHA,
            "foundation_checkpoint_sha256": args.foundation_sha256,
            "foundation_new_tokens": foundation_tokens,
            "refresh_manifest_sha256": foundation_manifest,
            "sft_source_archive_sha256": sha256_file(args.archive),
            "sft_source_revision": data_report["source_revision"],
            "sft_supervised_tokens": supervised,
            "sft_steps": STEPS,
            "train_blocks": BLOCKS,
            "trainable_parameters": trainable,
            "selection": {
                "validation_before": base_val, "validation_after": after_val,
                "pair_before": base_pair["overall_pct"],
                "pair_after": after_pair["overall_pct"],
                "exact_before": base_exact["accuracy_pct"],
                "exact_after": after_exact["accuracy_pct"],
                "no_pair_category_regression": no_category_regression,
            },
            "human_review_complete": False,
            "independent_holdout_checked": False,
            "production_eligible": False,
        }
        checkpoint = args.out / "candidate_refresh_plus_oasst1.pt"
        atomic_save(torch, state, checkpoint)
        candidate_sha = sha256_file(checkpoint)

    report = {
        "event":"phoenix_refresh_oasst1_sft_finished",
        "foundation_checkpoint_sha256":args.foundation_sha256,
        "foundation_new_tokens":foundation_tokens,
        "sft_supervised_answer_tokens":supervised,
        "training_wall_seconds":time.monotonic()-started,
        "validation_before":base_val,
        "validation_after":after_val,
        "pair_before":base_pair["overall_pct"],
        "pair_after":after_pair["overall_pct"],
        "exact_generation_before":base_exact["accuracy_pct"],
        "exact_generation_after":after_exact["accuracy_pct"],
        "categories_before":base_categories,
        "categories_after":after_categories,
        "qualifies":qualifies,
        "candidate_checkpoint_written":candidate_sha is not None,
        "candidate_sha256":candidate_sha,
        "production_eligible":False,
    }
    (args.out/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
