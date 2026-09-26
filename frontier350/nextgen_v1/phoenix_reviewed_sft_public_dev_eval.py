"""Public-development A/B for a reviewed Phoenix SFT v2 candidate.

This is development evidence only. It validates the exact candidate/base lineage and
FP32-master metadata, then compares both checkpoints on the existing public NIS2/free-answer
development suites. It does not open the sealed private holdout and cannot promote a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helios350_preview import TOKENIZER_SHA, trusted_asset
from phoenix_auto_sft_public_dev_eval import run_model, sha256_file
from phoenix_weight_only_pilot import _checked_historical_weights

CANDIDATE_SCHEMA = "nextgen_sft_pilot_v2_fp32_master"
PARAMETERS = 349_825_152


def load_candidate(torch, path: Path, expected_sha: str, base_sha: str, config: dict) -> dict:
    if sha256_file(path) != expected_sha:
        raise ValueError("reviewed SFT candidate SHA mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state.get("checkpoint_schema") != CANDIDATE_SCHEMA
        or state.get("base_sha256") != base_sha
        or state.get("config") != config
        or state.get("tokenizer_sha256") != TOKENIZER_SHA
        or state.get("master_parameter_dtype") != "float32"
        or state.get("adamw_moment_dtype") != "float32"
        or state.get("compute_autocast_dtype") != "bfloat16"
        or state.get("production_eligible") is not False
        or not isinstance(state.get("model"), dict)
        or not isinstance(state.get("pack_report_sha256"), str)
        or not isinstance(state.get("source_manifest_sha256"), str)
        or not isinstance(state.get("screen_report_sha256"), str)
        or not isinstance(state.get("holdout_sha256"), str)
    ):
        raise ValueError("reviewed SFT candidate metadata contract mismatch")
    if any(not isinstance(state.get(k), int) or state[k] < 0 for k in (
        "historical_pretraining_tokens",
        "new_foundation_lineage_tokens",
        "cumulative_reference_tokens",
        "sft_supervised_tokens",
    )):
        raise ValueError("reviewed SFT token lineage missing")
    if (
        state["historical_pretraining_tokens"] + state["new_foundation_lineage_tokens"]
        != state["cumulative_reference_tokens"]
    ):
        raise ValueError("reviewed SFT cumulative token lineage mismatch")
    return state


def load_base(torch, path: Path, expected_sha: str, model_class, config: dict) -> dict:
    if sha256_file(path) != expected_sha:
        raise ValueError("base checkpoint SHA mismatch")
    # Historical source gets the stricter historical validator. A Phoenix v2 foundation
    # checkpoint is locally trusted by explicit SHA and checked below.
    if expected_sha == "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7":
        return _checked_historical_weights(path, torch, model_class, config)
    state = torch.load(path, map_location="cpu", weights_only=False)
    lineage = state.get("lineage")
    if (
        state.get("checkpoint_schema") != "phoenix_foundation_refresh_dev_v2"
        or state.get("classification") != "DEVELOPMENT_ONLY_FP32_MASTER_FOUNDATION_REFRESH"
        or state.get("config") != config
        or state.get("tokenizer_sha256") != TOKENIZER_SHA
        or state.get("master_parameter_dtype") != "float32"
        or state.get("adamw_moment_dtype") != "float32"
        or state.get("compute_autocast_dtype") != "bfloat16"
        or state.get("production_eligible") is not False
        or not isinstance(lineage, dict)
        or lineage.get("kind") != "phoenix_foundation_refresh_fp32_master_adamw_v2"
        or not isinstance(state.get("model"), dict)
    ):
        raise ValueError("base is neither historical anchor nor valid Phoenix FP32 foundation v2")
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoints", action="store_true")
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--base-sha256", required=True)
    p.add_argument("--candidate-checkpoint", type=Path, required=True)
    p.add_argument("--candidate-sha256", required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoints:
        p.error("explicit trust in SHA-pinned own checkpoints required")
    if args.out.exists() or not args.out.is_absolute():
        p.error("output must be a new absolute local file")
    for name, digest in (("base", args.base_sha256), ("candidate", args.candidate_sha256)):
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            p.error(f"{name} SHA-256 must be lowercase 64-hex")

    root = args.assets_root.resolve(strict=True)
    tok_path = trusted_asset(
        root, root / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA
    )

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M

    guard.require_cuda_ready(torch)
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    if tokenizer.get_vocab_size() != FRONTIER_350.vocab_size or bos is None or eos is None or bos == eos:
        raise ValueError("tokenizer contract mismatch")
    device = torch.device("cuda")

    base_state = load_base(
        torch, args.base_checkpoint, args.base_sha256, Neral350M, FRONTIER_350.to_dict()
    )
    base_model = Neral350M().to(device=device, dtype=torch.bfloat16)
    base_model.load_state_dict(base_state["model"], strict=True)
    if base_model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("base architecture changed")
    del base_state
    baseline = run_model(torch, base_model, tokenizer, int(bos), int(eos))
    del base_model
    torch.cuda.empty_cache()

    candidate_state = load_candidate(
        torch, args.candidate_checkpoint, args.candidate_sha256,
        args.base_sha256, FRONTIER_350.to_dict()
    )
    candidate_meta = {
        key: candidate_state.get(key) for key in (
            "historical_pretraining_tokens",
            "new_foundation_lineage_tokens",
            "cumulative_reference_tokens",
            "sft_supervised_tokens",
            "steps",
            "base_lineage_kind",
            "pack_report_sha256",
            "source_manifest_sha256",
            "screen_report_sha256",
            "holdout_sha256",
        )
    }
    candidate_model = Neral350M().to(device=device, dtype=torch.bfloat16)
    candidate_model.load_state_dict(candidate_state["model"], strict=True)
    if candidate_model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("candidate architecture changed")
    del candidate_state
    candidate = run_model(torch, candidate_model, tokenizer, int(bos), int(eos))
    del candidate_model
    torch.cuda.empty_cache()
    guard.require_safe_runtime_temperature()

    b = baseline["summary"]
    c = candidate["summary"]
    delta = {
        key: c[key] - b[key]
        for key in b
        if isinstance(b[key], (int, float)) and isinstance(c.get(key), (int, float))
    }
    report = {
        "classification": "PUBLIC_DEVELOPMENT_REVIEWED_SFT_AB_NOT_SEALED_HOLDOUT",
        "base_sha256": args.base_sha256,
        "candidate_sha256": args.candidate_sha256,
        "candidate_meta": candidate_meta,
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_base": delta,
        "sealed_private_holdout_opened": False,
        "quality_gain_verified": False,
        "production_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "event": "phoenix_reviewed_sft_public_dev_ab_complete",
        "base": b,
        "candidate": c,
        "delta": delta,
        "candidate_sha256": args.candidate_sha256,
        "sealed_private_holdout_opened": False,
        "production_eligible": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
