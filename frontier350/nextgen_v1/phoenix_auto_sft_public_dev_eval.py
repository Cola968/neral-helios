"""Compare immutable Neral 350M anchor vs auto-SFT candidate on public dev suites.

This is DEVELOPMENT evidence only, not an independent holdout or promotion gate.
The auto-SFT training path exact-hash-excluded the NIS2 chat-core prompts before
training. Raw per-case generations are written only to a private local JSON file;
Actions logs receive aggregates only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_nis2_chat50 import PAIR_CASES, EXACT_GENERATION_CASES, suite_sha256
from helios350_preview import TOKENIZER_SHA, generate, trusted_asset
from helios_free_answer_dev_probe import selected_cases, instruction, score
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_weight_only_pilot import _checked_historical_weights

BASE_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
PARAMETERS = 349_825_152
CANDIDATE_SCHEMA = "phoenix_auto_sft_diagnostic_v2_fp32_master"
CANDIDATE_CLASS = "auto_screened_public_oasst2_diagnostic_not_human_approved"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def conditional_nll(torch, model, tokenizer, bos: int, prefix: str, continuation: str) -> float:
    prefix_ids = [bos, *tokenizer.encode(prefix, add_special_tokens=False).ids]
    cont = tokenizer.encode(continuation, add_special_tokens=False).ids
    if not cont:
        raise ValueError("empty continuation tokens")
    full = [*prefix_ids, *cont]
    if len(full) > 512:
        raise ValueError("development pair exceeds trained context")
    device = next(model.parameters()).device
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
        start = len(prefix_ids) - 1
        selected = logits[0, start:start + len(cont)].float()
        target = torch.tensor(cont, dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(selected, target, reduction="mean")
    value = float(loss)
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite conditional NLL")
    return value


def pair_eval(torch, model, tokenizer, bos: int) -> dict:
    categories = collections.defaultdict(list)
    rows = []
    for i, (category, prefix, preferred, distractor) in enumerate(PAIR_CASES):
        good = conditional_nll(torch, model, tokenizer, bos, prefix, preferred)
        bad = conditional_nll(torch, model, tokenizer, bos, prefix, distractor)
        correct = good < bad
        categories[category].append(correct)
        rows.append({
            "id": f"nis2-pair-{i:03d}", "category": category,
            "correct": bool(correct), "margin_nats_per_token": bad - good,
        })
    return {
        "overall_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "categories": {
            k: {"correct": sum(v), "total": len(v),
                "accuracy_pct": 100.0 * sum(v) / len(v)}
            for k, v in sorted(categories.items())
        },
        "rows": rows,
    }


def exact_generation_eval(model, tokenizer, bos: int, eos: int, torch) -> dict:
    rows = []
    for case_id, prompt, accepted in EXACT_GENERATION_CASES:
        answer, used = generate(model, tokenizer, bos, eos, torch, prompt, 24)
        text = answer.strip()
        rows.append({
            "id": case_id, "correct": text in accepted,
            "accepted": accepted, "answer": answer, "generated_tokens": used,
        })
    return {
        "correct": sum(r["correct"] for r in rows),
        "total": len(rows),
        "accuracy_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "rows": rows,
    }


def free_answer_eval(model, tokenizer, bos: int, eos: int, torch) -> dict:
    rows = []
    for case_id, domain, language, question, expected, _ in selected_cases():
        prompt = instruction(question, language)
        answer, used = generate(model, tokenizer, bos, eos, torch, prompt, 16)
        rows.append({
            "id": case_id, "domain": domain, "language": language,
            "expected": expected, "answer": answer, "generated_tokens": used,
            "score": score(answer, expected),
        })
    return {
        "strict_exact": sum(r["score"]["strict_exact"] for r in rows),
        "starts_with_correct_digit": sum(r["score"]["starts_with_correct_digit"] for r in rows),
        "nonempty": sum(r["score"]["nonempty"] for r in rows),
        "total": len(rows),
        "rows": rows,
    }


def summarize(pair: dict, exact: dict, free: dict) -> dict:
    return {
        "nis2_pair_overall_pct": pair["overall_pct"],
        "nis2_instruction_pair_pct": pair["categories"]["instruction_following"]["accuracy_pct"],
        "nis2_dialogue_state_pct": pair["categories"]["dialogue_state"]["accuracy_pct"],
        "nis2_truthfulness_pct": pair["categories"]["truthfulness"]["accuracy_pct"],
        "nis2_robustness_pct": pair["categories"]["robustness"]["accuracy_pct"],
        "nis2_compositional_reasoning_pct": pair["categories"]["compositional_reasoning"]["accuracy_pct"],
        "nis2_exact_generation_pct": exact["accuracy_pct"],
        "free_answer_strict": free["strict_exact"],
        "free_answer_digit_prefix": free["starts_with_correct_digit"],
        "free_answer_nonempty": free["nonempty"],
        "free_answer_total": free["total"],
    }


def run_model(torch, model, tokenizer, bos: int, eos: int) -> dict:
    model.eval()
    pair = pair_eval(torch, model, tokenizer, bos)
    exact = exact_generation_eval(model, tokenizer, bos, eos, torch)
    free = free_answer_eval(model, tokenizer, bos, eos, torch)
    return {"pair": pair, "exact_generation": exact, "free_answer": free,
            "summary": summarize(pair, exact, free)}


def load_candidate(torch, path: Path, expected_sha: str, config: dict):
    if sha256_file(path) != expected_sha:
        raise ValueError("candidate SHA mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("checkpoint_schema") != CANDIDATE_SCHEMA or
            state.get("classification") != CANDIDATE_CLASS or
            state.get("base_checkpoint_sha256") != BASE_SHA or
            state.get("config") != config or
            state.get("tokenizer_sha256") != TOKENIZER_SHA or
            state.get("production_eligible") is not False or
            state.get("master_parameter_dtype") != "float32" or
            state.get("adamw_moment_dtype") != "float32" or
            state.get("compute_autocast_dtype") != "bfloat16" or
            state.get("human_review_complete") is not False or
            state.get("independent_holdout_checked") is not False or
            not isinstance(state.get("model"), dict)):
        raise ValueError("candidate metadata contract mismatch")
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoints", action="store_true")
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--candidate-checkpoint", type=Path, required=True)
    p.add_argument("--candidate-sha256", required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoints:
        p.error("explicit opt-in to own SHA-pinned checkpoints required")
    if args.out.exists() or not args.out.is_absolute():
        p.error("output must be a new absolute local file")

    base_path = require_pinned_checkpoint(args.base_checkpoint)
    root = args.assets_root.resolve(strict=True)
    tok_path = trusted_asset(root, root / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as gpu_guard
    gpu_guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M

    gpu_guard.require_cuda_ready(torch)
    tokenizer = Tokenizer.from_file(str(tok_path))
    if tokenizer.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocab mismatch")
    bos, eos = tokenizer.token_to_id("<|bos|>"), tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None or bos == eos:
        raise ValueError("invalid BOS/EOS")
    device = torch.device("cuda")

    base_state = _checked_historical_weights(base_path, torch, Neral350M, FRONTIER_350.to_dict())
    base_model = Neral350M().to(device=device, dtype=torch.bfloat16)
    base_model.load_state_dict(base_state["model"], strict=True)
    del base_state
    baseline = run_model(torch, base_model, tokenizer, int(bos), int(eos))
    del base_model
    torch.cuda.empty_cache()

    candidate_state = load_candidate(
        torch, args.candidate_checkpoint, args.candidate_sha256, FRONTIER_350.to_dict()
    )
    candidate_model = Neral350M().to(device=device, dtype=torch.bfloat16)
    candidate_model.load_state_dict(candidate_state["model"], strict=True)
    candidate_meta = {
        "sft_steps": candidate_state.get("sft_steps"),
        "sft_supervised_tokens": candidate_state.get("sft_supervised_tokens"),
        "train_blocks": candidate_state.get("train_blocks"),
        "source_revision": candidate_state.get("source_revision"),
    }
    del candidate_state
    candidate = run_model(torch, candidate_model, tokenizer, int(bos), int(eos))
    del candidate_model
    torch.cuda.empty_cache()
    gpu_guard.require_safe_runtime_temperature()

    b, c = baseline["summary"], candidate["summary"]
    deltas = {}
    for key in b:
        if isinstance(b[key], (int, float)) and isinstance(c.get(key), (int, float)):
            deltas[key] = c[key] - b[key]

    report = {
        "classification": "PUBLIC_DEVELOPMENT_AB_NOT_INDEPENDENT_HOLDOUT",
        "suite_sha256": suite_sha256(),
        "base_sha256": BASE_SHA,
        "candidate_sha256": args.candidate_sha256,
        "candidate_meta": candidate_meta | {
            "master_parameter_dtype": "float32",
            "adamw_moment_dtype": "float32",
            "compute_autocast_dtype": "bfloat16",
        },
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_base": deltas,
        "nis2_prompts_exactly_excluded_from_auto_sft_selection": True,
        "independent_holdout": False,
        "quality_gain_verified": False,
        "production_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "event": "phoenix_auto_sft_public_dev_ab_complete",
        "base": b, "candidate": c, "delta": deltas,
        "candidate_sha256": args.candidate_sha256,
        "independent_holdout": False, "production_eligible": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
