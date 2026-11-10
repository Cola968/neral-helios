"""Bounded, opt-in real-checkpoint diagnostic. PUBLIC development items, NOT a holdout.

No training, network calls, GPU, data uploads or checkpoint writes. Outputs stay
in an existing private local directory. Exact-match rates are deliberately
narrow and cannot establish general chat quality or an improvement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import helios350_preview as raw
import helios_guarded_preview as guarded
from phoenix_continuity_cpu_probe import CONTINUITY_SHA, EXPECTED_TOKENS

# Intentionally tiny, fixed, publicly visible diagnostics; never a hidden test.
DEV_CASES = (
    ("de_add", "Antworte ausschließlich mit einer Ziffer: Was ist 2 + 2?", "4"),
    ("en_add", "Answer with a single digit only: What is 1 + 1?", "2"),
    ("de_sub", "Antworte ausschließlich mit einer Ziffer: Was ist 7 - 4?", "3"),
)
BUDGET = 16


def as_prompt(tokenizer, bos: int, question: str) -> list[int]:
    text = "User: " + question + "\nAssistant:"
    plain = tokenizer.encode(text, add_special_tokens=False).ids
    if tokenizer.decode(plain, skip_special_tokens=True) != text:
        raise ValueError("prompt tokenizer round-trip mismatch")
    if plain != tokenizer.encode(text).ids:
        raise ValueError("unexpected default tokenizer postprocessor: preview and SFT may disagree")
    ids = [bos, *plain]
    if len(ids) > raw.MAX_INPUT_TOKENS:
        raise ValueError("diagnostic prompt too long")
    return ids


def check_kv_parity(model, tokenizer, bos: int, torch, question: str) -> dict:
    """Compare native full causal logits with cached prompt and first appended token."""
    from kv_decode import cached_last_logits
    ids = as_prompt(tokenizer, bos, question)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        native, _ = model(x)
        cached, past = cached_last_logits(model, x)
        first = int(native[0, -1].argmax().item())
        extended = torch.tensor([ids + [first]], dtype=torch.long)
        native2, _ = model(extended)
        cached2, _ = cached_last_logits(model, torch.tensor([[first]], dtype=torch.long), past)
    results = []
    for reference, candidate in ((native[:, -1, :], cached), (native2[:, -1, :], cached2)):
        ref = reference.float()
        cand = candidate.float()
        delta = float((ref - cand).abs().max().item())
        same = int(ref.argmax().item()) == int(cand.argmax().item())
        results.append({"top1_equal": same, "max_abs_logit_delta": delta,
                        "close_atol_2e_3_rtol_1e_3": bool(torch.allclose(ref, cand, atol=2e-3, rtol=1e-3))})
    return {"prompt": results[0], "incremental_token": results[1],
            "top1_parity_pass": all(item["top1_equal"] for item in results),
            "numerical_parity_pass": all(item["close_atol_2e_3_rtol_1e_3"] for item in results)}


def output_target(private_dir: Path, name: str) -> Path:
    if not name.startswith("helios_quality_") or not name.endswith(".json") or not name.replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise ValueError("output must be a simple helios_quality_*.json filename")
    if private_dir.is_symlink() or not private_dir.is_dir():
        raise ValueError("existing, non-symlinked private directory required")
    root = private_dir.resolve(strict=True)
    if root.name.lower() != "phoenix" or root.parent.name.lower() != "neralprivateassets":
        raise ValueError("refuse output outside D:/NeralPrivateAssets/phoenix")
    target = root / name
    if target.is_symlink() or target.exists():
        raise FileExistsError("refuse existing diagnostic output")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--trusted-local-checkpoint", action="store_true")
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--private-dir", type=Path, required=True)
    ap.add_argument("--output-name", required=True)
    args = ap.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        ap.error("explicit execution and opt-in trusted local checkpoint are required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        ap.error("CPU isolation required: CUDA_VISIBLE_DEVICES=-1")
    target = output_target(args.private_dir, args.output_name)
    model, tok, bos, eos, processed, torch = guarded.load_continuity_cpu(args.assets_root, args.checkpoint)
    if processed != EXPECTED_TOKENS or next(model.parameters()).device.type != "cpu":
        raise RuntimeError("wrong checkpoint/device")
    parity = check_kv_parity(model, tok, bos, torch, DEV_CASES[0][1])
    rows = []
    for case_id, question, expected in DEV_CASES:
        as_prompt(tok, bos, question)  # Same prompt bytes and special-token contract.
        raw_text, raw_used = raw.generate(model, tok, bos, eos, torch, question, BUDGET)
        guard_text, guard_used, reason = guarded.guarded_generate(model, tok, bos, eos, torch, question, BUDGET)
        rows.append({"id": case_id, "question": question, "expected_exact": expected,
                     "raw": {"answer": raw_text, "generated_non_eos_tokens": raw_used,
                             "strict_exact": raw_text.strip() == expected},
                     "guarded": {"answer": guard_text, "steps": guard_used,
                                 "finish_reason": reason, "strict_exact": guard_text.strip() == expected}})
    report = {"status": "public_development_probe_not_independent_benchmark",
              "checkpoint_sha256": CONTINUITY_SHA, "processed_tokens": processed,
              "model_weights_modified": False, "training_started": False, "device": "cpu",
              "max_new_tokens_each": BUDGET, "same_checkpoint_same_prompt": True,
              "kv_native_cached_parity": parity, "cases": rows,
              "strict_exact_raw": sum(row["raw"]["strict_exact"] for row in rows),
              "strict_exact_guarded": sum(row["guarded"]["strict_exact"] for row in rows),
              "denominator": len(rows), "general_quality_improvement_proven": False,
              "independent_holdout": False}
    # Store actual own-model outputs only locally, never in CI logs or GitHub artifacts.
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"event": "own_weight_diagnostic_completed",
                      "checkpoint_sha256": CONTINUITY_SHA, "processed_tokens": processed,
                      "cases": len(rows), "strict_exact_raw": report["strict_exact_raw"],
                      "strict_exact_guarded": report["strict_exact_guarded"],
                      "kv_prompt_top1": parity["prompt"]["top1_equal"],
                      "kv_incremental_top1": parity["incremental_token"]["top1_equal"],
                      "kv_numerical_parity": parity["numerical_parity_pass"],
                      "local_private_output": str(target), "training_started": False,
                      "general_quality_improvement_proven": False}, sort_keys=True), flush=True)
    if not parity["top1_parity_pass"] or not parity["numerical_parity_pass"]:
        raise SystemExit("KV parity failed; do NOT enable cached inference")


if __name__ == "__main__":
    main()
