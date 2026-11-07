"""NERAL-EXT manual read-only three-way local evaluation (no training).

Variant runs always use a fresh process, one 4-bit model at a time.
Never upload raw model responses; combine emits scalar counts only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable

from scripts.neral_ext import (
    _mcnemar, _value, evaluate, read_cases, read_responses, write_report,
)
from scripts.neral_v5_style_eval import BASE_REVISION
from scripts.neral_ext_resume import write_variant_checkpointed

SYSTEM = ("Du bist NERAL. Löse technische Aufgaben präzise. "
          "Gib eine eindeutige Endantwort; erfinde keine Tests oder Quellen.")
VARIANTS = frozenset({"base", "base_fewshot", "neral_v5"})
FEWSHOT = (
    ("Wie viel ist 6 + 7? Gib nur das Ergebnis.", "13"),
    ("Welcher Bruch ist gleich 2/4? Gib nur den gekürzten Bruch.", "1/2"),
    ("Wie viel sind 150 Cent in EUR? Betrag und EUR angeben.", "1,50 EUR"),
)


def build_messages(case: dict, variant: str) -> list[dict]:
    """Same system prompt and target input; base_fewshot adds explicit controls."""
    if variant not in VARIANTS:
        raise ValueError("Unknown local benchmark variant")
    if not isinstance(case.get("prompt"), str):
        raise ValueError("Invalid task")
    messages = [{"role": "system", "content": SYSTEM}]
    if variant == "base_fewshot":
        for task, answer in FEWSHOT:
            messages.extend(({"role": "user", "content": task},
                             {"role": "assistant", "content": answer}))
    messages.append({"role": "user", "content": case["prompt"]})
    return messages


def write_variant(cases: list[dict], variant: str, dest: Path,
                  generate: Callable[[dict, list[dict]], dict]) -> dict:
    """Call a provided local generator, create output once, fail on malformed data."""
    dest = Path(dest)
    if variant not in VARIANTS or dest.is_symlink() or dest.exists() or not dest.parent.is_dir():
        raise ValueError("Variant/output path not approved")
    if not cases or len(cases) > 500 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Unexpected holdout IDs")
    rows = []
    tokens = 0
    incomplete = 0
    for case in cases:
        record = generate(case, build_messages(case, variant))
        if (not isinstance(record, dict) or set(record) != {"text", "complete", "new_tokens"}
                or not isinstance(record["text"], str)
                or type(record["complete"]) is not bool
                or type(record["new_tokens"]) is not int
                or not 0 <= record["new_tokens"] <= 512):
            raise ValueError("Untrusted generator record")
        raw = record["text"]
        complete = record["complete"] and 0 < len(raw) <= 4096
        if not complete:
            raw = ""  # Never score truncated/partial generations.
            incomplete += 1
        rows.append({"id": case["id"], "text": raw, "complete": complete})
        tokens += record["new_tokens"]
    with dest.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"cases": len(cases), "generated_tokens": tokens, "incomplete": incomplete,
            "status": "LOCAL_RESPONSES_SAVED_NOT_VALIDATED"}


def _format_ok(case: dict, response: dict) -> bool:
    """Format alone, without reference-answer correctness."""
    if not response["complete"]:
        return False
    s = response["text"].strip()
    if case["kind"] == "python":
        if len(s) > 4096:
            return False
        try:
            tree = ast.parse(s, mode="exec")
            return bool(tree.body and any(isinstance(x, ast.FunctionDef) for x in tree.body))
        except (SyntaxError, ValueError, RecursionError):
            return False
    return _value(case["kind"], s) is not None


def _statistics(cases: list[dict], scored: dict, responses: dict) -> dict:
    return {
        "total": len(cases),
        "format_valid": sum(_format_ok(case, responses[case["id"]]) for case in cases),
        "strict_correct": sum(scored[case["id"]]["strict"] for case in cases),
        "semantic_correct": sum(scored[case["id"]]["semantic"] == "correct" for case in cases),
        "ungraded": sum(scored[case["id"]]["semantic"] == "ungraded" for case in cases),
    }


def combine_scores(cases: list[dict], meta: list[dict], base: dict, few: dict, neral: dict,
                   *, case_sha: str, base_revision: str, adapter_sha: str) -> dict:
    """Aggregate three aligned evaluations. Never publish raw responses or prompts."""
    from scripts.neral_ext import SHAS, REV
    if (not isinstance(case_sha, str) or SHAS.fullmatch(case_sha) is None
            or not isinstance(adapter_sha, str) or SHAS.fullmatch(adapter_sha) is None
            or not isinstance(base_revision, str) or REV.fullmatch(base_revision) is None):
        raise ValueError("Unpinned model or dataset")
    if (len(cases) != len(meta) or len({c["id"] for c in cases}) != len(cases)
            or {c["id"] for c in cases} != {m.get("id") for m in meta}
            or any(m.get("stratum") not in "ABCD" for m in meta)):
        raise ValueError("Unmatched or invalid stratum manifest")
    results = {name: evaluate(cases, responses)
               for name, responses in [("base", base), ("base_fewshot", few),
                                        ("neral_v5", neral)]}
    responses = {"base": base, "base_fewshot": few, "neral_v5": neral}
    label = {m["id"]: m["stratum"] for m in meta}

    def subset(chunk: list[dict]) -> dict:
        stats = {name: _statistics(chunk, results[name], responses[name])
                 for name in results}
        improved = sum(results["base_fewshot"][c["id"]]["semantic"] != "correct"
                       and results["neral_v5"][c["id"]]["semantic"] == "correct"
                       for c in chunk)
        regressions = sum(results["base_fewshot"][c["id"]]["semantic"] == "correct"
                          and results["neral_v5"][c["id"]]["semantic"] != "correct"
                          for c in chunk)
        return {**stats, "paired_fewshot_vs_neral": {
            "improvements": improved, "regressions": regressions,
            "net_gain": improved - regressions,
            "exact_mcnemar_p": _mcnemar(improved, regressions)}}

    full = subset(cases)
    strata = {part: subset([c for c in cases if label[c["id"]] == part])
              for part in "ABCD"}
    transfer = [c for c in cases if label[c["id"]] in ("A", "B")]
    t = subset(transfer)
    gap = 100.0 * (t["neral_v5"]["semantic_correct"] -
                   t["base_fewshot"]["semantic_correct"]) / len(transfer) if transfer else 0.0
    return {
        "schema": 1, "status": "NERAL_EXT_THREE_WAY_UNVERIFIED_HOLDOUT",
        "total": len(cases), "case_sha256": case_sha,
        "base_revision": base_revision, "adapter_sha256": adapter_sha,
        "overall": full, "strata": strata,
        "format_adjusted_gap_A_B": round(gap, 4),
        "format_adjusted_gap_unit": "percentage_points",
        "independent_external_benchmark": False,
        "general_improvement_proven": False, "release_validated": False,
        "weights_promoted": False,
        "limitations": ("Public tasks not guaranteed unseen by Qwen pretraining; "
                        "MBPP Python correctness ungraded without sandbox; "
                        "overlap gate and identical decode configuration separately required."),
    }


def pinned_runtime_identity():
    """Use Helios' audited Qwen revision/LoRA validators without shadowable names.

    No trainer checkout imports; metadata and weight digests are checked against
    the same immutable Qwen2.5 foundation revision. The trainer folder itself
    remains required by the manually gated Windows evaluation workflow.
    """
    from neral_hf_identity import (
        CHAT_TEMPLATE, validate_hf_snapshot, validate_adapter,
    )
    return CHAT_TEMPLATE, validate_hf_snapshot, validate_adapter


def _gpu_run(cases: list[dict], variant: str, args) -> dict:
    """Manual only: sealed local Qwen and adapter, one model process, no network."""
    import os
    import torch
    from transformers import Qwen2TokenizerFast
    from peft import PeftModel
    from scripts.neral_math_eval import _load_model
    from scripts.neral_v3_eval_isolation import prepare_cuda, clean_cuda

    for env in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[env] = "1"
    root = args.trainer_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Pinned trainer missing")
    # No sys.path mutation: both control/ and trainer/ contain a model_catalog
    # module, and Python may cache the wrong one before GPU inference begins.
    CHAT_TEMPLATE, validate_hf_snapshot, validate_adapter = pinned_runtime_identity()
    snapshot = validate_hf_snapshot(args.base, verify_hashes=True)
    if snapshot.get("revision") != BASE_REVISION:
        raise ValueError("Wrong pinned Qwen foundation")
    seal = validate_adapter(args.adapter, snapshot)
    if seal.get("release_validated") is not False:
        raise ValueError("Only nonpromoted experimental v5 allowed")
    if not 64 <= args.max_new_tokens <= 384:
        raise ValueError("Unsafe new-token limit")
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        str(args.base), local_files_only=True, trust_remote_code=False, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepare_cuda(torch)
    model = None
    start = time.monotonic()
    try:
        model = _load_model(args.base)
        if variant == "neral_v5":
            model = PeftModel.from_pretrained(model, str(args.adapter),
                                              is_trainable=False, local_files_only=True)
        model.eval()
        def generate(case: dict, messages: list[dict]) -> dict:
            if time.monotonic() - start > args.max_seconds:
                raise TimeoutError("Whole run time budget exhausted before completion")
            tokens = tokenizer.apply_chat_template(
                messages, chat_template=CHAT_TEMPLATE, tokenize=True,
                add_generation_prompt=True, return_tensors="pt").to("cuda:0")
            if tokens.shape[-1] > 3072:
                return {"text": "", "complete": False, "new_tokens": 0}
            with torch.inference_mode():
                generated = model.generate(
                    tokens, attention_mask=torch.ones_like(tokens),
                    do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id)
            output = generated[0, tokens.shape[-1]:]
            count = int(output.shape[-1])
            text = tokenizer.decode(output, skip_special_tokens=True)
            complete = bool(count and int(output[-1]) == tokenizer.eos_token_id)
            return {"text": text, "complete": complete, "new_tokens": count}
        # Bind resumed answers to exact, locally validated weights and decoding.
        run_identity = {
            "base_revision": BASE_REVISION,
            "base_manifest_sha256": hashlib.sha256(
                (args.base / "helios-manifest.json").read_bytes()).hexdigest(),
            "adapter_weights_sha256": (
                seal["files"]["adapter_model.safetensors"]
                if variant == "neral_v5" else None),
            "variant": variant,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy_do_sample_false",
            "chat_template": CHAT_TEMPLATE,
            "checkpoint_protocol": 1,
        }
        run_key = hashlib.sha256(json.dumps(
            run_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return write_variant_checkpointed(
            cases, variant, args.out, generate, build_messages,
            run_key=run_key)
    finally:
        if model is not None:
            del model
        clean_cuda(torch)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--mode", choices=("run", "combine"), required=True)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--case-sha256", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--variant", choices=sorted(VARIANTS))
    p.add_argument("--base", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--trainer-root", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-seconds", type=int, default=21600)
    p.add_argument("--base-responses", type=Path)
    p.add_argument("--fewshot-responses", type=Path)
    p.add_argument("--neral-responses", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--adapter-sha")
    args = p.parse_args(argv)
    cases = read_cases(args.cases, args.case_sha256)
    if args.mode == "run":
        if (not args.variant or not args.base or not args.adapter or
                not args.trainer_root or not 3600 <= args.max_seconds <= 21600):
            p.error("Explicit local model paths, mode and bounded budget required")
        info = _gpu_run(cases, args.variant, args)
        print(json.dumps(info, sort_keys=True))  # No prompts or answers.
        return
    if not all([args.base_responses, args.fewshot_responses,
                args.neral_responses, args.manifest, args.adapter_sha]):
        p.error("Three responses, pinned manifest and adapter sha required")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != args.case_sha256:
        p.error("Case manifest hash mismatch")
    result = combine_scores(
        cases, manifest["cases"], read_responses(args.base_responses),
        read_responses(args.fewshot_responses), read_responses(args.neral_responses),
        case_sha=args.case_sha256, base_revision=BASE_REVISION,
        adapter_sha=args.adapter_sha)
    write_report(args.out, result)
    print("NERAL_EXT_THREE_WAY: scalar-only report saved, no weight promotion")


if __name__ == "__main__":
    main()
