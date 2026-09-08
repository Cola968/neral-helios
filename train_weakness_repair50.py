"""Resume-safe Neral-50M I2 repair for EOS, repetition, math and code weaknesses.

I2 starts from the completed I1 instruction checkpoint. It reuses the already contamination-
screened I1 SFT data, deterministically rebalances it toward math/code, upweights the real EOS
target, and keeps document-bounded base-LM replay. Checkpoint selection uses only held-out I1
validation rows plus RawMath-dev and base-LM guards; the frozen benchmark is run only after the
weight stage. No decoding penalties, forced EOS, tools, external models or foreign weights are
used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from architectures import FIFTY
from experiment import atomic_save, load_model, sha, amp
from free_generation50 import diagnostics
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from train_instruct50 import (
    EXPECTED_PARAMS,
    TOKENIZER,
    assistant_loss,
    data_hashes,
    evaluate_sft,
    lr_at,
    make_sft_batch,
    next_rows,
    read_jsonl,
)
from train_instruct50_i1 import replay_signature
from train_raw_math50 import dev_math_nll
from weakness_repair_data50 import eligible_rows, select_profile_mix, selection_summary

STAGE_VERSION = 1
SEED = 968_720
CRITICAL_ARGS = (
    "steps", "seq", "batch_size", "replay_batch", "sft_weight", "eos_weight",
    "peak_lr", "min_lr", "warmup_fraction", "eval_every", "patience",
    "max_train_examples", "max_val_examples", "dev_generation_examples",
    "desired_de_fraction", "min_answer_bytes", "max_answer_bytes",
    "max_lm_regression", "max_math_regression", "min_health_gain",
    "min_sft_nll_gain", "min_score_gain",
)


def args_signature(args) -> dict:
    return {key: getattr(args, key) for key in CRITICAL_ARGS}


def domain_nll(report: dict, domain: str) -> float:
    values = [float(v["answer_nll"]) for key, v in report.get("slices", {}).items() if key.endswith("/" + domain)]
    if not values:
        raise ValueError(f"I2 validation has no {domain!r} slice")
    return sum(values) / len(values)


@torch.inference_mode()
def evaluate_free_dev(model, rows, device, max_new_tokens=80):
    was = model.training
    model.eval()
    items = []
    for row in rows:
        ids = TOKENIZER.encode("User: " + row["prompt"].strip() + "\nAssistant:", bos=True)
        x = torch.tensor([ids], device=device)
        with amp(device):
            out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0, eos_id=TOKENIZER.eos_id)
        generated = out[0, len(ids):].tolist()
        text = TOKENIZER.decode(generated)
        d = diagnostics(generated, text)
        items.append({"id": row.get("id"), "language": row.get("language"), "domain": row.get("domain"), "tokens": len(generated), "diagnostics": d})
    model.train(was)
    n = max(1, len(items))
    eos = 100 * sum(float(x["diagnostics"]["eos"]) for x in items) / n
    rep = 100 * sum(float(x["diagnostics"]["repeated_word_ngrams"]["3"]) for x in items) / n
    return {
        "examples": len(items),
        "utf8_valid_pct": 100 * sum(float(x["diagnostics"]["valid_utf8"]) for x in items) / n,
        "eos_pct": eos,
        "repeated_trigram_pct": rep,
        "mean_generated_tokens": sum(x["tokens"] for x in items) / n,
        "health_score": eos - rep,
        "note": "I2 held-out validation generation only; never a release benchmark.",
    }


def weakness_score(sft: dict, gen: dict, math_dev: dict, baseline_sft: dict, baseline_gen: dict, baseline_math: dict) -> float:
    b_math = domain_nll(baseline_sft, "math")
    b_code = domain_nll(baseline_sft, "code")
    c_math = domain_nll(sft, "math")
    c_code = domain_nll(sft, "code")
    math_sft_gain = 100 * (b_math - c_math) / max(b_math, 1e-9)
    code_sft_gain = 100 * (b_code - c_code) / max(b_code, 1e-9)
    math_dev_gain = 100 * (float(baseline_math["nll"]) - float(math_dev["nll"])) / max(float(baseline_math["nll"]), 1e-9)
    health_gain = float(gen["health_score"]) - float(baseline_gen["health_score"])
    return health_gain + 0.50 * math_sft_gain + 0.30 * code_sft_gain + 0.20 * math_dev_gain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--expected-source-sha")
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay-data", required=True)
    ap.add_argument("--out", default="runs/neral50m_i2_weakness_e1")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--seq", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--replay-batch", type=int, default=8)
    ap.add_argument("--sft-weight", type=float, default=.72)
    ap.add_argument("--eos-weight", type=float, default=3.0)
    ap.add_argument("--peak-lr", type=float, default=8e-6)
    ap.add_argument("--min-lr", type=float, default=8e-7)
    ap.add_argument("--warmup-fraction", type=float, default=.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max-train-examples", type=int, default=8000)
    ap.add_argument("--max-val-examples", type=int, default=512)
    ap.add_argument("--dev-generation-examples", type=int, default=96)
    ap.add_argument("--desired-de-fraction", type=float, default=.35)
    ap.add_argument("--min-answer-bytes", type=int, default=2)
    ap.add_argument("--max-answer-bytes", type=int, default=384)
    ap.add_argument("--max-lm-regression", type=float, default=.02)
    ap.add_argument("--max-math-regression", type=float, default=.02)
    ap.add_argument("--min-health-gain", type=float, default=1.5)
    ap.add_argument("--min-sft-nll-gain", type=float, default=.003)
    ap.add_argument("--min-score-gain", type=float, default=.50)
    args = ap.parse_args()

    positive = (args.steps, args.seq, args.batch_size, args.replay_batch, args.eval_every, args.patience,
                args.max_train_examples, args.max_val_examples, args.dev_generation_examples)
    if min(positive) <= 0:
        raise ValueError("positive I2 budgets required")
    if args.seq > FIFTY.max_seq_len:
        raise ValueError("sequence exceeds model context")
    if not .60 <= args.sft_weight <= .82:
        raise ValueError("I2 sft-weight must be 0.60..0.82")
    if not 1.5 <= args.eos_weight <= 3.0:
        raise ValueError("I2 eos-weight must be 1.5..3.0")
    if not 0 < args.min_lr <= args.peak_lr <= 1.2e-5:
        raise ValueError("I2 uses a bounded low learning rate <=1.2e-5")
    if not 0 <= args.warmup_fraction <= .2:
        raise ValueError("warmup-fraction must be 0..0.2")
    if not 0 <= args.desired_de_fraction <= 1:
        raise ValueError("desired-de-fraction must be 0..1")
    if not 0 <= args.max_lm_regression <= .05 or not 0 <= args.max_math_regression <= .05:
        raise ValueError("invalid I2 regression guards")
    if args.expected_source_sha and (len(args.expected_source_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.expected_source_sha)):
        raise ValueError("expected-source-sha must be a SHA-256")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("I2 requires BF16 CUDA")

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    rng = random.Random(SEED)
    device = torch.device("cuda")
    root = Path(args.data)
    replay_root = Path(args.replay_data)
    hashes = data_hashes(root)
    replay_sig = replay_signature(replay_root)

    raw_train = read_jsonl(root / "train.jsonl")
    raw_val = read_jsonl(root / "val.jsonl")
    train = select_profile_mix(
        eligible_rows(raw_train, seq=args.seq, min_answer_bytes=args.min_answer_bytes, max_answer_bytes=args.max_answer_bytes),
        max_examples=args.max_train_examples, min_examples=min(4000, args.max_train_examples), desired_de_fraction=args.desired_de_fraction,
    )
    val = select_profile_mix(
        eligible_rows(raw_val, seq=args.seq, min_answer_bytes=args.min_answer_bytes, max_answer_bytes=args.max_answer_bytes),
        max_examples=args.max_val_examples, min_examples=min(160, args.max_val_examples), desired_de_fraction=args.desired_de_fraction,
    )
    dev_rows = select_profile_mix(
        val, max_examples=min(args.dev_generation_examples, len(val)), min_examples=min(48, len(val)), desired_de_fraction=args.desired_de_fraction,
    )
    train_groups = {r["prompt_group_sha256"] for r in train}
    val_groups = {r["prompt_group_sha256"] for r in val}
    if train_groups & val_groups:
        raise ValueError("I2 train/val prompt groups overlap")

    replay_streams = load_b2_streams(replay_root, "train")
    replay_val = load_b2_streams(replay_root, "val")
    replay_docs = load_b2_documents(replay_root)
    replay_blocks = document_blocks(replay_docs, "de", args.seq, "train") + document_blocks(replay_docs, "en", args.seq, "train")
    if len(replay_blocks) < args.replay_batch:
        raise ValueError("insufficient replay blocks")
    replay_val_schedule, replay_val_meta = build_validation_schedule(replay_docs, args.seq, 96, SEED + 17)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    resume = latest.is_file()
    if not resume and any(out.iterdir()):
        raise FileExistsError("I2 output exists without latest.pt; refusing ambiguous resume")
    load_path = latest if resume else Path(args.init_from)
    model, payload = load_model(load_path, device)
    if model.parameter_report()["trainable"] != EXPECTED_PARAMS:
        raise ValueError("I2 accepts exact Neral-50M only")
    model.train()
    source_sha = sha(args.init_from)
    if args.expected_source_sha and source_sha.lower() != args.expected_source_sha.lower():
        raise ValueError("I2 source checkpoint SHA-256 mismatch")

    if resume:
        if payload.get("i2_stage") != "weakness_e1" or payload.get("i2_version") != STAGE_VERSION:
            raise ValueError("latest.pt is not I2 weakness E1")
        if payload.get("source_sha256") != source_sha:
            raise ValueError("I2 source checkpoint changed")
        if payload.get("sft_data_hashes") != hashes or payload.get("replay_signature") != replay_sig:
            raise ValueError("I2 data changed")
        if payload.get("args_signature") != args_signature(args):
            raise ValueError("I2 hyperparameters changed")
        source_tokens = int(payload["source_payload_tokens"])
        start_step = int(payload["i2_step"])
        selected_step = int(payload.get("i2_selected_step", 0))
        baseline_sft = payload["baseline_sft"]
        baseline_gen = payload["baseline_gen"]
        baseline_lm = payload["baseline_lm"]
        baseline_math = payload["baseline_math"]
        best_score = float(payload["best_weakness_score"])
        best_health = float(payload["best_health_score"])
        bad = int(payload.get("bad_evals", 0))
        order = list(payload["order"])
        cursor = int(payload["cursor"])
        replay_order = list(payload["replay_order"])
        replay_cursor = int(payload["replay_cursor"])
        input_tokens = int(payload.get("input_tokens", 0))
        answer_tokens = int(payload.get("answer_tokens", 0))
        replay_tokens = int(payload.get("replay_tokens", 0))
        rng.setstate(payload["rng_python"])
        torch.set_rng_state(payload["rng_torch"])
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
        if sorted(order) != list(range(len(train))) or sorted(replay_order) != list(range(len(replay_blocks))):
            raise ValueError("I2 resume sampling order incompatible with current data")
    else:
        source_tokens = int(payload.get("tokens", 0))
        start_step = 0
        selected_step = 0
        baseline_sft = evaluate_sft(model, val, device, args.seq)
        baseline_gen = evaluate_free_dev(model, dev_rows, device)
        baseline_lm = validate_document_bounded(model, replay_val, replay_val_schedule, args.seq, device)
        baseline_math = dev_math_nll(model, device, min(args.seq, 256))
        best_score = 0.0
        best_health = float(baseline_gen["health_score"])
        bad = 0
        order = list(range(len(train))); rng.shuffle(order); cursor = 0
        replay_order = list(range(len(replay_blocks))); rng.shuffle(replay_order); replay_cursor = 0
        input_tokens = answer_tokens = replay_tokens = 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(.9, .95), weight_decay=.03, fused=True)
    if resume:
        if "optimizer" not in payload:
            raise ValueError("I2 resume checkpoint missing optimizer")
        opt.load_state_dict(payload["optimizer"])

    run = {
        "stage": "I2 weakness repair E1",
        "stage_version": STAGE_VERSION,
        "parameters": EXPECTED_PARAMS,
        "source_checkpoint": str(Path(args.init_from).resolve()),
        "source_sha256": source_sha,
        "source_payload_tokens": source_tokens,
        "data_hashes": hashes,
        "replay_signature": replay_sig,
        "train_selection": selection_summary(train),
        "val_selection": selection_summary(val),
        "dev_selection": selection_summary(dev_rows),
        "replay_validation": replay_val_meta,
        "objective": {
            "assistant_only": True,
            "real_terminal_eos_supervision": True,
            "eos_weight": args.eos_weight,
            "sft_weight": args.sft_weight,
            "base_lm_replay_weight": 1 - args.sft_weight,
            "domain_profile": "40% math / 25% code / 10% reasoning / 10% concise / 15% general when capacity permits",
        },
        "selection": "held-out EOS-minus-repetition health plus math/code SFT and RawMath-dev gains, with base-LM/math regression guards",
        "decoding_hacks": {"repetition_penalty": False, "no_repeat_ngram": False, "forced_eos": False, "eos_logit_bias": False},
        "resume_safe": True,
        "solve_used": False,
        "tools_used": False,
        "external_model_used": False,
        "benchmark_policy": "Full unchanged model-only benchmark bundle after I2; no frozen benchmark is used for checkpoint selection.",
        "baseline_sft": baseline_sft,
        "baseline_free_dev": baseline_gen,
        "baseline_lm": baseline_lm,
        "baseline_math": baseline_math,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "args": vars(args),
    }
    if not resume:
        (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf8")

    step = start_step

    def checkpoint() -> dict:
        return {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "config": FIFTY.to_dict(), "tokens": source_tokens,
            "source_payload_tokens": source_tokens, "i2_stage": "weakness_e1", "i2_version": STAGE_VERSION,
            "i2_step": step, "i2_selected_step": selected_step, "best_weakness_score": best_score,
            "best_health_score": best_health, "bad_evals": bad, "source_sha256": source_sha,
            "sft_data_hashes": hashes, "replay_signature": replay_sig, "args_signature": args_signature(args),
            "baseline_sft": baseline_sft, "baseline_gen": baseline_gen, "baseline_lm": baseline_lm, "baseline_math": baseline_math,
            "order": order, "cursor": cursor, "replay_order": replay_order, "replay_cursor": replay_cursor,
            "rng_python": rng.getstate(), "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            "input_tokens": input_tokens, "answer_tokens": answer_tokens, "replay_tokens": replay_tokens,
        }

    if not resume:
        atomic_save(checkpoint(), out / "best_repair.pt")
        atomic_save(checkpoint(), latest)

    start = time.perf_counter()
    update_seconds = 0.0
    stop = "max_steps"
    interrupted = False
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, args.steps + 1):
            rows, cursor = next_rows(train, order, cursor, args.batch_size, rng)
            x, y = make_sft_batch(rows, args.seq, device)
            replay_items = []
            for _ in range(args.replay_batch):
                if replay_cursor >= len(replay_order):
                    rng.shuffle(replay_order); replay_cursor = 0
                replay_items.append(replay_blocks[replay_order[replay_cursor]])
                replay_cursor += 1
            lx, ly = make_batch(replay_streams, replay_items, args.seq, device)
            lr = lr_at(step, args.steps, args.peak_lr, args.min_lr, args.warmup_fraction)
            for group in opt.param_groups:
                group["lr"] = lr
            torch.cuda.synchronize(); tick = time.perf_counter(); opt.zero_grad(set_to_none=True)
            sft_loss = assistant_loss(model, x, y, args.eos_weight)
            base_loss = lm_loss(model, lx, ly)
            total = args.sft_weight * sft_loss + (1 - args.sft_weight) * base_loss
            if not torch.isfinite(total):
                stop = "nonfinite_loss"; raise FloatingPointError("non-finite I2 loss")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); torch.cuda.synchronize(); dt = time.perf_counter() - tick
            update_seconds += dt
            input_tokens += int((x != TOKENIZER.pad_id).sum())
            answer_tokens += int((y != -100).sum())
            replay_tokens += lx.numel()
            row = {
                "step": step, "loss": float(total), "sft_loss": float(sft_loss.detach()), "base_replay_loss": float(base_loss.detach()),
                "lr": lr, "grad_norm": float(grad_norm), "seconds": dt, "input_tokens": input_tokens,
                "answer_tokens": answer_tokens, "replay_tokens": replay_tokens,
            }
            if step % args.eval_every == 0 or step == args.steps:
                sft_val = evaluate_sft(model, val, device, args.seq)
                gen_val = evaluate_free_dev(model, dev_rows, device)
                lm_val = validate_document_bounded(model, replay_val, replay_val_schedule, args.seq, device)
                math_val = dev_math_nll(model, device, min(args.seq, 256))
                score = weakness_score(sft_val, gen_val, math_val, baseline_sft, baseline_gen, baseline_math)
                lm_ok = lm_val["macro"] <= baseline_lm["macro"] * (1 + args.max_lm_regression)
                math_ok = math_val["nll"] <= baseline_math["nll"] * (1 + args.max_math_regression)
                utf8_ok = gen_val["utf8_valid_pct"] >= 99.0
                length_ok = gen_val["mean_generated_tokens"] >= 5.0
                health_ok = gen_val["health_score"] >= baseline_gen["health_score"] + args.min_health_gain
                sft_ok = sft_val["macro_answer_nll"] <= baseline_sft["macro_answer_nll"] - args.min_sft_nll_gain
                math_slice_ok = domain_nll(sft_val, "math") <= domain_nll(baseline_sft, "math") * 1.005
                code_slice_ok = domain_nll(sft_val, "code") <= domain_nll(baseline_sft, "code") * 1.005
                improved = (lm_ok and math_ok and utf8_ok and length_ok and health_ok and sft_ok and math_slice_ok and code_slice_ok
                            and score >= best_score + args.min_score_gain)
                row.update(
                    sft_validation=sft_val, free_generation_dev=gen_val, base_validation=lm_val, math_dev=math_val,
                    weakness_score=score, lm_guard=lm_ok, math_guard=math_ok, utf8_guard=utf8_ok, length_guard=length_ok,
                    health_guard=health_ok, sft_guard=sft_ok, math_slice_guard=math_slice_ok, code_slice_guard=code_slice_ok,
                    selected=improved,
                )
                if improved:
                    best_score = score; best_health = float(gen_val["health_score"]); selected_step = step; bad = 0
                    atomic_save(checkpoint(), out / "best_repair.pt")
                else:
                    bad += 1
                row["bad_evals"] = bad
                atomic_save(checkpoint(), latest)
                print(json.dumps(row), flush=True)
                if bad >= args.patience:
                    stop = "early_stopping"; break
            with (out / "metrics.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(row) + "\n")
    except KeyboardInterrupt:
        stop = "interrupted"; interrupted = True
    except Exception:
        if stop == "max_steps": stop = "error"
        raise
    finally:
        atomic_save(checkpoint(), latest)
        summary = {
            "stage": "I2 weakness repair E1", "stop": stop, "complete": stop in {"max_steps", "early_stopping"},
            "completed_step": step, "selected_step": selected_step, "effective_weight_candidate": selected_step > 0,
            "source_payload_tokens": source_tokens, "best_weakness_score": best_score, "best_health_score": best_health,
            "baseline_health_score": baseline_gen["health_score"], "input_tokens": input_tokens, "answer_tokens": answer_tokens,
            "replay_tokens": replay_tokens, "optimizer_seconds_this_session": update_seconds,
            "elapsed_seconds_this_session": time.perf_counter() - start,
            "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
            "best_sha256": sha(out / "best_repair.pt"), "latest_sha256": sha(latest), "resume_safe": True,
            "solve_used": False, "tools_used": False, "external_model_used": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps(summary), flush=True)
    if interrupted:
        raise KeyboardInterrupt
    if stop not in {"max_steps", "early_stopping"}:
        raise RuntimeError(f"I2 stopped: {stop}")


if __name__ == "__main__":
    main()
