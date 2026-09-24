"""72-hour Helios chat-first SFT trainer for the OWN Neral 350M weights.

Goal: obtain a usable DE/EN chatbot quickly, not finish foundation pretraining.
Trains the upper transformer blocks + final norm with FP32 master weights and BF16
compute. Uses assistant-only next-token loss, explicit EOS weighting and optional
replay from data the historical model already saw to reduce catastrophic forgetting.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPECTED_BASE_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
EXPECTED_TOKENIZER_SHA = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
PARAMETERS = 349_825_152
SEED = 968_720
MAX_RESERVED = 10_750 * 1024**2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"invalid JSONL row in {path}")
                rows.append(row)
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def atomic_save(torch, state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="resume exactly from out/resume.pt; data and hyperparameters must match")
    p.add_argument("--replay-root", type=Path)
    p.add_argument("--train-blocks", type=int, choices=(4, 6, 8), default=8)
    p.add_argument("--steps", type=int, default=2000,
                   help="fixed trajectory target; keep unchanged across resume invocations")
    p.add_argument("--stop-after-step", type=int, default=0,
                   help="optional staged pause <= steps; does not change optimizer trajectory")
    p.add_argument("--batch", type=int, choices=(2, 4), default=4)
    p.add_argument("--context", type=int, choices=(256, 384, 512), default=512)
    p.add_argument("--peak-lr", type=float, default=1.5e-5)
    p.add_argument("--min-lr", type=float, default=1.5e-6)
    p.add_argument("--sft-weight", type=float, default=0.85)
    p.add_argument("--eos-weight", type=float, default=1.75)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.005)
    args = p.parse_args()

    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit own-checkpoint opt-in required")
    if not 200 <= args.steps <= 5000:
        p.error("steps must be 200..5000")
    invocation_stop = args.stop_after_step or args.steps
    if not 1 <= invocation_stop <= args.steps:
        p.error("stop-after-step must be 1..steps or zero for the full trajectory")
    if not 0.70 <= args.sft_weight <= 1.0:
        p.error("sft-weight must be 0.70..1.0")
    if not 1.0 <= args.eos_weight <= 3.0:
        p.error("eos-weight must be 1..3")
    if not 0 < args.min_lr <= args.peak_lr <= 5e-5:
        p.error("invalid learning rate")
    if min(args.eval_every, args.patience) <= 0:
        p.error("positive eval interval/patience required")
    if invocation_stop < args.steps and invocation_stop % args.eval_every:
        p.error("intermediate stop-after-step must align to eval-every")
    resume_path = args.out / "resume.pt"
    if args.resume:
        if not args.out.is_dir() or not resume_path.is_file():
            p.error("--resume requires an existing out/resume.pt")
    elif args.out.exists():
        p.error("fresh sprint output must be create-only")
    if sha256_file(args.checkpoint) != EXPECTED_BASE_SHA:
        raise ValueError("historical 2.073B source SHA mismatch")
    if sha256_file(args.tokenizer) != EXPECTED_TOKENIZER_SHA:
        raise ValueError("Neral tokenizer SHA mismatch")

    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    data_identity = {
        "manifest_sha256": sha256_file(manifest_path),
        "train_sha256": sha256_file(args.data / "train.jsonl"),
        "validation_sha256": sha256_file(args.data / "val.jsonl"),
    }
    if manifest.get("external_model_calls") is not False:
        raise ValueError("SFT dataset must contain no external-model calls")
    if manifest.get("prompt_group_overlap_audit", {}).get("cross_split_prompt_group_overlap") != 0:
        raise ValueError("SFT split leakage")
    if manifest.get("target_train_examples", 0) < 10_000:
        raise ValueError("sprint SFT pack is unexpectedly small")
    train_raw = read_jsonl(args.data / "train.jsonl")
    val_raw = read_jsonl(args.data / "val.jsonl")

    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from model350 import FRONTIER_350, Neral350M
    from phoenix_cuda_diagnostic import cooldown_until_safe, require_cuda_ready, require_stably_idle_gpu
    from phoenix_master_precision import require_fp32_trainable_subset
    from train350_compiled import PersistentMMapPool

    require_stably_idle_gpu()
    require_cuda_ready(torch)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None or bos == eos:
        raise ValueError("BOS/EOS contract missing")

    def encode(row: dict) -> dict | None:
        prompt, answer = row.get("prompt"), row.get("answer")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError("SFT rows require prompt/answer")
        prefix = [bos] + tokenizer.encode(f"User: {prompt.strip()}\nAssistant:").ids
        answer_ids = tokenizer.encode(" " + answer.strip()).ids
        if not answer_ids or len(prefix) + len(answer_ids) > args.context:
            return None
        x = prefix + answer_ids
        # At the token immediately before the answer, predict answer token 1.
        y = [-100] * (len(prefix) - 1) + answer_ids + [eos]
        if len(x) != len(y):
            raise AssertionError("causal label alignment failed")
        return {"id": str(row.get("id")), "x": x, "y": y,
                "language": row.get("language"), "domain": row.get("domain")}

    train = [x for r in train_raw if (x := encode(r)) is not None]
    valid = [x for r in val_raw if (x := encode(r)) is not None]
    if len(train) < 10_000 or len(valid) < 100:
        raise ValueError(f"too few tokenized sprint rows: train={len(train)} valid={len(valid)}")

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    base = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if base.get("config") != FRONTIER_350.to_dict():
        raise ValueError("base architecture changed")
    if base.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA:
        raise ValueError("base tokenizer identity changed")
    historical_tokens = int(base.get("processed_tokens", -1))
    if historical_tokens != 2_073_161_216:
        raise ValueError("unexpected historical token counter")

    model = Neral350M().to(device=device, dtype=torch.float32)
    model.load_state_dict(base["model"], strict=True)
    del base
    if model.parameter_report()["total"] != PARAMETERS:
        raise ValueError("350M architecture changed")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks[-args.train_blocks:]:
        block.float()
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    model.norm.float()
    for parameter in model.norm.parameters():
        parameter.requires_grad_(True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    trainable_parameters = sum(p.numel() for p in params)
    trainable_names = [n for n, _ in named]
    args_signature = {
        "train_blocks": args.train_blocks,
        "steps": args.steps,
        "batch": args.batch,
        "context": args.context,
        "peak_lr": args.peak_lr,
        "min_lr": args.min_lr,
        "sft_weight": args.sft_weight,
        "eos_weight": args.eos_weight,
        "eval_every": args.eval_every,
        "patience": args.patience,
        "min_delta": args.min_delta,
    }

    optimizer = torch.optim.AdamW(
        params, lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.01, fused=True,
    )
    require_fp32_trainable_subset(model, optimizer, torch, require_moments=False)

    replay = None
    replay_weight = max(0.0, 1.0 - args.sft_weight)
    replay_rng = random.Random(SEED + 1)
    replay_identity = None
    if replay_weight > 0:
        if args.replay_root is None:
            raise ValueError("replay-root required when sft-weight < 1")
        replay_root = args.replay_root.resolve(strict=True)
        replay_manifest = replay_root.parent / "manifest.json"
        replay_identity = {
            "root": str(replay_root),
            "manifest_sha256": sha256_file(replay_manifest) if replay_manifest.is_file() else None,
        }
        replay = PersistentMMapPool(replay_root, FRONTIER_350.vocab_size)

    order = list(range(len(train)))
    sampler = random.Random(SEED + 2)
    sampler.shuffle(order)
    cursor = 0

    def make_batch(rows: list[dict], indices: list[int]):
        selected = [rows[i] for i in indices]
        width = max(len(r["x"]) for r in selected)
        x = torch.full((len(selected), width), eos, dtype=torch.long, device=device)
        y = torch.full((len(selected), width), -100, dtype=torch.long, device=device)
        for i, r in enumerate(selected):
            n = len(r["x"])
            x[i, :n] = torch.tensor(r["x"], dtype=torch.long, device=device)
            y[i, :n] = torch.tensor(r["y"], dtype=torch.long, device=device)
        return x, y

    def next_indices(n: int) -> list[int]:
        nonlocal cursor, order
        out = []
        for _ in range(n):
            if cursor >= len(order):
                sampler.shuffle(order)
                cursor = 0
            out.append(order[cursor])
            cursor += 1
        return out

    def assistant_loss(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_y = y.reshape(-1)
        per = F.cross_entropy(flat_logits, flat_y, ignore_index=-100, reduction="none")
        mask = flat_y != -100
        weights = torch.ones_like(per)
        weights[(flat_y == eos) & mask] = args.eos_weight
        return (per[mask] * weights[mask]).sum() / weights[mask].sum().clamp_min(1.0)

    # Fixed validation spread; never use the holdout split for training decisions.
    val_count = min(128, len(valid))
    val_indices = [round(i * (len(valid) - 1) / max(1, val_count - 1)) for i in range(val_count)]

    def evaluate() -> float:
        model.eval()
        total = 0.0
        tokens = 0
        with torch.no_grad():
            for start in range(0, len(val_indices), 4):
                inds = val_indices[start:start + 4]
                x, y = make_batch(valid, inds)
                loss = assistant_loss(x, y)
                n = int((y != -100).sum().item())
                total += float(loss) * n
                tokens += n
        model.train()
        return total / max(1, tokens)

    metrics = args.out / "metrics.jsonl"
    start_step = 0
    supervised_tokens = 0
    replay_tokens = 0
    bad = 0
    best_step = 0

    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            state.get("checkpoint_schema") != "helios_chat_sprint72h_resume_v2"
            or state.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA
            or state.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA
            or state.get("data_identity") != data_identity
            or state.get("args_signature") != args_signature
            or state.get("replay_identity") != replay_identity
            or state.get("trainable_names") != trainable_names
            or state.get("production_eligible") is not False
        ):
            raise ValueError("resume checkpoint contract/data/hyperparameters changed")
        start_step = int(state.get("sft_step", -1))
        if not 0 < start_step < args.steps:
            raise ValueError("resume step is outside the active sprint")
        if start_step >= invocation_stop:
            raise ValueError("resume is already at or beyond this staged stop")
        saved = state.get("trainable_model")
        if not isinstance(saved, dict) or set(saved) != set(trainable_names):
            raise ValueError("resume trainable-weight set changed")
        by_name = dict(model.named_parameters())
        with torch.no_grad():
            for name in trainable_names:
                value = saved[name]
                if not torch.is_tensor(value) or value.shape != by_name[name].shape:
                    raise ValueError(f"resume tensor mismatch: {name}")
                by_name[name].copy_(value.to(device=device, dtype=torch.float32))
        optimizer.load_state_dict(state["optimizer"])
        require_fp32_trainable_subset(model, optimizer, torch, require_moments=True)
        order = list(state["sampler_order"])
        cursor = int(state["sampler_cursor"])
        if sorted(order) != list(range(len(train))) or not 0 <= cursor <= len(order):
            raise ValueError("resume sampler does not match current training set")
        sampler.setstate(state["sampler_rng"])
        replay_rng.setstate(state["replay_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        baseline_val = float(state["baseline_validation_loss"])
        best_val = float(state["best_validation_loss"])
        bad = int(state["bad_evals"])
        best_step = int(state["best_step"])
        supervised_tokens = int(state["supervised_answer_tokens"])
        replay_tokens = int(state["replay_tokens"])
        if not all(math.isfinite(v) for v in (baseline_val, best_val)):
            raise ValueError("nonfinite resume validation state")
        with metrics.open("a", encoding="utf8") as log:
            log.write(json.dumps({"event": "resume", "from_step": start_step,
                                  "best_step": best_step,
                                  "best_validation_loss": best_val}) + "\n")
        del state
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        baseline_val = evaluate()
        best_val = baseline_val

    def lr_at(step: int) -> float:
        warm = max(1, int(args.steps * 0.05))
        if step <= warm:
            return args.peak_lr * step / warm
        q = (step - warm) / max(1, args.steps - warm)
        return args.min_lr + 0.5 * (args.peak_lr - args.min_lr) * (1 + math.cos(math.pi * q))

    def checkpoint(step: int, val: float, kind: str) -> dict:
        return {
            "checkpoint_schema": "helios_chat_sprint72h_v1",
            "classification": kind,
            "model": {n: t.detach().cpu().float() for n, t in model.state_dict().items()},
            "config": FRONTIER_350.to_dict(),
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
            "base_checkpoint_sha256": EXPECTED_BASE_SHA,
            "processed_tokens": historical_tokens,
            "sft_step": step,
            "sft_validation_loss": val,
            "train_blocks": args.train_blocks,
            "trainable_parameters": trainable_parameters,
            "data_identity": data_identity,
            "args_signature": args_signature,
            "replay_identity": replay_identity,
            "master_parameter_dtype": "float32",
            "compute_autocast_dtype": "bfloat16",
            "production_eligible": False,
        }

    started = time.monotonic()
    model.train()
    stop_reason = "max_steps"
    step = start_step
    for step in range(start_step + 1, invocation_stop + 1):
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = make_batch(train, next_indices(args.batch))
        optimizer.zero_grad(set_to_none=True)
        sft = assistant_loss(x, y)
        if not bool(torch.isfinite(sft)):
            raise FloatingPointError("nonfinite SFT loss")
        total = sft * args.sft_weight

        replay_loss_value = None
        if replay is not None and replay_weight > 0:
            rx, ry = replay.batch(1, 256, device, replay_rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, replay_loss = model(rx, ry)
            if replay_loss is None or not bool(torch.isfinite(replay_loss)):
                raise FloatingPointError("nonfinite replay loss")
            total = total + replay_loss * replay_weight
            replay_loss_value = float(replay_loss.detach())
            replay_tokens += int(rx.numel())

        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1:
            require_fp32_trainable_subset(model, optimizer, torch, require_moments=True)
        supervised_tokens += int((y != -100).sum().item())

        if torch.cuda.memory_reserved() > MAX_RESERVED:
            raise RuntimeError("sprint SFT exceeded 10.75 GiB reserved VRAM")
        if step % 25 == 0:
            torch.cuda.synchronize()
            thermal = cooldown_until_safe()
        else:
            thermal = {"cooled": False}

        row = {
            "step": step, "lr": lr, "sft_loss": float(sft.detach()),
            "replay_loss": replay_loss_value, "combined_loss": float(total.detach()),
            "grad_norm": float(grad), "supervised_answer_tokens": supervised_tokens,
            "replay_tokens": replay_tokens,
        }
        if thermal.get("cooled"):
            row["thermal_cooldown"] = thermal

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate()
            improved = val < best_val - args.min_delta
            row.update({"validation_loss": val, "best_validation_loss": best_val,
                        "selected": improved})
            if improved:
                best_val = val
                best_step = step
                bad = 0
                atomic_save(torch, checkpoint(step, val, "best_chat_sprint_candidate"),
                            args.out / "best_chat.pt")
            else:
                bad += 1
            # Compact exact resume: lower frozen weights are reconstructed from the
            # immutable SHA-pinned base instead of being duplicated on every snapshot.
            resume = {
                "checkpoint_schema": "helios_chat_sprint72h_resume_v2",
                "classification": "chat_sprint_resume",
                "base_checkpoint_sha256": EXPECTED_BASE_SHA,
                "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
                "data_identity": data_identity,
                "args_signature": args_signature,
                "replay_identity": replay_identity,
                "trainable_names": trainable_names,
                "trainable_model": {
                    name: param.detach().cpu().float() for name, param in named
                },
                "optimizer": optimizer.state_dict(),
                "sft_step": step,
                "baseline_validation_loss": baseline_val,
                "best_validation_loss": best_val,
                "best_step": best_step,
                "bad_evals": bad,
                "supervised_answer_tokens": supervised_tokens,
                "replay_tokens": replay_tokens,
                "sampler_order": order,
                "sampler_cursor": cursor,
                "sampler_rng": sampler.getstate(),
                "replay_rng": replay_rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()],
                "production_eligible": False,
            }
            atomic_save(torch, resume, args.out / "resume.next.pt")
            os.replace(args.out / "resume.next.pt", resume_path)
            if bad >= args.patience and best_step > 0:
                stop_reason = "early_stopping"
                with metrics.open("a", encoding="utf8") as f:
                    f.write(json.dumps(row) + "\n")
                break

        with metrics.open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")

    elapsed = time.monotonic() - started
    if best_step == 0:
        raise RuntimeError("SFT validation never improved; do not promote a chat candidate")
    if stop_reason == "max_steps" and step < args.steps:
        stop_reason = "stage_boundary"
    training_complete = stop_reason == "early_stopping" or step >= args.steps
    best_path = args.out / "best_chat.pt"
    summary = {
        "event": "helios_chat_sprint72h_finished",
        "base_checkpoint_sha256": EXPECTED_BASE_SHA,
        "best_checkpoint_sha256": sha256_file(best_path),
        "baseline_validation_loss": baseline_val,
        "best_validation_loss": best_val,
        "best_step": best_step,
        "resumed_from_step": start_step,
        "completed_step": step,
        "stop_reason": stop_reason,
        "training_complete": training_complete,
        "trajectory_target_steps": args.steps,
        "invocation_stop_step": invocation_stop,
        "train_blocks": args.train_blocks,
        "trainable_parameters": trainable_parameters,
        "supervised_answer_tokens": supervised_tokens,
        "foundation_replay_tokens": replay_tokens,
        "training_wall_seconds": elapsed,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "productive_foundation_tokens": 0,
        "quality_verified": False,
        "next_gate": "free-answer + multi-turn chat evaluation before release",
    }
    (args.out / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    if training_complete:
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
