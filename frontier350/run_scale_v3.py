from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from recovery_identity import file_sha

TARGET_TOKENS_DEFAULT = 10_000_000_000
BLOCK_LADDER = (2_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000, 500_000_000)
PACK = 6
SEQ = 512
BATCH = 4
MICROBATCHES = 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROBATCHES
STATE_NAME = "frontier350_scale_v3_state.json"
RECIPES = (
    {"name": "balanced", "lr": 3.0e-5, "lr_end": 1.5e-5, "replay_boost": 1.35},
    {"name": "retention", "lr": 2.2e-5, "lr_end": 1.1e-5, "replay_boost": 1.55},
    {"name": "rescue", "lr": 1.6e-5, "lr_end": 8.0e-6, "replay_boost": 1.75},
)


def run(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def checkpoint_tokens(path: Path) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return int(ck["processed_tokens"])


def discover_anchor(assets: Path) -> tuple[Path, str, int]:
    state_path = assets / STATE_NAME
    if state_path.is_file():
        state = load_json(state_path)
        p = Path(state["safe_anchor_path"])
        if p.is_file() and file_sha(p) == state["safe_anchor_sha256"] and checkpoint_tokens(p) == int(state["safe_anchor_tokens"]):
            return p, state["safe_anchor_sha256"], int(state["safe_anchor_tokens"])
    candidates = []
    for p in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        if not p.is_file():
            continue
        try:
            tokens = checkpoint_tokens(p)
            candidates.append((tokens, p, file_sha(p)))
        except Exception:
            continue
    if not candidates:
        raise ValueError("no promoted Neral 350M anchor found")
    tokens, path, sha = max(candidates, key=lambda x: x[0])
    return path, sha, tokens


def train_rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj.get("event") == "train":
                out.append(obj)
    return out


def p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def strict_bilingual(base: dict, cand: dict, dev: dict) -> bool:
    for obj in (base, cand):
        if obj.get("cases") != 400 or set(obj.get("directions", {})) != {"de_to_en", "en_to_de"}:
            raise ValueError("invalid bilingual evidence")
    for key in ("cases_sha256", "tokenizer_sha256", "protocol_sha256"):
        if base.get(key) != cand.get(key):
            raise ValueError("bilingual protocol changed")
    for key in ("checkpoint_sha256", "checkpoint_processed_tokens", "tokenizer_sha256"):
        if cand.get(key) != dev.get(key):
            raise ValueError("mixed checkpoint evidence")
    return cand["overall_accuracy_pct"] >= base["overall_accuracy_pct"] and all(
        cand["directions"][d]["accuracy_pct"] >= base["directions"][d]["accuracy_pct"]
        for d in base["directions"]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    for n in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS_DEFAULT)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--max-wall-minutes", type=int, default=180)
    args = ap.parse_args()
    assets, report, scratch = map(lambda p: Path(p).resolve(), (args.assets, args.report, args.scratch))
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report not empty")
    if scratch.exists():
        raise FileExistsError("scratch already exists")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_root = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    main_data = main_root / "train"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    cases = main_root / "bilingual_holdout.json"
    state_path = assets / STATE_NAME
    started = time.monotonic()

    anchor, anchor_sha, anchor_tokens = discover_anchor(assets)
    persisted = load_json(state_path) if state_path.is_file() else {}
    block_index = int(persisted.get("block_index", 0))
    success_streak = int(persisted.get("success_streak", 0))
    conservative_level = int(persisted.get("conservative_level", 0))
    history = []

    def eval_core(checkpoint: Path, tag: str, tokens: int, baseline_tag: str | None = None) -> dict:
        val = report / f"{tag}_validation.json"
        dev = report / f"{tag}_dev.json"
        run("eval_validation_loss350.py", "--checkpoint", checkpoint, "--valid-data", valid, "--out", val,
            "--seq", 512, "--batch", 4, "--batches", 32, "--seed", 351337)
        run("eval_recovery_dev_v2.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer, "--out", dev,
            "--expected-tokens", tokens)
        out = {"validation": load_json(val), "dev": load_json(dev)}
        sha = file_sha(checkpoint)
        for obj in out.values():
            if obj.get("checkpoint_sha256") != sha or obj.get("checkpoint_processed_tokens") != tokens:
                raise ValueError("core evaluation identity mismatch")
        return out

    def eval_bilingual(checkpoint: Path, tag: str, tokens: int, dev: dict) -> dict:
        bi = report / f"{tag}_bilingual.json"
        run("eval_bilingual_holdout_v1.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer,
            "--cases", cases, "--out", bi, "--expected-tokens", tokens)
        obj = load_json(bi)
        if obj.get("checkpoint_sha256") != file_sha(checkpoint) or obj.get("checkpoint_processed_tokens") != tokens:
            raise ValueError("bilingual evaluation identity mismatch")
        if obj.get("tokenizer_sha256") != dev.get("tokenizer_sha256"):
            raise ValueError("tokenizer identity mismatch")
        return obj

    def baseline_for_current_anchor() -> dict:
        cached = persisted.get("baseline")
        if cached and cached.get("checkpoint_sha256") == anchor_sha and int(cached.get("checkpoint_processed_tokens", -1)) == anchor_tokens:
            return cached["metrics"]
        tag = f"baseline_{anchor_tokens}"
        core = eval_core(anchor, tag, anchor_tokens)
        bi = eval_bilingual(anchor, tag, anchor_tokens, core["dev"])
        return {**core, "bilingual": bi}

    baseline = baseline_for_current_anchor()

    def train_block(block_tokens: int, recipe: dict, round_tag: str) -> tuple[Path, Path | None, dict]:
        out = scratch / round_tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(anchor), "--anchor-sha", anchor_sha,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", "replay_only", "--new-tokens", str(block_tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",
               "--lr", str(recipe["lr"]), "--lr-end", str(recipe["lr_end"]),
               "--replay-boost", str(recipe["replay_boost"]), "--snapshot-fraction", "0.5", "--fast-replay-only"]
        print(json.dumps({"event": "train_start", "round": round_tag, "block_tokens": block_tokens, "recipe": recipe}), flush=True)
        subprocess.run(cmd, check=True)
        summary = load_json(out / "summary.json")
        if not summary.get("complete") or not summary.get("frozen_weights_unchanged"):
            raise ValueError("training contract failed")
        rows = [r for r in train_rows(out / "metrics.jsonl")[5:] if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
        rates = [float(r["tokens_per_second"]) for r in rows]
        throughput = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates),
                      "p10_tps": p10(rates), "peak_tps": max(rates)} if rates else {}
        return out / "resume.pt", (out / "midpoint.pt" if (out / "midpoint.pt").is_file() else None), throughput

    def evaluate_candidate(raw: Path, raw_tokens: int, alpha: float, tag: str, recipe: dict, throughput: dict) -> dict:
        snap = scratch / f"{tag}.pt"
        run("interpolate_recovery_checkpoint.py", "--anchor", anchor, "--candidate", raw, "--out", snap, "--alpha", alpha)
        core = eval_core(snap, tag, raw_tokens)
        gate = report / f"{tag}_legacy_gate.json"
        run("select_recovery_lr_pilot.py", "--baseline-dev", report / f"baseline_{anchor_tokens}_dev.json",
            "--baseline-validation", report / f"baseline_{anchor_tokens}_validation.json", "--candidate", tag, recipe["lr"],
            report / f"{tag}_dev.json", report / f"{tag}_validation.json", "--out", gate)
        legacy = load_json(gate)
        legacy_ok = bool(legacy.get("selection_allowed"))
        bi = None
        bi_ok = False
        if legacy_ok:
            bi = eval_bilingual(snap, tag, raw_tokens, core["dev"])
            bi_ok = strict_bilingual(baseline["bilingual"], bi, core["dev"])
        improved = core["validation"]["validation_loss"] < baseline["validation"]["validation_loss"] or any(
            core["dev"]["categories"][k]["accuracy_pct"] > baseline["dev"]["categories"][k]["accuracy_pct"]
            for k in baseline["dev"]["categories"])
        if bi is not None:
            improved = improved or bi["overall_accuracy_pct"] > baseline["bilingual"]["overall_accuracy_pct"]
        passed = legacy_ok and bi_ok and improved
        return {"pass": passed, "alpha": alpha, "effective_lr": recipe["lr"] * alpha,
                "legacy_gate": legacy_ok, "bilingual_gate": bi_ok if bi is not None else None,
                "strict_improvement": improved, "validation_loss": core["validation"]["validation_loss"],
                "overall_pct": core["dev"]["overall_accuracy_pct"],
                "categories": {k: v["accuracy_pct"] for k, v in core["dev"]["categories"].items()},
                "bilingual_pct": bi["overall_accuracy_pct"] if bi else None,
                "de_to_en_pct": bi["directions"]["de_to_en"]["accuracy_pct"] if bi else None,
                "en_to_de_pct": bi["directions"]["en_to_de"]["accuracy_pct"] if bi else None,
                "checkpoint_sha256": file_sha(snap), "path": str(snap), "tokens": raw_tokens,
                "recipe": recipe, "throughput": throughput,
                "metrics": {**core, "bilingual": bi} if bi is not None else None}

    def adaptive_line_search(raw: Path, recipe: dict, round_tag: str, source_name: str, throughput: dict) -> tuple[dict | None, list[dict]]:
        raw_tokens = checkpoint_tokens(raw)
        alpha = 0.025
        seen = set()
        attempts = []
        for attempt in range(3):
            alpha = max(0.00625, min(0.08, alpha))
            key = round(alpha, 8)
            if key in seen:
                break
            seen.add(key)
            tag = f"{round_tag}_{source_name}_a{str(alpha).replace('.', 'p')}"
            result = evaluate_candidate(raw, raw_tokens, alpha, tag, recipe, throughput)
            attempts.append(result)
            print(json.dumps({"event": "candidate", **{k:v for k,v in result.items() if k not in ('metrics',)}}), flush=True)
            if result["pass"]:
                return result, attempts
            # Regression -> smaller trust step. Safe but no measurable gain -> larger trust step.
            if not result["legacy_gate"] or result["bilingual_gate"] is False:
                alpha *= 0.5
            elif not result["strict_improvement"]:
                alpha *= 1.6
            else:
                alpha *= 0.5
        return None, attempts

    round_no = 0
    while anchor_tokens < args.target_tokens and round_no < args.max_rounds:
        if (time.monotonic() - started) / 60 >= args.max_wall_minutes:
            break
        round_no += 1
        remaining = args.target_tokens - anchor_tokens
        block_tokens = min(BLOCK_LADDER[min(block_index, len(BLOCK_LADDER)-1)], remaining)
        if block_tokens < 100_000:
            break
        recipe_start = min(conservative_level, len(RECIPES)-1)
        selected = None
        round_attempts = []

        # At most two training recipes per round. Each run yields both a full and midpoint
        # checkpoint, so the shorter fallback is free: no second compilation or retraining.
        for recipe_index in range(recipe_start, min(recipe_start + 2, len(RECIPES))):
            recipe = RECIPES[recipe_index]
            tag = f"r{round_no:02d}_{recipe['name']}_{block_tokens}"
            raw, midpoint, throughput = train_block(block_tokens, recipe, tag)
            winner, attempts = adaptive_line_search(raw, recipe, tag, "full", throughput)
            round_attempts.extend(attempts)
            if winner is None and midpoint is not None:
                winner, attempts = adaptive_line_search(midpoint, recipe, tag, "mid", throughput)
                round_attempts.extend(attempts)
            if winner is not None:
                selected = winner
                conservative_level = max(0, recipe_index - 1)
                break

        if selected is None:
            success_streak = 0
            conservative_level = min(len(RECIPES)-1, conservative_level + 1)
            block_index = max(0, block_index - 1)
            history.append({"round": round_no, "anchor_tokens": anchor_tokens, "block_tokens": block_tokens,
                            "success": False, "conservative_level": conservative_level, "attempts": round_attempts})
            break

        destdir = assets / "runs" / f"neral350_scalev3_{args.request_id}_r{round_no:02d}_selected"
        if destdir.exists():
            raise FileExistsError("promotion destination exists")
        destdir.mkdir(parents=True)
        dest = destdir / "resume.pt"
        shutil.copy2(selected["path"], dest)
        if file_sha(dest) != selected["checkpoint_sha256"]:
            raise RuntimeError("promotion hash mismatch")

        old_tokens = anchor_tokens
        anchor, anchor_sha, anchor_tokens = dest, selected["checkpoint_sha256"], int(selected["tokens"])
        baseline = selected["metrics"]
        if baseline is None:
            raise RuntimeError("selected candidate lacks full evidence")
        gained = anchor_tokens - old_tokens
        full_block = gained >= int(block_tokens * 0.90)
        success_streak = success_streak + 1 if full_block else 0
        if success_streak >= 2 and block_index < len(BLOCK_LADDER)-1:
            block_index += 1
            success_streak = 0

        state = {"schema": 3, "safe_anchor_path": str(anchor), "safe_anchor_sha256": anchor_sha,
                 "safe_anchor_tokens": anchor_tokens, "target_tokens": args.target_tokens,
                 "block_index": block_index, "success_streak": success_streak,
                 "conservative_level": conservative_level,
                 "baseline": {"checkpoint_sha256": anchor_sha, "checkpoint_processed_tokens": anchor_tokens, "metrics": baseline},
                 "updated_by_request": args.request_id, "updated_round": round_no}
        write_json(state_path, state)
        history.append({"round": round_no, "anchor_tokens_before": old_tokens, "anchor_tokens_after": anchor_tokens,
                        "block_tokens": block_tokens, "gained_tokens": gained, "success": True,
                        "full_block": full_block, "selected": {k:v for k,v in selected.items() if k != "metrics"},
                        "next_block_index": block_index, "conservative_level": conservative_level,
                        "attempts": [{k:v for k,v in x.items() if k != "metrics"} for x in round_attempts]})
        write_json(report / "progress.json", history)

    result = {"schema": 3, "controller": "neral350-scale-v3-adaptive", "request_id": args.request_id,
              "target_tokens": args.target_tokens, "final_safe_tokens": anchor_tokens,
              "final_safe_sha256": anchor_sha, "final_safe_checkpoint": str(anchor),
              "rounds": history, "block_ladder": list(BLOCK_LADDER), "recipes": list(RECIPES),
              "safe_anchor_preserved_on_failure": True, "benchmarks_used_for_training": False,
              "full_promotion_gate_unchanged": True, "final_audit_pending": True,
              "wall_minutes": (time.monotonic()-started)/60}
    write_json(report / "selection.json", result)
    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
