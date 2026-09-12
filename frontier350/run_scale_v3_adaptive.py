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

from recovery_identity import file_sha, validate_pair

TARGET_DEFAULT = 10_000_000_000
BLOCK_LADDER = (2_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000, 500_000_000)
PACK = 6
SEQ = 512
BATCH = 4
MICROBATCHES = 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROBATCHES
MIN_FINAL_REPLAY = 3 * SEQ * BATCH
STATE_NAME = "frontier350_scale_v3_state.json"
BASE_RECIPES = (
    {"name": "balanced", "lr": 3.0e-5, "lr_end": 1.5e-5, "replay_boost": 1.35},
    {"name": "retention", "lr": 2.2e-5, "lr_end": 1.1e-5, "replay_boost": 1.55},
    {"name": "rescue", "lr": 1.6e-5, "lr_end": 8.0e-6, "replay_boost": 1.75},
)
EXPECTED_CATEGORIES = {"bilingual", "code", "factual", "language", "math", "reasoning", "science"}


def run(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def checkpoint_tokens(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["processed_tokens"])


def discover_anchor(assets: Path) -> tuple[Path, str, int, dict]:
    state_path = assets / STATE_NAME
    if state_path.is_file():
        state = read_json(state_path)
        p = Path(state["safe_anchor_path"])
        if p.is_file() and file_sha(p) == state["safe_anchor_sha256"] and checkpoint_tokens(p) == int(state["safe_anchor_tokens"]):
            return p, state["safe_anchor_sha256"], int(state["safe_anchor_tokens"]), state
    found = []
    for p in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        if not p.is_file():
            continue
        try:
            found.append((checkpoint_tokens(p), p, file_sha(p)))
        except Exception:
            pass
    if not found:
        raise RuntimeError("no promoted Neral 350M checkpoint found")
    tokens, path, sha = max(found, key=lambda x: x[0])
    return path, sha, tokens, {}


def p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def train_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event") == "train":
            rows.append(obj)
    return rows


def normalized_budget(value: int) -> int:
    if value <= 0:
        return 0
    rem = value % TOKENS_PER_UPDATE
    if 0 < rem < MIN_FINAL_REPLAY:
        value -= rem
    return value


def scaled_recipe(base: dict, block_tokens: int) -> dict:
    # Larger horizons get a lower LR and slightly stronger replay retention.
    factor = max(0.25, min(1.0, (2_000_000 / max(block_tokens, 2_000_000)) ** 0.25))
    growth = max(0.0, math.log10(max(1.0, block_tokens / 2_000_000)))
    return {
        "name": base["name"],
        "lr": base["lr"] * factor,
        "lr_end": base["lr_end"] * factor,
        "replay_boost": min(2.0, base["replay_boost"] + 0.08 * growth),
        "scale_factor": factor,
    }


def verify_dev(report: dict) -> None:
    if report.get("suite") != "arctus-recovery-dev-v1" or int(report.get("cases", 0)) != 140:
        raise ValueError("wrong development suite")
    cats = report.get("categories") or {}
    if set(cats) != EXPECTED_CATEGORIES:
        raise ValueError("unexpected development categories")
    total = correct = 0
    for name, row in cats.items():
        n, k = row.get("total"), row.get("correct")
        if type(n) is not int or type(k) is not int or n != 20 or not 0 <= k <= n:
            raise ValueError(f"invalid category counts: {name}")
        if abs(float(row["accuracy_pct"]) - 100 * k / n) > 1e-9:
            raise ValueError("category percentage mismatch")
        total += n
        correct += k
    if total != 140 or abs(float(report["overall_accuracy_pct"]) - 100 * correct / total) > 1e-9:
        raise ValueError("overall percentage mismatch")


def legacy_gate(base: dict, cand: dict) -> tuple[bool, list[str]]:
    base_dev, base_val = base["dev"], base["validation"]
    dev, val = cand["dev"], cand["validation"]
    verify_dev(base_dev)
    verify_dev(dev)
    validate_pair(base_dev, base_val, dev, val)
    failures = []
    if float(dev["overall_accuracy_pct"]) < float(base_dev["overall_accuracy_pct"]):
        failures.append("overall")
    for cat in sorted(EXPECTED_CATEGORIES):
        if float(dev["categories"][cat]["accuracy_pct"]) < float(base_dev["categories"][cat]["accuracy_pct"]):
            failures.append(cat)
    if float(val["validation_loss"]) > float(base_val["validation_loss"]):
        failures.append("validation")
    return not failures, failures


def bilingual_gate(base: dict, cand: dict, dev: dict) -> bool:
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
    for name in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--target-tokens", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--max-wall-minutes", type=int, default=180)
    a = ap.parse_args()
    assets, report, scratch = map(lambda p: Path(p).resolve(), (a.assets, a.report, a.scratch))
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

    anchor, anchor_sha, anchor_tokens, state = discover_anchor(assets)
    block_index = int(state.get("block_index", 0))
    success_streak = int(state.get("success_streak", 0))
    conservative_level = int(state.get("conservative_level", 0))
    history: list[dict] = []

    def eval_core(checkpoint: Path, tag: str, tokens: int) -> dict:
        val_path = report / f"{tag}_validation.json"
        dev_path = report / f"{tag}_dev.json"
        run("eval_validation_loss350.py", "--checkpoint", checkpoint, "--valid-data", valid, "--out", val_path,
            "--seq", 512, "--batch", 4, "--batches", 32, "--seed", 351337)
        run("eval_recovery_dev_v2.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer, "--out", dev_path,
            "--expected-tokens", tokens)
        out = {"validation": read_json(val_path), "dev": read_json(dev_path)}
        sha = file_sha(checkpoint)
        for obj in out.values():
            if obj.get("checkpoint_sha256") != sha or int(obj.get("checkpoint_processed_tokens", -1)) != tokens:
                raise ValueError("core evaluation identity mismatch")
        return out

    def eval_bi(checkpoint: Path, tag: str, tokens: int, dev: dict) -> dict:
        path = report / f"{tag}_bilingual.json"
        run("eval_bilingual_holdout_v1.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer,
            "--cases", cases, "--out", path, "--expected-tokens", tokens)
        obj = read_json(path)
        if obj.get("checkpoint_sha256") != file_sha(checkpoint) or int(obj.get("checkpoint_processed_tokens", -1)) != tokens:
            raise ValueError("bilingual evaluation identity mismatch")
        if obj.get("tokenizer_sha256") != dev.get("tokenizer_sha256"):
            raise ValueError("tokenizer identity mismatch")
        return obj

    cached = state.get("baseline")
    if cached and cached.get("checkpoint_sha256") == anchor_sha and int(cached.get("checkpoint_processed_tokens", -1)) == anchor_tokens:
        baseline = cached["metrics"]
    else:
        tag = f"baseline_{anchor_tokens}"
        core = eval_core(anchor, tag, anchor_tokens)
        baseline = {**core, "bilingual": eval_bi(anchor, tag, anchor_tokens, core["dev"])}

    def train_block(block_tokens: int, recipe: dict, tag: str) -> tuple[Path, Path | None, dict]:
        out = scratch / tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(anchor), "--anchor-sha", anchor_sha,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", "replay_only", "--new-tokens", str(block_tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",
               "--lr", str(recipe["lr"]), "--lr-end", str(recipe["lr_end"]),
               "--replay-boost", str(recipe["replay_boost"]), "--snapshot-fraction", "0.5", "--fast-replay-only"]
        print(json.dumps({"event": "train_start", "tag": tag, "block_tokens": block_tokens, "recipe": recipe}), flush=True)
        subprocess.run(cmd, check=True)
        summary = read_json(out / "summary.json")
        if not summary.get("complete") or not summary.get("frozen_weights_unchanged"):
            raise RuntimeError("training contract failed")
        rates = [float(r["tokens_per_second"]) for r in train_rows(out / "metrics.jsonl")[5:]
                 if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
        throughput = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates),
                      "p10_tps": p10(rates), "peak_tps": max(rates)} if rates else {}
        midpoint = out / "midpoint.pt"
        return out / "resume.pt", midpoint if midpoint.is_file() else None, throughput

    def evaluate(raw: Path, recipe: dict, tag: str, alpha: float, throughput: dict) -> dict:
        tokens = checkpoint_tokens(raw)
        snap = scratch / f"{tag}.pt"
        run("interpolate_recovery_checkpoint.py", "--anchor", anchor, "--candidate", raw, "--out", snap, "--alpha", alpha)
        core = eval_core(snap, tag, tokens)
        core_ok, failures = legacy_gate(baseline, core)
        bi = None
        bi_ok = False
        if core_ok:
            bi = eval_bi(snap, tag, tokens, core["dev"])
            bi_ok = bilingual_gate(baseline["bilingual"], bi, core["dev"])
        improved = float(core["validation"]["validation_loss"]) < float(baseline["validation"]["validation_loss"]) or any(
            float(core["dev"]["categories"][k]["accuracy_pct"]) > float(baseline["dev"]["categories"][k]["accuracy_pct"])
            for k in EXPECTED_CATEGORIES)
        if bi is not None:
            improved = improved or float(bi["overall_accuracy_pct"]) > float(baseline["bilingual"]["overall_accuracy_pct"])
        passed = core_ok and bi_ok and improved
        return {"pass": passed, "alpha": alpha, "effective_lr": recipe["lr"] * alpha,
                "legacy_gate": core_ok, "legacy_failures": failures,
                "bilingual_gate": bi_ok if bi is not None else None, "strict_improvement": improved,
                "validation_loss": core["validation"]["validation_loss"],
                "overall_pct": core["dev"]["overall_accuracy_pct"],
                "categories": {k: v["accuracy_pct"] for k, v in core["dev"]["categories"].items()},
                "bilingual_pct": bi["overall_accuracy_pct"] if bi else None,
                "de_to_en_pct": bi["directions"]["de_to_en"]["accuracy_pct"] if bi else None,
                "en_to_de_pct": bi["directions"]["en_to_de"]["accuracy_pct"] if bi else None,
                "tokens": tokens, "path": str(snap), "checkpoint_sha256": file_sha(snap),
                "recipe": recipe, "throughput": throughput,
                "metrics": {**core, "bilingual": bi} if bi is not None else None}

    def line_search(raw: Path, recipe: dict, tag: str, throughput: dict) -> tuple[dict | None, list[dict]]:
        # Trust-step search is sequential rather than a fixed alpha grid. Usually one or two
        # full evaluations are enough; three is the hard cap.
        alpha = 0.04
        seen = set()
        attempts = []
        for index in range(3):
            alpha = max(0.005, min(0.10, alpha))
            key = round(alpha, 8)
            if key in seen:
                break
            seen.add(key)
            result = evaluate(raw, recipe, f"{tag}_a{str(alpha).replace('.', 'p')}", alpha, throughput)
            attempts.append(result)
            print(json.dumps({"event": "candidate", **{k:v for k,v in result.items() if k != "metrics"}}, flush=True))
            if result["pass"]:
                return result, attempts
            if not result["legacy_gate"] or result["bilingual_gate"] is False:
                alpha *= 0.5
            elif not result["strict_improvement"]:
                alpha *= 1.6
            else:
                alpha *= 0.5
        return None, attempts

    round_no = 0
    while anchor_tokens < a.target_tokens and round_no < a.max_rounds:
        if (time.monotonic() - started) / 60 >= a.max_wall_minutes:
            break
        round_no += 1
        remaining = a.target_tokens - anchor_tokens
        block_tokens = normalized_budget(min(BLOCK_LADDER[min(block_index, len(BLOCK_LADDER)-1)], remaining))
        if block_tokens < 100_000:
            break
        selected = None
        round_attempts = []
        recipe_start = min(conservative_level, len(BASE_RECIPES)-1)

        # Maximum two train runs per round. Every run produces a free midpoint snapshot,
        # so a half-block fallback never needs a second training pass.
        for recipe_index in range(recipe_start, min(recipe_start + 2, len(BASE_RECIPES))):
            recipe = scaled_recipe(BASE_RECIPES[recipe_index], block_tokens)
            tag = f"r{round_no:02d}_{recipe['name']}_{block_tokens}"
            raw, midpoint, throughput = train_block(block_tokens, recipe, tag)
            winner, tries = line_search(raw, recipe, tag + "_full", throughput)
            round_attempts.extend(tries)
            if winner is None and midpoint is not None:
                winner, tries = line_search(midpoint, recipe, tag + "_mid", throughput)
                round_attempts.extend(tries)
            if winner is not None:
                selected = winner
                conservative_level = max(0, recipe_index - 1)
                break

        if selected is None:
            success_streak = 0
            conservative_level = min(len(BASE_RECIPES)-1, conservative_level + 1)
            block_index = max(0, block_index - 1)
            history.append({"round": round_no, "success": False, "anchor_tokens": anchor_tokens,
                            "block_tokens": block_tokens, "attempts": [{k:v for k,v in x.items() if k != "metrics"} for x in round_attempts]})
            write_json(report / "progress.json", history)
            break

        destdir = assets / "runs" / f"neral350_scalev3_{a.request_id}_r{round_no:02d}_selected"
        if destdir.exists():
            raise FileExistsError("promotion destination exists")
        destdir.mkdir(parents=True)
        dest = destdir / "resume.pt"
        shutil.copy2(selected["path"], dest)
        if file_sha(dest) != selected["checkpoint_sha256"]:
            raise RuntimeError("promotion hash mismatch")

        before = anchor_tokens
        anchor, anchor_sha, anchor_tokens = dest, selected["checkpoint_sha256"], int(selected["tokens"])
        baseline = selected["metrics"]
        if baseline is None:
            raise RuntimeError("selected candidate lacks complete gate evidence")
        gained = anchor_tokens - before
        full_block = gained >= int(block_tokens * 0.90)
        success_streak = success_streak + 1 if full_block else 0
        if success_streak >= 2 and block_index < len(BLOCK_LADDER)-1:
            block_index += 1
            success_streak = 0

        state = {"schema": 3, "safe_anchor_path": str(anchor), "safe_anchor_sha256": anchor_sha,
                 "safe_anchor_tokens": anchor_tokens, "target_tokens": a.target_tokens,
                 "block_index": block_index, "success_streak": success_streak,
                 "conservative_level": conservative_level,
                 "baseline": {"checkpoint_sha256": anchor_sha, "checkpoint_processed_tokens": anchor_tokens, "metrics": baseline},
                 "updated_by_request": a.request_id, "updated_round": round_no}
        write_json(state_path, state)
        history.append({"round": round_no, "success": True, "anchor_tokens_before": before,
                        "anchor_tokens_after": anchor_tokens, "gained_tokens": gained, "block_tokens": block_tokens,
                        "full_block": full_block, "selected": {k:v for k,v in selected.items() if k != "metrics"},
                        "next_block_index": block_index, "conservative_level": conservative_level,
                        "attempts": [{k:v for k,v in x.items() if k != "metrics"} for x in round_attempts]})
        write_json(report / "progress.json", history)

    result = {"schema": 3, "controller": "neral350-scale-v3-adaptive", "request_id": a.request_id,
              "target_tokens": a.target_tokens, "final_safe_tokens": anchor_tokens,
              "final_safe_sha256": anchor_sha, "final_safe_checkpoint": str(anchor),
              "rounds": history, "block_ladder": list(BLOCK_LADDER), "base_recipes": list(BASE_RECIPES),
              "safe_anchor_preserved_on_failure": True, "full_promotion_gate_unchanged": True,
              "benchmarks_used_for_training": False, "final_audit_pending": True,
              "wall_minutes": (time.monotonic()-started)/60}
    write_json(report / "selection.json", result)
    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
