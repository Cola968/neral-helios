from __future__ import annotations

"""Scale V3 success controller for Neral 350M.

Design goal: materially raise safe-promotion probability and reduce evaluation overhead
without weakening the immutable promotion gate. The controller learns only control
parameters (block size / recipe / recovery alpha) from evaluation outcomes; benchmark
content is never added to training.
"""

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
LADDER = (1_000_000, 2_000_000, 5_000_000, 10_000_000, 25_000_000,
          50_000_000, 100_000_000, 250_000_000, 500_000_000)
PACK, SEQ, BATCH, MICROS = 6, 512, 4, 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROS
MIN_REPLAY_TAIL = 3 * SEQ * BATCH
STATE_FILE = "frontier350_scale_v3_state.json"
CATEGORIES = {"bilingual", "code", "factual", "language", "math", "reasoning", "science"}

RECIPES = (
    {"name": "balanced", "lr": 3.0e-5, "lr_end": 1.2e-5, "replay_boost": 1.45, "mode": "replay_only"},
    {"name": "retention", "lr": 2.2e-5, "lr_end": 8.0e-6, "replay_boost": 1.65, "mode": "replay_only"},
    {"name": "projected_rescue", "lr": 1.5e-5, "lr_end": 5.0e-6, "replay_boost": 1.90, "mode": "projected"},
)


def exec_script(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def ck_tokens(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["processed_tokens"])


def discover_anchor(assets: Path) -> tuple[Path, str, int, dict]:
    state_path = assets / STATE_FILE
    if state_path.is_file():
        state = read_json(state_path)
        p = Path(state["safe_anchor_path"])
        if p.is_file() and file_sha(p) == state["safe_anchor_sha256"] and ck_tokens(p) == int(state["safe_anchor_tokens"]):
            return p, state["safe_anchor_sha256"], int(state["safe_anchor_tokens"]), state
    candidates = []
    for p in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        if p.is_file():
            try:
                candidates.append((ck_tokens(p), p, file_sha(p)))
            except Exception:
                pass
    if not candidates:
        raise RuntimeError("no promoted Neral 350M anchor found")
    tokens, path, sha = max(candidates, key=lambda x: x[0])
    return path, sha, tokens, {}


def valid_budget(value: int) -> int:
    rem = value % TOKENS_PER_UPDATE
    if 0 < rem < MIN_REPLAY_TAIL:
        value -= rem
    return max(0, value)


def scaled_recipe(base: dict, block_tokens: int, failure_pressure: int) -> dict:
    scale = max(0.28, min(1.0, (1_000_000 / max(1_000_000, block_tokens)) ** 0.18))
    decades = max(0.0, math.log10(max(1.0, block_tokens / 1_000_000)))
    pressure = min(3, max(0, failure_pressure))
    return {
        **base,
        "lr": base["lr"] * scale * (0.90 ** pressure),
        "lr_end": base["lr_end"] * scale * (0.90 ** pressure),
        "replay_boost": min(2.35, base["replay_boost"] + 0.08 * decades + 0.08 * pressure),
        "scale": scale,
        "failure_pressure": pressure,
    }


def train_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj.get("event") == "train":
                rows.append(obj)
    return rows


def p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def check_dev(report: dict) -> None:
    if report.get("suite") != "arctus-recovery-dev-v1" or int(report.get("cases", 0)) != 140:
        raise ValueError("wrong dev suite")
    if set(report.get("categories", {})) != CATEGORIES:
        raise ValueError("wrong dev categories")
    for name, row in report["categories"].items():
        if row.get("total") != 20 or not 0 <= int(row.get("correct", -1)) <= 20:
            raise ValueError(f"invalid category counts: {name}")


def core_gate(base: dict, cand: dict) -> tuple[bool, list[str]]:
    bd, bv, cd, cv = base["dev"], base["validation"], cand["dev"], cand["validation"]
    check_dev(bd)
    check_dev(cd)
    validate_pair(bd, bv, cd, cv)
    failures = []
    if float(cd["overall_accuracy_pct"]) < float(bd["overall_accuracy_pct"]):
        failures.append("overall")
    for cat in CATEGORIES:
        if float(cd["categories"][cat]["accuracy_pct"]) < float(bd["categories"][cat]["accuracy_pct"]):
            failures.append(cat)
    if float(cv["validation_loss"]) > float(bv["validation_loss"]):
        failures.append("validation")
    return not failures, sorted(failures)


def bi_gate(base: dict, cand: dict, dev: dict) -> bool:
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
        cand["directions"][d]["accuracy_pct"] >= base["directions"][d]["accuracy_pct"] for d in base["directions"])


def alpha_pool(preferred: float) -> list[float]:
    values = [preferred, preferred * 0.5, preferred * 1.6]
    out = []
    for value in values:
        value = round(max(0.005, min(0.10, value)), 8)
        if value not in out:
            out.append(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--target-tokens", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--max-wall-minutes", type=int, default=180)
    a = ap.parse_args()
    assets, report, scratch = map(lambda x: Path(x).resolve(), (a.assets, a.report, a.scratch))
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report not empty")
    if scratch.exists():
        raise FileExistsError("scratch exists")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_root = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    main_data, cases = main_root / "train", main_root / "bilingual_holdout.json"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    state_path = assets / STATE_FILE
    started = time.monotonic()

    anchor, anchor_sha, anchor_tokens, state = discover_anchor(assets)
    block_index = int(state.get("block_index", 0))
    streak = int(state.get("success_streak", 0))
    conservative = int(state.get("conservative_level", 1))
    preferred_alpha = float(state.get("preferred_alpha", 0.04))
    failure_pressure = int(state.get("failure_pressure", 0))
    stalled_rounds = 0
    history = []

    def eval_core(checkpoint: Path, tag: str, tokens: int) -> dict:
        vp, dp = report / f"{tag}_validation.json", report / f"{tag}_dev.json"
        exec_script("eval_validation_loss350.py", "--checkpoint", checkpoint, "--valid-data", valid, "--out", vp,
                    "--seq", 512, "--batch", 4, "--batches", 32, "--seed", 351337)
        exec_script("eval_recovery_dev_v2.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer, "--out", dp,
                    "--expected-tokens", tokens)
        out = {"validation": read_json(vp), "dev": read_json(dp)}
        sha = file_sha(checkpoint)
        for obj in out.values():
            if obj.get("checkpoint_sha256") != sha or int(obj.get("checkpoint_processed_tokens", -1)) != tokens:
                raise ValueError("core identity mismatch")
        return out

    def eval_bilingual(checkpoint: Path, tag: str, tokens: int, dev: dict) -> dict:
        bp = report / f"{tag}_bilingual.json"
        exec_script("eval_bilingual_holdout_v1.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer,
                    "--cases", cases, "--out", bp, "--expected-tokens", tokens)
        obj = read_json(bp)
        if obj.get("checkpoint_sha256") != file_sha(checkpoint) or int(obj.get("checkpoint_processed_tokens", -1)) != tokens:
            raise ValueError("bilingual identity mismatch")
        if obj.get("tokenizer_sha256") != dev.get("tokenizer_sha256"):
            raise ValueError("tokenizer mismatch")
        return obj

    cached = state.get("baseline")
    if cached and cached.get("checkpoint_sha256") == anchor_sha and int(cached.get("checkpoint_processed_tokens", -1)) == anchor_tokens:
        baseline = cached["metrics"]
    else:
        tag = f"baseline_{anchor_tokens}"
        core = eval_core(anchor, tag, anchor_tokens)
        baseline = {**core, "bilingual": eval_bilingual(anchor, tag, anchor_tokens, core["dev"])}

    def smoke_baseline(rotation: int, tag: str) -> tuple[dict[str, dict], float]:
        selected: dict[str, dict] = {}
        per_cat: dict[str, list[dict]] = {}
        for row in baseline["dev"]["rows"]:
            per_cat.setdefault(row["category"], []).append(row)
        for cat in sorted(per_cat):
            rows = per_cat[cat]
            for i, row in enumerate(rows):
                if i % 4 == rotation:
                    selected[row["id"]] = row
        if len(selected) != 35:
            raise RuntimeError("baseline smoke subset must contain 35 cases")
        vp = report / f"{tag}_baseline_smoke_validation.json"
        exec_script("eval_validation_loss350.py", "--checkpoint", anchor, "--valid-data", valid, "--out", vp,
                    "--seq", 512, "--batch", 4, "--batches", 8, "--seed", 351337)
        return selected, float(read_json(vp)["validation_loss"])

    def train(block_tokens: int, recipe: dict, tag: str) -> tuple[Path, Path | None, dict]:
        out = scratch / tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(anchor), "--anchor-sha", anchor_sha,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", recipe["mode"], "--new-tokens", str(block_tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",
               "--lr", str(recipe["lr"]), "--lr-end", str(recipe["lr_end"]),
               "--replay-boost", str(recipe["replay_boost"]), "--snapshot-fraction", "0.5"]
        if recipe["mode"] == "replay_only":
            cmd.append("--fast-replay-only")
        print(json.dumps({"event": "train_start", "tag": tag, "block_tokens": block_tokens, "recipe": recipe}), flush=True)
        subprocess.run(cmd, check=True)
        summary = read_json(out / "summary.json")
        if not summary.get("complete") or not summary.get("frozen_weights_unchanged"):
            raise RuntimeError("training contract failed")
        rates = [float(r["tokens_per_second"]) for r in train_rows(out / "metrics.jsonl")[5:]
                 if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
        perf = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates),
                "p10_tps": p10(rates), "peak_tps": max(rates)} if rates else {}
        mid = out / "midpoint.pt"
        return out / "resume.pt", mid if mid.is_file() else None, perf

    def make_interp(raw: Path, alpha: float, tag: str) -> Path:
        snap = scratch / f"{tag}.pt"
        exec_script("interpolate_recovery_checkpoint.py", "--anchor", anchor, "--candidate", raw,
                    "--out", snap, "--alpha", alpha)
        return snap

    def smoke(raw: Path, alpha: float, tag: str, rotation: int,
              base_rows: dict[str, dict], base_val: float) -> dict:
        snap = make_interp(raw, alpha, tag)
        tokens = ck_tokens(snap)
        sp, vp = report / f"{tag}_smoke.json", report / f"{tag}_smoke_validation.json"
        exec_script("eval_recovery_smoke_v3.py", "--checkpoint", snap, "--tokenizer", tokenizer,
                    "--out", sp, "--expected-tokens", tokens, "--rotation", rotation)
        exec_script("eval_validation_loss350.py", "--checkpoint", snap, "--valid-data", valid, "--out", vp,
                    "--seq", 512, "--batch", 4, "--batches", 8, "--seed", 351337)
        sm = read_json(sp)
        val = float(read_json(vp)["validation_loss"])
        regressions = gains = 0
        margin_delta = 0.0
        for row in sm["rows"]:
            base = base_rows[row["id"]]
            if bool(base["correct"]) and not bool(row["correct"]):
                regressions += 1
            if not bool(base["correct"]) and bool(row["correct"]):
                gains += 1
            margin_delta += float(row["margin_nats_per_bpe"]) - float(base["margin_nats_per_bpe"])
        margin_delta /= len(sm["rows"])
        rank = (regressions, -gains, max(0.0, val - base_val), val - base_val, -margin_delta)
        result = {"alpha": alpha, "path": str(snap), "tokens": tokens, "rank": list(rank),
                  "smoke_regressions": regressions, "smoke_gains": gains,
                  "smoke_margin_delta": margin_delta, "smoke_validation_loss": val,
                  "smoke_validation_delta": val - base_val}
        print(json.dumps({"event": "smoke_candidate", **result}), flush=True)
        return result

    def full_gate(smoke_row: dict, recipe: dict, tag: str, perf: dict) -> dict:
        snap = Path(smoke_row["path"])
        tokens = ck_tokens(snap)
        core = eval_core(snap, tag, tokens)
        core_ok, failures = core_gate(baseline, core)
        bi = eval_bilingual(snap, tag, tokens, core["dev"]) if core_ok else None
        bi_ok = bi_gate(baseline["bilingual"], bi, core["dev"]) if bi is not None else False
        category_improved = any(
            float(core["dev"]["categories"][c]["accuracy_pct"]) >
            float(baseline["dev"]["categories"][c]["accuracy_pct"])
            for c in CATEGORIES
        )
        improved = (
            float(core["validation"]["validation_loss"]) < float(baseline["validation"]["validation_loss"])
            or category_improved
        )
        if bi is not None:
            improved = improved or float(bi["overall_accuracy_pct"]) > float(baseline["bilingual"]["overall_accuracy_pct"])
        passed = core_ok and bi_ok and improved
        result = {**smoke_row, "pass": passed, "effective_lr": recipe["lr"] * smoke_row["alpha"],
                  "legacy_gate": core_ok, "legacy_failures": failures,
                  "bilingual_gate": bi_ok if bi is not None else None,
                  "strict_improvement": improved,
                  "validation_loss": core["validation"]["validation_loss"],
                  "overall_pct": core["dev"]["overall_accuracy_pct"],
                  "categories": {k: v["accuracy_pct"] for k, v in core["dev"]["categories"].items()},
                  "bilingual_pct": bi["overall_accuracy_pct"] if bi else None,
                  "de_to_en_pct": bi["directions"]["de_to_en"]["accuracy_pct"] if bi else None,
                  "en_to_de_pct": bi["directions"]["en_to_de"]["accuracy_pct"] if bi else None,
                  "checkpoint_sha256": file_sha(snap), "recipe": recipe, "throughput": perf,
                  "metrics": {**core, "bilingual": bi} if bi is not None else None}
        print(json.dumps({"event": "full_candidate", **{k: v for k, v in result.items() if k != "metrics"}}), flush=True)
        return result

    def search_checkpoint(raw: Path, recipe: dict, tag: str, perf: dict, rotation: int,
                          base_rows: dict[str, dict], base_val: float) -> tuple[dict | None, list[dict]]:
        screened = [
            smoke(raw, alpha, f"{tag}_a{str(alpha).replace('.', 'p')}", rotation, base_rows, base_val)
            for alpha in alpha_pool(preferred_alpha)
        ]
        screened.sort(key=lambda x: tuple(x["rank"]))
        attempts = []
        for index, row in enumerate(screened[:2]):
            result = full_gate(row, recipe, f"{tag}_full{index + 1}", perf)
            attempts.append(result)
            if result["pass"]:
                return result, attempts
        return None, attempts

    def persist_control() -> None:
        write_json(state_path, {
            "schema": 4,
            "safe_anchor_path": str(anchor),
            "safe_anchor_sha256": anchor_sha,
            "safe_anchor_tokens": anchor_tokens,
            "target_tokens": a.target_tokens,
            "block_index": block_index,
            "success_streak": streak,
            "conservative_level": conservative,
            "preferred_alpha": preferred_alpha,
            "failure_pressure": failure_pressure,
            "baseline": {"checkpoint_sha256": anchor_sha,
                         "checkpoint_processed_tokens": anchor_tokens,
                         "metrics": baseline},
            "updated_by_request": a.request_id,
            "history_tail": history[-4:],
        })

    round_no = 0
    while (anchor_tokens < a.target_tokens and round_no < a.max_rounds and
           (time.monotonic() - started) / 60 < a.max_wall_minutes):
        round_no += 1
        remaining = a.target_tokens - anchor_tokens
        block = valid_budget(min(LADDER[min(block_index, len(LADDER) - 1)], remaining))
        if block < 100_000:
            break
        rotation = (round_no - 1) % 4
        base_smoke_rows, base_smoke_val = smoke_baseline(rotation, f"r{round_no:02d}")
        selected = None
        attempts_all = []
        start_recipe = min(conservative, len(RECIPES) - 1)

        for ri in range(start_recipe, min(start_recipe + 2, len(RECIPES))):
            recipe = scaled_recipe(RECIPES[ri], block, failure_pressure)
            tag = f"r{round_no:02d}_{recipe['name']}_{block}"
            raw, midpoint, perf = train(block, recipe, tag)
            winner, tries = search_checkpoint(raw, recipe, tag + "_end", perf,
                                               rotation, base_smoke_rows, base_smoke_val)
            attempts_all.extend(tries)
            if winner is None and midpoint is not None:
                winner, tries = search_checkpoint(midpoint, recipe, tag + "_mid", perf,
                                                   rotation, base_smoke_rows, base_smoke_val)
                attempts_all.extend(tries)
            if winner is not None:
                selected = winner
                conservative = max(0, ri - 1)
                break

        if selected is None:
            stalled_rounds += 1
            streak = 0
            failure_pressure = min(3, failure_pressure + 1)
            conservative = min(len(RECIPES) - 1, conservative + 1)
            if block_index > 0:
                block_index -= 1
            if attempts_all:
                best = min(
                    attempts_all,
                    key=lambda x: (
                        len(x.get("legacy_failures", [])),
                        0 if x.get("bilingual_gate") is not False else 1,
                        abs(float(x.get("smoke_validation_delta", 0.0))),
                    ),
                )
                if not best.get("legacy_gate") or best.get("bilingual_gate") is False:
                    preferred_alpha = max(0.005, float(best["alpha"]) * 0.6)
                elif not best.get("strict_improvement"):
                    preferred_alpha = min(0.10, float(best["alpha"]) * 1.5)
            history.append({"round": round_no, "success": False,
                            "anchor_tokens": anchor_tokens, "block_tokens": block,
                            "next_block_index": block_index, "preferred_alpha": preferred_alpha,
                            "failure_pressure": failure_pressure,
                            "attempts": [{k: v for k, v in x.items() if k != "metrics"}
                                         for x in attempts_all]})
            write_json(report / "progress.json", history)
            persist_control()
            if stalled_rounds >= 2 and block_index == 0:
                break
            continue

        stalled_rounds = 0
        failure_pressure = max(0, failure_pressure - 1)
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
            raise RuntimeError("selected candidate missing complete evidence")
        preferred_alpha = max(0.005, min(0.10, float(selected["alpha"])))
        gained = anchor_tokens - before
        full_block = gained >= int(block * 0.90)
        streak = streak + 1 if full_block else 0
        needed = 1 if block_index == 0 else 2
        if full_block and streak >= needed and block_index < len(LADDER) - 1:
            block_index += 1
            streak = 0
        history.append({"round": round_no, "success": True,
                        "anchor_tokens_before": before, "anchor_tokens_after": anchor_tokens,
                        "gained_tokens": gained, "block_tokens": block, "full_block": full_block,
                        "selected": {k: v for k, v in selected.items() if k != "metrics"},
                        "next_block_index": block_index, "preferred_alpha": preferred_alpha,
                        "attempts": [{k: v for k, v in x.items() if k != "metrics"}
                                     for x in attempts_all]})
        write_json(report / "progress.json", history)
        persist_control()

    result = {
        "schema": 4,
        "controller": "neral350-scale-v3-success",
        "request_id": a.request_id,
        "target_tokens": a.target_tokens,
        "final_safe_tokens": anchor_tokens,
        "final_safe_sha256": anchor_sha,
        "final_safe_checkpoint": str(anchor),
        "rounds": history,
        "block_ladder": list(LADDER),
        "preferred_alpha": preferred_alpha,
        "conservative_level": conservative,
        "failure_pressure": failure_pressure,
        "safe_anchor_preserved_on_failure": True,
        "full_promotion_gate_unchanged": True,
        "smoke_is_ranking_only": True,
        "benchmarks_used_for_training": False,
        "final_audit_pending": True,
        "wall_minutes": (time.monotonic() - started) / 60,
    }
    write_json(report / "selection.json", result)
    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
