from __future__ import annotations

"""Scale V4 staged controller for Neral 350M.

Goals:
- materially reduce evaluation overhead by using larger stages;
- use the measured-stable Pack9 execution path;
- break the 1M strict-improvement deadlock without weakening SAFE promotion gates;
- allow a clearly labelled SHADOW checkpoint to accumulate healthy training when a
  candidate is statistically tied, while the last fully promoted SAFE anchor remains
  immutable and recoverable.

A shadow checkpoint is NEVER reported as a safe promotion. Full safe promotion still
requires: no fixed validation/dev regression, no bilingual regression, and at least one
strict quality improvement versus the current safe anchor.
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
# Keep stages at <=100M. The trainer always writes a 50%-point snapshot, so the largest
# stage now has a 50M recovery point and cannot burn another full 250M before a quality
# boundary. This preserves the SAFE/SHADOW gates and training recipe math.
LADDER = (5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000)
PACK, SEQ, BATCH, MICROS = 9, 512, 4, 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROS
MIN_REPLAY_TAIL = 3 * SEQ * BATCH
STATE_FILE = "frontier350_scale_v4_state.json"
CATEGORIES = {"bilingual", "code", "factual", "language", "math", "reasoning", "science"}

RECIPES = (
    {"name": "balanced", "lr": 2.6e-5, "lr_end": 1.0e-5, "replay_boost": 1.50, "mode": "replay_only"},
    {"name": "retention", "lr": 2.0e-5, "lr_end": 7.0e-6, "replay_boost": 1.78, "mode": "replay_only"},
    {"name": "projected_rescue", "lr": 1.35e-5, "lr_end": 4.5e-6, "replay_boost": 2.00, "mode": "projected"},
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


def valid_checkpoint(path: Path, sha: str, tokens: int) -> bool:
    try:
        return path.is_file() and file_sha(path) == sha and ck_tokens(path) == int(tokens)
    except Exception:
        return False


def discover_safe_anchor(assets: Path) -> tuple[Path, str, int, dict]:
    state_path = assets / STATE_FILE
    if state_path.is_file():
        state = read_json(state_path)
        p = Path(state["safe_anchor_path"])
        if valid_checkpoint(p, state["safe_anchor_sha256"], int(state["safe_anchor_tokens"])):
            return p, state["safe_anchor_sha256"], int(state["safe_anchor_tokens"]), state

    # Prefer the existing Scale V3 state because it is the immediate rollback source.
    v3 = assets / "frontier350_scale_v3_state.json"
    if v3.is_file():
        state = read_json(v3)
        p = Path(state["safe_anchor_path"])
        if valid_checkpoint(p, state["safe_anchor_sha256"], int(state["safe_anchor_tokens"])):
            return p, state["safe_anchor_sha256"], int(state["safe_anchor_tokens"]), {}

    candidates = []
    for p in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        if p.is_file():
            try:
                candidates.append((ck_tokens(p), p, file_sha(p)))
            except Exception:
                pass
    if not candidates:
        raise RuntimeError("no promoted Neral 350M safe anchor found")
    tokens, path, sha = max(candidates, key=lambda x: x[0])
    return path, sha, tokens, {}


def valid_budget(value: int) -> int:
    rem = value % TOKENS_PER_UPDATE
    if 0 < rem < MIN_REPLAY_TAIL:
        value -= rem
    return max(0, value)


def scaled_recipe(base: dict, block_tokens: int, pressure: int) -> dict:
    # Larger blocks need a lower LR; pressure makes recovery progressively gentler.
    scale = max(0.24, min(1.0, (5_000_000 / max(5_000_000, block_tokens)) ** 0.17))
    decades = max(0.0, math.log10(max(1.0, block_tokens / 5_000_000)))
    pressure = min(3, max(0, pressure))
    return {
        **base,
        "lr": base["lr"] * scale * (0.90 ** pressure),
        "lr_end": base["lr_end"] * scale * (0.90 ** pressure),
        "replay_boost": min(2.45, base["replay_boost"] + 0.10 * decades + 0.08 * pressure),
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


def bi_protocol_ok(base: dict, cand: dict, dev: dict) -> None:
    for obj in (base, cand):
        if obj.get("cases") != 400 or set(obj.get("directions", {})) != {"de_to_en", "en_to_de"}:
            raise ValueError("invalid bilingual evidence")
    for key in ("cases_sha256", "tokenizer_sha256", "protocol_sha256"):
        if base.get(key) != cand.get(key):
            raise ValueError("bilingual protocol changed")
    for key in ("checkpoint_sha256", "checkpoint_processed_tokens", "tokenizer_sha256"):
        if cand.get(key) != dev.get(key):
            raise ValueError("mixed checkpoint evidence")


def bi_gate(base: dict, cand: dict, dev: dict) -> bool:
    bi_protocol_ok(base, cand, dev)
    return cand["overall_accuracy_pct"] >= base["overall_accuracy_pct"] and all(
        cand["directions"][d]["accuracy_pct"] >= base["directions"][d]["accuracy_pct"]
        for d in base["directions"]
    )


def bi_shadow_gate(base: dict, cand: dict, dev: dict) -> bool:
    """Allow at most one bilingual case of noise in SHADOW only, never SAFE.

    With 400 total cases and 200 per direction this corresponds to <=0.25 pp
    overall and <=0.50 pp in either direction. Core validation/dev must still be
    non-regressing before this function is considered.
    """
    bi_protocol_ok(base, cand, dev)
    if float(cand["overall_accuracy_pct"]) < float(base["overall_accuracy_pct"]) - 0.2500001:
        return False
    for d in base["directions"]:
        if float(cand["directions"][d]["accuracy_pct"]) < float(base["directions"][d]["accuracy_pct"]) - 0.5000001:
            return False
    return True


def alpha_pool(preferred: float) -> list[float]:
    # Two candidates instead of three: less interpolation/eval overhead. The preferred
    # alpha is always tested first, then one gentler fallback.
    values = [preferred, preferred * 0.60]
    out = []
    for value in values:
        value = round(max(0.006, min(0.12, value)), 8)
        if value not in out:
            out.append(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--target-tokens", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--max-rounds", type=int, default=16)
    ap.add_argument("--max-wall-minutes", type=int, default=660)
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

    safe_anchor, safe_sha, safe_tokens, state = discover_safe_anchor(assets)
    baseline = None
    block_index = int(state.get("block_index", 0))
    preferred_alpha = float(state.get("preferred_alpha", 0.06))
    conservative = int(state.get("conservative_level", 0))
    failure_pressure = int(state.get("failure_pressure", 0))
    history = []

    # Resume an explicitly labelled shadow only if it was derived from this exact safe
    # anchor. It is intentionally not discoverable by the normal *_selected scan.
    train_anchor, train_sha, train_tokens = safe_anchor, safe_sha, safe_tokens
    shadow = state.get("shadow")
    if shadow and shadow.get("base_safe_sha256") == safe_sha:
        p = Path(shadow.get("path", ""))
        if valid_checkpoint(p, shadow.get("sha256", ""), int(shadow.get("tokens", -1))):
            train_anchor, train_sha, train_tokens = p, shadow["sha256"], int(shadow["tokens"])

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
    if cached and cached.get("checkpoint_sha256") == safe_sha and int(cached.get("checkpoint_processed_tokens", -1)) == safe_tokens:
        baseline = cached["metrics"]
    else:
        tag = f"baseline_{safe_tokens}"
        core = eval_core(safe_anchor, tag, safe_tokens)
        baseline = {**core, "bilingual": eval_bilingual(safe_anchor, tag, safe_tokens, core["dev"])}

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
        exec_script("eval_validation_loss350.py", "--checkpoint", safe_anchor, "--valid-data", valid, "--out", vp,
                    "--seq", 512, "--batch", 4, "--batches", 4, "--seed", 351337)
        return selected, float(read_json(vp)["validation_loss"])

    def train(block_tokens: int, recipe: dict, tag: str) -> tuple[Path, Path | None, dict]:
        out = scratch / tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(train_anchor), "--anchor-sha", train_sha,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", recipe["mode"], "--new-tokens", str(block_tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",
               "--lr", str(recipe["lr"]), "--lr-end", str(recipe["lr_end"]),
               "--replay-boost", str(recipe["replay_boost"]), "--snapshot-fraction", "0.5"]
        if recipe["mode"] == "replay_only":
            cmd.append("--fast-replay-only")
        print(json.dumps({"event": "train_start", "tag": tag, "from_tokens": train_tokens,
                          "block_tokens": block_tokens, "pack": PACK, "recipe": recipe}), flush=True)
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
        exec_script("interpolate_recovery_checkpoint.py", "--anchor", train_anchor, "--candidate", raw,
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
                    "--seq", 512, "--batch", 4, "--batches", 4, "--seed", 351337)
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
        shadow_bi_ok = bi_shadow_gate(baseline["bilingual"], bi, core["dev"]) if bi is not None else False
        category_improved = any(
            float(core["dev"]["categories"][c]["accuracy_pct"]) >
            float(baseline["dev"]["categories"][c]["accuracy_pct"])
            for c in CATEGORIES
        )
        improved = (float(core["validation"]["validation_loss"]) < float(baseline["validation"]["validation_loss"])
                    or category_improved)
        if bi is not None:
            improved = improved or float(bi["overall_accuracy_pct"]) > float(baseline["bilingual"]["overall_accuracy_pct"])
        passed = core_ok and bi_ok and improved
        shadow_ok = core_ok and shadow_bi_ok
        result = {**smoke_row, "pass": passed, "shadow_ok": shadow_ok,
                  "shadow_tolerance_only": bool(shadow_ok and not bi_ok),
                  "effective_lr": recipe["lr"] * smoke_row["alpha"],
                  "legacy_gate": core_ok, "legacy_failures": failures,
                  "bilingual_gate": bi_ok if bi is not None else None,
                  "bilingual_shadow_gate": shadow_bi_ok if bi is not None else None,
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
                          base_rows: dict[str, dict], base_val: float) -> tuple[dict | None, dict | None, list[dict]]:
        screened = [smoke(raw, alpha, f"{tag}_a{str(alpha).replace('.', 'p')}", rotation, base_rows, base_val)
                    for alpha in alpha_pool(preferred_alpha)]
        screened.sort(key=lambda x: tuple(x["rank"]))
        attempts = []
        shadow_candidate = None
        for index, row in enumerate(screened):
            result = full_gate(row, recipe, f"{tag}_full{index + 1}", perf)
            attempts.append(result)
            if result["pass"]:
                return result, None, attempts
            # This is the key V4 deadlock breaker: a healthy/tied model may keep training
            # as SHADOW, but SAFE remains untouched. Stop here to avoid a second expensive
            # full evaluation when we already have a usable staging checkpoint.
            if result["shadow_ok"]:
                shadow_candidate = result
                break
        return None, shadow_candidate, attempts

    def persist_control(shadow_row: dict | None = None) -> None:
        shadow_state = None
        if train_sha != safe_sha and train_tokens > safe_tokens:
            shadow_state = {"path": str(train_anchor), "sha256": train_sha, "tokens": train_tokens,
                            "base_safe_sha256": safe_sha, "base_safe_tokens": safe_tokens,
                            "tolerance_only": bool(shadow_row and shadow_row.get("shadow_tolerance_only"))}
        write_json(state_path, {
            "schema": 5,
            "safe_anchor_path": str(safe_anchor),
            "safe_anchor_sha256": safe_sha,
            "safe_anchor_tokens": safe_tokens,
            "shadow": shadow_state,
            "target_tokens": a.target_tokens,
            "block_index": block_index,
            "preferred_alpha": preferred_alpha,
            "conservative_level": conservative,
            "failure_pressure": failure_pressure,
            "baseline": {"checkpoint_sha256": safe_sha, "checkpoint_processed_tokens": safe_tokens, "metrics": baseline},
            "updated_by_request": a.request_id,
            "history_tail": history[-4:],
        })

    round_no = 0
    hard_failures = 0
    while (train_tokens < a.target_tokens and round_no < a.max_rounds and
           (time.monotonic() - started) / 60 < a.max_wall_minutes):
        round_no += 1
        remaining = a.target_tokens - train_tokens
        block = valid_budget(min(LADDER[min(block_index, len(LADDER) - 1)], remaining))
        if block < 100_000:
            break
        rotation = (round_no - 1) % 4
        base_smoke_rows, base_smoke_val = smoke_baseline(rotation, f"r{round_no:02d}")
        selected = None
        shadow_candidate = None
        attempts_all = []
        start_recipe = min(conservative, len(RECIPES) - 1)

        # One recipe normally; a second recipe is attempted only after a hard quality
        # failure. A healthy shadow immediately advances to the next stage.
        for ri in range(start_recipe, min(start_recipe + 2, len(RECIPES))):
            recipe = scaled_recipe(RECIPES[ri], block, failure_pressure)
            tag = f"r{round_no:02d}_{recipe['name']}_{block}"
            raw, midpoint, perf = train(block, recipe, tag)
            winner, shadow_try, tries = search_checkpoint(raw, recipe, tag + "_end", perf,
                                                           rotation, base_smoke_rows, base_smoke_val)
            attempts_all.extend(tries)
            if winner is not None:
                selected = winner
                conservative = max(0, ri - 1)
                break
            if shadow_try is not None:
                shadow_candidate = shadow_try
                # If the shadow used the one-case bilingual tolerance, force retention
                # next stage; otherwise return toward the faster balanced recipe.
                conservative = 1 if shadow_try.get("shadow_tolerance_only") else max(0, ri - 1)
                break
            # Endpoint hard-failed. A midpoint gets one rescue check before changing recipe.
            if midpoint is not None:
                winner, shadow_try, tries = search_checkpoint(midpoint, recipe, tag + "_mid", perf,
                                                               rotation, base_smoke_rows, base_smoke_val)
                attempts_all.extend(tries)
                if winner is not None:
                    selected = winner
                    conservative = max(0, ri - 1)
                    break
                if shadow_try is not None:
                    shadow_candidate = shadow_try
                    conservative = 1 if shadow_try.get("shadow_tolerance_only") else max(0, ri - 1)
                    break
        if selected is not None:
            before_safe = safe_tokens
            destdir = assets / "runs" / f"neral350_scalev4_{a.request_id}_r{round_no:02d}_selected"
            if destdir.exists():
                raise FileExistsError("safe promotion destination exists")
            destdir.mkdir(parents=True)
            dest = destdir / "resume.pt"
            shutil.copy2(selected["path"], dest)
            if file_sha(dest) != selected["checkpoint_sha256"]:
                raise RuntimeError("safe promotion hash mismatch")
            safe_anchor, safe_sha, safe_tokens = dest, selected["checkpoint_sha256"], int(selected["tokens"])
            baseline = selected["metrics"]
            if baseline is None:
                raise RuntimeError("selected candidate missing complete evidence")
            train_anchor, train_sha, train_tokens = safe_anchor, safe_sha, safe_tokens
            preferred_alpha = max(0.006, min(0.12, float(selected["alpha"])))
            failure_pressure = max(0, failure_pressure - 1)
            hard_failures = 0
            if block_index < len(LADDER) - 1:
                block_index += 1
            history.append({"round": round_no, "status": "SAFE_PROMOTION",
                            "safe_tokens_before": before_safe, "safe_tokens_after": safe_tokens,
                            "trained_tokens": train_tokens, "block_tokens": block,
                            "next_block_index": block_index,
                            "selected": {k: v for k, v in selected.items() if k != "metrics"},
                            "attempts": [{k: v for k, v in x.items() if k != "metrics"} for x in attempts_all]})
            write_json(report / "progress.json", history)
            persist_control()
            continue

        if shadow_candidate is not None:
            before_train = train_tokens
            destdir = assets / "runs" / f"neral350_scalev4_{a.request_id}_r{round_no:02d}_shadow"
            if destdir.exists():
                raise FileExistsError("shadow destination exists")
            destdir.mkdir(parents=True)
            dest = destdir / "resume.pt"
            shutil.copy2(shadow_candidate["path"], dest)
            if file_sha(dest) != shadow_candidate["checkpoint_sha256"]:
                raise RuntimeError("shadow hash mismatch")
            train_anchor, train_sha, train_tokens = dest, shadow_candidate["checkpoint_sha256"], int(shadow_candidate["tokens"])
            preferred_alpha = max(0.006, min(0.12, float(shadow_candidate["alpha"]) *
                                             (0.80 if shadow_candidate.get("shadow_tolerance_only") else 1.25)))
            failure_pressure = max(0, failure_pressure - 1)
            hard_failures = 0
            if block_index < len(LADDER) - 1:
                block_index += 1
            history.append({"round": round_no, "status": "SHADOW_PROGRESS",
                            "safe_tokens": safe_tokens, "train_tokens_before": before_train,
                            "train_tokens_after": train_tokens, "block_tokens": block,
                            "next_block_index": block_index,
                            "shadow": {k: v for k, v in shadow_candidate.items() if k != "metrics"},
                            "attempts": [{k: v for k, v in x.items() if k != "metrics"} for x in attempts_all]})
            write_json(report / "progress.json", history)
            persist_control(shadow_candidate)
            continue

        # Hard regression: throw away shadow trajectory, return to immutable SAFE, and
        # retry a smaller/more-retentive stage. No unsafe checkpoint is promoted.
        hard_failures += 1
        train_anchor, train_sha, train_tokens = safe_anchor, safe_sha, safe_tokens
        failure_pressure = min(3, failure_pressure + 1)
        conservative = min(len(RECIPES) - 1, conservative + 1)
        preferred_alpha = max(0.006, preferred_alpha * 0.65)
        if block_index > 0:
            block_index -= 1
        history.append({"round": round_no, "status": "HARD_REJECT_RESET_TO_SAFE",
                        "safe_tokens": safe_tokens, "block_tokens": block,
                        "next_block_index": block_index, "failure_pressure": failure_pressure,
                        "attempts": [{k: v for k, v in x.items() if k != "metrics"} for x in attempts_all]})
        write_json(report / "progress.json", history)
        persist_control()
        if hard_failures >= 3 and block_index == 0:
            break

    result = {
        "schema": 5,
        "controller": "neral350-scale-v4-staged",
        "request_id": a.request_id,
        "target_tokens": a.target_tokens,
        "final_safe_tokens": safe_tokens,
        "final_safe_sha256": safe_sha,
        "final_safe_checkpoint": str(safe_anchor),
        "final_train_tokens": train_tokens,
        "final_train_sha256": train_sha,
        "final_status": "SAFE" if train_sha == safe_sha else "SHADOW_UNPROMOTED",
        "rounds": history,
        "block_ladder": list(LADDER),
        "pack": PACK,
        "preferred_alpha": preferred_alpha,
        "conservative_level": conservative,
        "failure_pressure": failure_pressure,
        "safe_anchor_preserved_on_failure": True,
        "safe_promotion_gate_unchanged": True,
        "shadow_one_case_bilingual_tolerance_only": True,
        "shadow_is_never_safe_promotion": True,
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
