from __future__ import annotations

"""Scale V5 monotonic-continuity controller for Neral 350M.

Design goals
------------
* Never discard more than the currently trained 5M-token micro-stage.
* Keep SAFE/release quality separate from training continuity.
* Persist every accepted continuity checkpoint on the runner's D: drive using an
  append-only two-generation journal, so a restart resumes the newest validated point.
* Run cheap health screening every 5M and expensive full SAFE audits only every 50M.
* On audit regression, recover forward with a more retentive recipe instead of jumping
  back hundreds of millions of tokens.
* Keep the strict SAFE promotion gate unchanged. CONTINUITY is explicitly not SAFE.

The trainer is patched by ensure_eval_runtime.sh before this controller is launched, so
--snapshot-interval-tokens and --resume-save-interval-tokens are available.
"""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from recovery_identity import file_sha, validate_pair

TARGET_DEFAULT = 10_000_000_000
STAGE_TOKENS = 5_000_000
FULL_AUDIT_INTERVAL = 50_000_000
PACK, SEQ, BATCH, MICROS = 9, 512, 4, 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROS
MIN_REPLAY_TAIL = 3 * SEQ * BATCH
STATE_FILE = "frontier350_scale_v5_state.json"
USABLE_MANIFEST = "frontier350_latest_usable.json"
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


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def ck_tokens(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["processed_tokens"])


def valid_checkpoint(path: Path, sha: str | None = None, tokens: int | None = None) -> bool:
    try:
        if not path.is_file():
            return False
        if sha is not None and file_sha(path) != sha:
            return False
        if tokens is not None and ck_tokens(path) != int(tokens):
            return False
        return True
    except Exception:
        return False


def valid_budget(value: int) -> int:
    rem = value % TOKENS_PER_UPDATE
    if 0 < rem < MIN_REPLAY_TAIL:
        value -= rem
    return max(0, value)


def train_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
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
    failures: list[str] = []
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


def scaled_recipe(base: dict, pressure: int) -> dict:
    pressure = min(5, max(0, pressure))
    return {
        **base,
        "lr": base["lr"] * (0.90 ** pressure),
        "lr_end": base["lr_end"] * (0.90 ** pressure),
        "replay_boost": min(2.55, base["replay_boost"] + 0.08 * pressure),
        "failure_pressure": pressure,
    }


def newest_v4_continuity(assets: Path) -> tuple[Path, str, int] | None:
    candidates: list[tuple[int, Path, str]] = []
    state_path = assets / "frontier350_scale_v4_state.json"
    if state_path.is_file():
        try:
            st = read_json(state_path)
            for key, path_key, sha_key, tok_key in (
                ("safe", "safe_anchor_path", "safe_anchor_sha256", "safe_anchor_tokens"),
                ("shadow", "path", "sha256", "tokens"),
            ):
                obj = st if key == "safe" else st.get("shadow")
                if not obj:
                    continue
                p = Path(obj[path_key])
                sha = str(obj[sha_key])
                tok = int(obj[tok_key])
                if valid_checkpoint(p, sha, tok):
                    candidates.append((tok, p, sha))
        except Exception:
            pass
    for pattern in ("neral350_scalev4_*_selected/resume.pt", "neral350_scalev4_*_shadow/resume.pt"):
        for p in (assets / "runs").glob(pattern):
            if p.is_file():
                try:
                    candidates.append((ck_tokens(p), p, file_sha(p)))
                except Exception:
                    pass
    if not candidates:
        return None
    tok, p, sha = max(candidates, key=lambda x: x[0])
    return p, sha, tok


def discover_initial(assets: Path, journal: Path) -> tuple[Path, str, int, Path, str, int, dict]:
    state_path = assets / STATE_FILE
    state: dict = {}
    if state_path.is_file():
        try:
            state = read_json(state_path)
            safe_obj = state.get("safe", {})
            cont_obj = state.get("continuity", {})
            safe = Path(safe_obj.get("path", ""))
            cont = Path(cont_obj.get("path", ""))
            if valid_checkpoint(safe, safe_obj.get("sha256"), safe_obj.get("tokens")) and \
               valid_checkpoint(cont, cont_obj.get("sha256"), cont_obj.get("tokens")):
                return (safe, safe_obj["sha256"], int(safe_obj["tokens"]),
                        cont, cont_obj["sha256"], int(cont_obj["tokens"]), state)
        except Exception:
            state = {}

    # Recover the newest V5 journal point even if the JSON state update was interrupted.
    journal_candidates: list[tuple[int, Path, str]] = []
    if journal.is_dir():
        for p in journal.glob("continuity_*.pt"):
            try:
                journal_candidates.append((ck_tokens(p), p, file_sha(p)))
            except Exception:
                pass

    v4 = newest_v4_continuity(assets)
    safe_candidates: list[tuple[int, Path, str]] = []
    v4_state = assets / "frontier350_scale_v4_state.json"
    if v4_state.is_file():
        try:
            st = read_json(v4_state)
            p = Path(st["safe_anchor_path"])
            sha = st["safe_anchor_sha256"]
            tok = int(st["safe_anchor_tokens"])
            if valid_checkpoint(p, sha, tok):
                safe_candidates.append((tok, p, sha))
        except Exception:
            pass
    for p in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        try:
            safe_candidates.append((ck_tokens(p), p, file_sha(p)))
        except Exception:
            pass
    if not safe_candidates:
        raise RuntimeError("no validated SAFE anchor found")
    safe_tok, safe, safe_sha = max(safe_candidates, key=lambda x: x[0])

    continuity_candidates = journal_candidates[:]
    if v4 is not None:
        continuity_candidates.append((v4[2], v4[0], v4[1]))
    continuity_candidates.append((safe_tok, safe, safe_sha))
    cont_tok, cont, cont_sha = max(continuity_candidates, key=lambda x: x[0])
    return safe, safe_sha, safe_tok, cont, cont_sha, cont_tok, state


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--target-tokens", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--max-wall-minutes", type=int, default=4200)
    a = ap.parse_args()

    assets, report, scratch = map(lambda x: Path(x).resolve(), (a.assets, a.report, a.scratch))
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report not empty")
    if scratch.exists():
        raise FileExistsError("scratch exists")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    # The journal is intentionally outside the request scratch directory. On the self-hosted
    # runner this remains on D: across workflow runs, while moves from scratch are same-volume
    # renames instead of multi-hundred-MB copies.
    journal = scratch.parent / "_continuity_v5"
    journal.mkdir(parents=True, exist_ok=True)

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_root = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    main_data, cases = main_root / "train", main_root / "bilingual_holdout.json"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    state_path = assets / STATE_FILE
    usable_manifest = assets / USABLE_MANIFEST
    started = time.monotonic()

    safe_anchor, safe_sha, safe_tokens, train_anchor, train_sha, train_tokens, state = discover_initial(assets, journal)
    pressure = int(state.get("failure_pressure", 0))
    recipe_index = int(state.get("recipe_index", 1))
    preferred_alpha = float(state.get("preferred_alpha", 0.05))
    last_full_audit = int(state.get("last_full_audit_tokens", safe_tokens))
    accepted_stages = int(state.get("accepted_stages", 0))
    history: list[dict] = []

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

    cached = state.get("safe_baseline")
    if cached and cached.get("checkpoint_sha256") == safe_sha and int(cached.get("checkpoint_processed_tokens", -1)) == safe_tokens:
        safe_baseline = cached["metrics"]
    else:
        tag = f"v5_safe_baseline_{safe_tokens}"
        core = eval_core(safe_anchor, tag, safe_tokens)
        safe_baseline = {**core, "bilingual": eval_bilingual(safe_anchor, tag, safe_tokens, core["dev"])}

    def baseline_smoke(rotation: int, tag: str) -> tuple[dict[str, dict], float]:
        selected: dict[str, dict] = {}
        per_cat: dict[str, list[dict]] = {}
        for row in safe_baseline["dev"]["rows"]:
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

    def train_stage(recipe: dict, tag: str, block_tokens: int) -> tuple[Path, dict]:
        out = scratch / tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(train_anchor), "--anchor-sha", train_sha,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", recipe["mode"], "--new-tokens", str(block_tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",
               "--lr", str(recipe["lr"]), "--lr-end", str(recipe["lr_end"]),
               "--replay-boost", str(recipe["replay_boost"]),
               "--snapshot-interval-tokens", str(STAGE_TOKENS),
               "--resume-save-interval-tokens", str(STAGE_TOKENS)]
        if recipe["mode"] == "replay_only":
            cmd.append("--fast-replay-only")
        print(json.dumps({"event": "v5_train_start", "tag": tag, "from_tokens": train_tokens,
                          "block_tokens": block_tokens, "recipe": recipe}), flush=True)
        subprocess.run(cmd, check=True)
        summary = read_json(out / "summary.json")
        if not summary.get("complete") or not summary.get("frozen_weights_unchanged"):
            raise RuntimeError("training contract failed")
        raw = out / "resume.pt"
        if not valid_checkpoint(raw, summary.get("checkpoint_sha256"), ck_tokens(raw)):
            raise RuntimeError("invalid endpoint checkpoint")
        rates = [float(r["tokens_per_second"]) for r in train_rows(out / "metrics.jsonl")[5:]
                 if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
        perf = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates),
                "p10_tps": p10(rates), "peak_tps": max(rates)} if rates else {}
        return raw, perf

    def make_interp(raw: Path, alpha: float, tag: str) -> Path:
        snap = scratch / f"{tag}.pt"
        exec_script("interpolate_recovery_checkpoint.py", "--anchor", train_anchor, "--candidate", raw,
                    "--out", snap, "--alpha", alpha)
        return snap

    def health_screen(raw: Path, alpha: float, tag: str, rotation: int,
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
        # Continuity is deliberately less strict than SAFE. It blocks abrupt damage but
        # does not turn one discrete benchmark flip into a 100M+ rollback.
        val_delta = val - base_val
        healthy = math.isfinite(val) and regressions <= 2 and (regressions - gains) <= 1 and val_delta <= 0.08
        result = {"healthy": healthy, "path": str(snap), "tokens": tokens, "alpha": alpha,
                  "regressions": regressions, "gains": gains, "margin_delta": margin_delta,
                  "validation_loss": val, "validation_delta": val_delta,
                  "checkpoint_sha256": file_sha(snap)}
        print(json.dumps({"event": "v5_health_screen", **result}), flush=True)
        return result

    def full_audit(checkpoint: Path, tag: str, tokens: int) -> dict:
        core = eval_core(checkpoint, tag, tokens)
        core_ok, failures = core_gate(safe_baseline, core)
        bi = eval_bilingual(checkpoint, tag, tokens, core["dev"])
        bi_ok = bi_gate(safe_baseline["bilingual"], bi, core["dev"])
        category_improved = any(
            float(core["dev"]["categories"][c]["accuracy_pct"]) >
            float(safe_baseline["dev"]["categories"][c]["accuracy_pct"])
            for c in CATEGORIES
        )
        improved = (float(core["validation"]["validation_loss"]) < float(safe_baseline["validation"]["validation_loss"])
                    or category_improved
                    or float(bi["overall_accuracy_pct"]) > float(safe_baseline["bilingual"]["overall_accuracy_pct"]))
        safe_pass = core_ok and bi_ok and improved

        # A wider audit envelope controls the forward-recovery recipe. Failing this does
        # not rewind continuity; it only makes the next stages more retentive.
        overall_drop = float(safe_baseline["dev"]["overall_accuracy_pct"]) - float(core["dev"]["overall_accuracy_pct"])
        worst_cat_drop = max(
            float(safe_baseline["dev"]["categories"][c]["accuracy_pct"]) - float(core["dev"]["categories"][c]["accuracy_pct"])
            for c in CATEGORIES
        )
        val_worse = float(core["validation"]["validation_loss"]) - float(safe_baseline["validation"]["validation_loss"])
        bi_drop = float(safe_baseline["bilingual"]["overall_accuracy_pct"]) - float(bi["overall_accuracy_pct"])
        catastrophic = overall_drop > 2.0 or worst_cat_drop > 10.0 or val_worse > 0.12 or bi_drop > 2.0
        return {"safe_pass": safe_pass, "core_ok": core_ok, "bilingual_ok": bi_ok,
                "strict_improvement": improved, "failures": failures, "catastrophic": catastrophic,
                "overall_drop_pp": overall_drop, "worst_category_drop_pp": worst_cat_drop,
                "validation_delta": val_worse, "bilingual_drop_pp": bi_drop,
                "metrics": {**core, "bilingual": bi}, "checkpoint_sha256": file_sha(checkpoint),
                "tokens": tokens}

    def persist_checkpoint(candidate: Path, sha: str, tokens: int) -> Path:
        if not valid_checkpoint(candidate, sha, tokens):
            raise RuntimeError("candidate identity invalid before journal commit")
        dest = journal / f"continuity_{tokens:012d}_{sha[:12]}.pt"
        if dest.exists():
            if not valid_checkpoint(dest, sha, tokens):
                raise RuntimeError("journal name collision")
            return dest
        # Same D: volume as request scratch: this is normally an atomic/cheap rename.
        os.replace(candidate, dest)
        if not valid_checkpoint(dest, sha, tokens):
            raise RuntimeError("journal commit identity mismatch")
        return dest

    def prune_journal(keep: int = 3) -> None:
        rows: list[tuple[int, Path]] = []
        for p in journal.glob("continuity_*.pt"):
            try:
                rows.append((ck_tokens(p), p))
            except Exception:
                continue
        rows.sort(reverse=True)
        for _, p in rows[keep:]:
            try:
                p.unlink()
            except OSError:
                pass

    def persist_state(status: str, audit: dict | None = None) -> None:
        atomic_json(state_path, {
            "schema": 6,
            "controller": "neral350-scale-v5-monotonic",
            "safe": {"path": str(safe_anchor), "sha256": safe_sha, "tokens": safe_tokens},
            "continuity": {"path": str(train_anchor), "sha256": train_sha, "tokens": train_tokens,
                           "status": status, "not_safe_release": train_sha != safe_sha},
            "target_tokens": a.target_tokens,
            "stage_tokens": STAGE_TOKENS,
            "full_audit_interval": FULL_AUDIT_INTERVAL,
            "max_software_rollback_tokens": STAGE_TOKENS,
            "failure_pressure": pressure,
            "recipe_index": recipe_index,
            "preferred_alpha": preferred_alpha,
            "last_full_audit_tokens": last_full_audit,
            "accepted_stages": accepted_stages,
            "safe_baseline": {"checkpoint_sha256": safe_sha,
                              "checkpoint_processed_tokens": safe_tokens, "metrics": safe_baseline},
            "last_audit": audit,
            "updated_by_request": a.request_id,
            "history_tail": history[-8:],
        })
        atomic_json(usable_manifest, {
            "schema": 1,
            "status": "SAFE" if train_sha == safe_sha else "CONTINUITY_CANDIDATE",
            "checkpoint": str(train_anchor), "checkpoint_sha256": train_sha,
            "processed_tokens": train_tokens, "safe_tokens": safe_tokens,
            "warning": None if train_sha == safe_sha else "Continuity checkpoint passed 5M health gates but is not a strict SAFE release."
        })

    # Materialize an initial state immediately, so a process interruption never loses the
    # discovered continuity anchor.
    persist_state("RESUMED")

    round_no = 0
    consecutive_rejects = 0
    while (train_tokens < a.target_tokens and round_no < a.max_rounds and
           (time.monotonic() - started) / 60 < a.max_wall_minutes):
        round_no += 1
        before_tokens = train_tokens
        remaining = a.target_tokens - train_tokens
        block = valid_budget(min(STAGE_TOKENS, remaining))
        if block < 100_000:
            break

        rotation = accepted_stages % 4
        base_rows, base_val = baseline_smoke(rotation, f"v5_r{round_no:04d}")
        recipe = scaled_recipe(RECIPES[min(recipe_index, len(RECIPES) - 1)], pressure)
        tag = f"v5_r{round_no:04d}_{recipe['name']}_{block}"
        raw, perf = train_stage(recipe, tag, block)

        # First try the preferred interpolation; if it fails the cheap guard, use a gentler
        # blend from the same raw endpoint before spending another training stage.
        alphas = [max(0.006, min(0.10, preferred_alpha)), max(0.006, min(0.10, preferred_alpha * 0.5))]
        screened: list[dict] = []
        accepted: dict | None = None
        for idx, alpha in enumerate(dict.fromkeys(round(x, 8) for x in alphas), 1):
            row = health_screen(raw, alpha, f"{tag}_a{idx}", rotation, base_rows, base_val)
            row["throughput"] = perf
            screened.append(row)
            if row["healthy"]:
                accepted = row
                break

        if accepted is None:
            # Only this 5M stage is discarded. The previous continuity anchor remains the
            # live anchor. Recovery happens forward with a gentler recipe; never jump to SAFE.
            consecutive_rejects += 1
            pressure = min(5, pressure + 1)
            recipe_index = min(len(RECIPES) - 1, recipe_index + 1)
            preferred_alpha = max(0.006, preferred_alpha * 0.70)
            history.append({"round": round_no, "status": "STAGE_REJECTED_NO_REWIND",
                            "continuity_tokens": train_tokens, "discarded_stage_tokens": block,
                            "max_rollback_tokens": block, "recipe": recipe,
                            "screens": screened})
            atomic_json(report / "progress.json", history)
            persist_state("STAGE_REJECTED")
            if consecutive_rejects >= 4:
                print(json.dumps({"event": "v5_stop", "reason": "four consecutive 5M stage rejects",
                                  "continuity_tokens": train_tokens}), flush=True)
                break
            continue

        consecutive_rejects = 0
        committed = persist_checkpoint(Path(accepted["path"]), accepted["checkpoint_sha256"], int(accepted["tokens"]))
        train_anchor = committed
        train_sha = accepted["checkpoint_sha256"]
        train_tokens = int(accepted["tokens"])
        accepted_stages += 1
        preferred_alpha = max(0.006, min(0.10, float(accepted["alpha"]) * 1.05))
        pressure = max(0, pressure - 1)
        if recipe_index > 0 and not accepted["regressions"] and accepted["validation_delta"] <= 0.02:
            recipe_index -= 1

        audit_result: dict | None = None
        due_audit = (train_tokens - last_full_audit >= FULL_AUDIT_INTERVAL) or train_tokens >= a.target_tokens
        if due_audit:
            audit_result = full_audit(train_anchor, f"v5_audit_{train_tokens}", train_tokens)
            last_full_audit = train_tokens
            if audit_result["safe_pass"]:
                destdir = assets / "runs" / f"neral350_scalev5_safe_{train_tokens}"
                destdir.mkdir(parents=True, exist_ok=True)
                dest = destdir / "resume.pt"
                if not dest.exists():
                    shutil.copy2(train_anchor, dest)
                if file_sha(dest) != train_sha:
                    raise RuntimeError("SAFE copy hash mismatch")
                safe_anchor, safe_sha, safe_tokens = dest, train_sha, train_tokens
                safe_baseline = audit_result["metrics"]
                recipe_index = max(0, recipe_index - 1)
                pressure = max(0, pressure - 1)
                audit_status = "SAFE_PROMOTION"
            else:
                # Forward recovery: keep the accepted continuity point. A bad audit never
                # rewinds prior accepted stages. More severe drift selects projected rescue.
                if audit_result["catastrophic"]:
                    recipe_index = 2
                    pressure = min(5, pressure + 2)
                    preferred_alpha = max(0.006, preferred_alpha * 0.65)
                    audit_status = "AUDIT_ALERT_FORWARD_RESCUE"
                else:
                    recipe_index = max(1, recipe_index)
                    pressure = min(5, pressure + 1)
                    audit_status = "AUDIT_NOT_SAFE_CONTINUE"
            public_audit = {k: v for k, v in audit_result.items() if k != "metrics"}
            public_audit["status"] = audit_status
        else:
            public_audit = None

        history.append({"round": round_no, "status": "CONTINUITY_COMMIT",
                        "tokens_before": before_tokens, "tokens_after": train_tokens,
                        "stage_tokens": train_tokens - before_tokens,
                        "safe_tokens": safe_tokens, "recipe": recipe,
                        "health": {k: v for k, v in accepted.items() if k != "path"},
                        "audit": public_audit})
        atomic_json(report / "progress.json", history)
        persist_state("SAFE" if train_sha == safe_sha else "CONTINUITY", public_audit)
        prune_journal()

    result = {
        "schema": 6,
        "controller": "neral350-scale-v5-monotonic",
        "request_id": a.request_id,
        "target_tokens": a.target_tokens,
        "final_safe_tokens": safe_tokens,
        "final_safe_sha256": safe_sha,
        "final_safe_checkpoint": str(safe_anchor),
        "final_train_tokens": train_tokens,
        "final_train_sha256": train_sha,
        "final_train_checkpoint": str(train_anchor),
        "final_status": "SAFE" if train_sha == safe_sha else "CONTINUITY_CANDIDATE",
        "stage_tokens": STAGE_TOKENS,
        "max_software_rollback_tokens": STAGE_TOKENS,
        "full_audit_interval_tokens": FULL_AUDIT_INTERVAL,
        "accepted_stages": accepted_stages,
        "rounds_this_run": round_no,
        "safe_gate_unchanged": True,
        "audit_failure_recovery": "FORWARD_ONLY_NO_REWIND",
        "journal_generations_kept": 3,
        "wall_minutes": (time.monotonic() - started) / 60,
    }
    atomic_json(report / "selection.json", result)
    persist_state(result["final_status"])
    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
