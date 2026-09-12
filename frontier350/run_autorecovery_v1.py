from __future__ import annotations

import argparse, json, math, shutil, statistics, subprocess, sys
from pathlib import Path
from recovery_identity import file_sha

ANCHOR_SHA = "b780d089a72a54dea94b4669b696924259c24add909b010ee62670fe0c066894"
ANCHOR_TOKENS = 1_009_003_584
TARGET_TOKENS = 1_010_003_584
FAST_ALPHAS = (0.05, 0.025, 0.0125, 0.00625)
LEGACY_ALPHAS = (0.025, 0.0125, 0.00625, 0.003125)


def run(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    a = ap.parse_args()
    assets, report, scratch = map(lambda p: Path(p).resolve(), (a.assets, a.report, a.scratch))
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report not empty")
    if scratch.exists():
        raise FileExistsError("scratch already exists")
    report.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True)

    anchor = assets / "runs/neral350_autolearn_neral-autolearn-v1-20260912-r2_r01_selected/resume.pt"
    if not anchor.is_file() or file_sha(anchor) != ANCHOR_SHA:
        raise ValueError("safe 1.009B anchor missing or changed")
    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    cases = main_data / "bilingual_holdout.json"

    def evaluate(checkpoint: Path, tag: str, tokens: int) -> dict:
        val, dev, bi = report/f"{tag}_validation.json", report/f"{tag}_dev.json", report/f"{tag}_bilingual.json"
        run("eval_validation_loss350.py", "--checkpoint", checkpoint, "--valid-data", valid, "--out", val, "--seq", 512, "--batch", 4, "--batches", 32, "--seed", 351337)
        run("eval_recovery_dev_v2.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer, "--out", dev, "--expected-tokens", tokens)
        run("eval_bilingual_holdout_v1.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer, "--cases", cases, "--out", bi, "--expected-tokens", tokens)
        out = {"validation": load(val), "dev": load(dev), "bilingual": load(bi)}
        sha = file_sha(checkpoint)
        for x in out.values():
            if x.get("checkpoint_sha256") != sha or x.get("checkpoint_processed_tokens") != tokens:
                raise ValueError("evaluation identity mismatch")
        return out

    baseline = evaluate(anchor, "anchor", ANCHOR_TOKENS)

    def train(tag: str, fast: bool, pack: int, tokens: int) -> Path:
        out = scratch/tag
        args = ["--anchor", anchor, "--anchor-sha", ANCHOR_SHA, "--main-data", main_data/"train", "--replay", replay,
                "--out", out, "--mode", "replay_only", "--new-tokens", tokens, "--batch-size", 4,
                "--microbatches", 12, "--main-pack", pack, "--compile-mode", "default"]
        if fast: args.append("--fast-replay-only")
        run("train_guarded_v9.py", *args)
        s = load(out/"summary.json")
        if not s.get("complete") or s.get("processed_tokens") != ANCHOR_TOKENS + tokens or not s.get("frozen_weights_unchanged"):
            raise ValueError("training contract failed")
        shutil.copyfile(out/"summary.json", report/f"{tag}_summary.json")
        shutil.copyfile(out/"metrics.jsonl", report/f"{tag}_metrics.jsonl")
        return out/"resume.pt"

    # A/B semantic diagnostic on identical batch geometry and seed.
    ab_tokens = 245_760
    legacy_ab = train("ab_legacy_pack1", False, 1, ab_tokens)
    fast_ab = train("ab_fast_pack1", True, 1, ab_tokens)
    ls, fs = load(legacy_ab.parent/"summary.json"), load(fast_ab.parent/"summary.json")
    def rows(path: Path) -> list[dict]:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("event") == "train"]
    lr, fr = rows(legacy_ab.parent/"metrics.jsonl"), rows(fast_ab.parent/"metrics.jsonl")
    max_loss_delta = max(abs(x["loss"]-y["loss"]) for x,y in zip(lr,fr))
    max_grad_rel = max(abs(x["grad_norm"]-y["grad_norm"])/max(abs(x["grad_norm"]),1e-9) for x,y in zip(lr,fr))
    ab_ok = ls["batch_chain"] == fs["batch_chain"] and max_loss_delta <= 0.03 and max_grad_rel <= 0.15
    ab = {"batch_chain_equal": ls["batch_chain"] == fs["batch_chain"], "max_loss_delta": max_loss_delta,
          "max_grad_relative_delta": max_grad_rel, "semantic_diagnostic_pass": ab_ok}
    (report/"ab_semantics.json").write_text(json.dumps(ab, indent=2))
    print(json.dumps({"event":"ab_semantics", **ab}), flush=True)

    attempts = []
    variants = [("fast_pack4", True, 4, FAST_ALPHAS)] if ab_ok else []
    variants.append(("legacy_pack1", False, 1, LEGACY_ALPHAS))
    selected = None

    for variant, fast, pack, alphas in variants:
        raw = train(variant, fast, pack, 1_000_000)
        mrows = rows(raw.parent/"metrics.jsonl")
        full = [r["tokens_per_second"] for r in mrows if r.get("tokens_per_update") == 24576][4:]
        throughput = {"median_tps": statistics.median(full), "p10_tps": sorted(full)[max(0, math.ceil(.1*len(full))-1)], "peak_tps": max(full)} if full else {}
        candidates = []
        for alpha in alphas:
            tag = f"{variant}_a{str(alpha).replace('.', 'p')}"
            snap = scratch/f"{tag}.pt"
            run("interpolate_recovery_checkpoint.py", "--anchor", anchor, "--candidate", raw, "--out", snap, "--alpha", alpha)
            measured = evaluate(snap, tag, TARGET_TOKENS)
            gate = report/f"{tag}_legacy_gate.json"
            run("select_recovery_lr_pilot.py", "--baseline-dev", report/"anchor_dev.json", "--baseline-validation", report/"anchor_validation.json",
                "--candidate", tag, 7.5e-5, report/f"{tag}_dev.json", report/f"{tag}_validation.json", "--out", gate)
            legacy = load(gate)
            bi_ok = strict_bilingual(baseline["bilingual"], measured["bilingual"], measured["dev"])
            improved = measured["validation"]["validation_loss"] < baseline["validation"]["validation_loss"] or measured["bilingual"]["overall_accuracy_pct"] > baseline["bilingual"]["overall_accuracy_pct"] or any(
                measured["dev"]["categories"][k]["accuracy_pct"] > baseline["dev"]["categories"][k]["accuracy_pct"] for k in baseline["dev"]["categories"])
            passed = bool(legacy.get("selection_allowed")) and bi_ok and improved
            row = {"variant": variant, "alpha": alpha, "pass": passed, "legacy_gate": bool(legacy.get("selection_allowed")),
                   "legacy_failures": legacy.get("candidates", [{}])[0].get("failures", []), "bilingual_gate": bi_ok,
                   "strict_improvement": improved, "validation_loss": measured["validation"]["validation_loss"],
                   "overall_pct": measured["dev"]["overall_accuracy_pct"],
                   "categories": {k:v["accuracy_pct"] for k,v in measured["dev"]["categories"].items()},
                   "bilingual_pct": measured["bilingual"]["overall_accuracy_pct"],
                   "de_to_en_pct": measured["bilingual"]["directions"]["de_to_en"]["accuracy_pct"],
                   "en_to_de_pct": measured["bilingual"]["directions"]["en_to_de"]["accuracy_pct"],
                   "checkpoint_sha256": file_sha(snap), "throughput": throughput, "path": str(snap)}
            candidates.append(row)
            if not passed: snap.unlink(missing_ok=True)
            (report/"progress.json").write_text(json.dumps(attempts + [{"variant":variant,"candidates":candidates}], indent=2))
        attempts.append({"variant":variant,"throughput":throughput,"candidates":candidates})
        passing = [c for c in candidates if c["pass"]]
        if passing:
            selected = min(passing, key=lambda c:(c["validation_loss"], -c["bilingual_pct"], -c["alpha"]))
            break

    promoted = None
    if selected:
        destdir = assets/"runs"/f"neral350_autorecovery_{a.request_id}_selected"
        if destdir.exists(): raise FileExistsError("promotion destination exists")
        destdir.mkdir(parents=True)
        dest = destdir/"resume.pt"
        shutil.copy2(selected["path"], dest)
        if file_sha(dest) != selected["checkpoint_sha256"]: raise RuntimeError("promotion hash mismatch")
        promoted = str(dest)

    result = {"schema":1,"controller":"neral350-autorecovery-v1","request_id":a.request_id,"anchor_sha256":ANCHOR_SHA,
              "anchor_tokens":ANCHOR_TOKENS,"target_tokens":TARGET_TOKENS,"ab_semantics":ab,"attempts":attempts,
              "success": selected is not None,"selected":selected,"promoted_checkpoint":promoted,
              "safe_anchor_preserved_if_failure":True,"benchmarks_used_for_training":False,"final_audit_pending":True}
    (report/"selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"event":"final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)

if __name__ == "__main__": main()
