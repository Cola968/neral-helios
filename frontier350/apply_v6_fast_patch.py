from __future__ import annotations

"""Execution-only V6 fast patch layered on top of the validated V5 turbo runtime.

This patch does not change model architecture, sampled token counts, optimizer, LR/replay
recipes, 5M continuity stages, health thresholds, SAFE gates, or rollback semantics.
It only removes duplicate deterministic work and lowers host-sync telemetry overhead.

Changes:
- cache the deterministic SAFE 4-batch validation baseline once per controller process
  instead of recomputing it before every 5M stage;
- in fused replay-only training, synchronize/write telemetry and run the expensive full
  trainable-parameter finite scan every 10 updates rather than every 5 updates. Loss and
  gradient finite checks still happen every update, and the endpoint is always checked.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_guarded_v9.py"
CONTROLLER = ROOT / "run_scale_v5.py"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f"V6 fast patch anchor mismatch for {label}: {text.count(old)}")
    return text.replace(old, new, 1)


def patch_controller(text: str) -> str:
    anchor = """        safe_baseline = {**core, \"bilingual\": eval_bilingual(safe_anchor, tag, safe_tokens, core[\"dev\"])}\n\n    def baseline_smoke(rotation: int, tag: str) -> tuple[dict[str, dict], float]:\n"""
    replacement = """        safe_baseline = {**core, \"bilingual\": eval_bilingual(safe_anchor, tag, safe_tokens, core[\"dev\"])}\n\n    # The SAFE checkpoint, validation shard, seed, seq/batch and batch count are identical\n    # for every 5M stage. Compute this deterministic 4-batch reference once per controller\n    # process instead of launching the same evaluator subprocess every round.\n    _baseline_val_cache: float | None = None\n\n    def baseline_smoke(rotation: int, tag: str) -> tuple[dict[str, dict], float]:\n        nonlocal _baseline_val_cache\n"""
    text = once(text, anchor, replacement, "baseline cache declaration")

    old = """        vp = report / f\"{tag}_baseline_smoke_validation.json\"\n        exec_script(\"eval_validation_loss350.py\", \"--checkpoint\", safe_anchor, \"--valid-data\", valid, \"--out\", vp,\n                    \"--seq\", 512, \"--batch\", 4, \"--batches\", 4, \"--seed\", 351337)\n        return selected, float(read_json(vp)[\"validation_loss\"])\n"""
    new = """        if _baseline_val_cache is None:\n            vp = report / f\"{tag}_baseline_smoke_validation.json\"\n            exec_script(\"eval_validation_loss350.py\", \"--checkpoint\", safe_anchor, \"--valid-data\", valid, \"--out\", vp,\n                        \"--seq\", 512, \"--batch\", 4, \"--batches\", 4, \"--seed\", 351337)\n            _baseline_val_cache = float(read_json(vp)[\"validation_loss\"])\n            print(json.dumps({\"event\": \"v6_safe_baseline_cached\", \"validation_loss\": _baseline_val_cache}), flush=True)\n        return selected, _baseline_val_cache\n"""
    return once(text, old, new, "baseline evaluator cache")


def patch_trainer(text: str) -> str:
    text = once(
        text,
        "            check_params=(step%5==0) or (done+total>=args.new_tokens)\n",
        "            check_params=(step%10==0) or (done+total>=args.new_tokens)\n",
        "finite scan cadence",
    )
    text = once(
        text,
        "        metric_now=(not args.fused_replay_main) or step%5==0 or done==args.new_tokens\n",
        "        metric_now=(not args.fused_replay_main) or step%10==0 or done==args.new_tokens\n",
        "telemetry cadence",
    )
    text = text.replace(
        "telemetry cadence (or endpoint). This removes four scalar-sync/file-write stalls\n        # out of five",
        "telemetry cadence (or endpoint). V6 Fast uses ten-update host-sync windows\n        # while retaining per-update loss/gradient checks",
    )
    return text


def main() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    patched_controller = patch_controller(controller)
    if patched_controller != controller:
        CONTROLLER.write_text(patched_controller, encoding="utf-8")

    trainer = TRAINER.read_text(encoding="utf-8")
    patched_trainer = patch_trainer(trainer)
    if patched_trainer != trainer:
        TRAINER.write_text(patched_trainer, encoding="utf-8")

    print("V6 fast patch applied: cached SAFE baseline + 10-update fused host-sync cadence; gates unchanged")


if __name__ == "__main__":
    main()
