from __future__ import annotations

"""V7.2-only checkpoint I/O fixes.

V4 defines --resume-save-interval-tokens=0 as the legacy full-resume cadence
(every 20 updates). V7 intended compact 1M exact recovery states to replace
those interval full checkpoints, but passed 0 and therefore accidentally kept
legacy full saves alive.

V7.2 keeps exact 1M compact recovery unchanged, writes a full model endpoint
only at 5M, and omits redundant FP32 masters/AdamW state from that successful
endpoint. The latest <=1M compact recovery remains the exact optimizer/RNG
resume point if an endpoint write is interrupted. The redundant initial full
resume is also skipped under V7 exact recovery; a failure before the first 1M
snapshot therefore loses <1M tokens and falls back to the accepted anchor.

V5/V6 source and fallback behavior are unchanged because this patch activates
only after the V7 recovery arguments have been installed.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "run_scale_v5.py"
TRAINER = ROOT / "train_guarded_v9.py"

CTRL_OLD = (
    '               "--snapshot-interval-tokens", "0",\n'
    '               "--resume-save-interval-tokens", "0",\n'
    '               "--v7-recovery-interval-tokens", str(V7_RECOVERY_INTERVAL_TOKENS)]\n'
)
CTRL_NEW = (
    '               "--snapshot-interval-tokens", "0",\n'
    '               "--resume-save-interval-tokens", str(STAGE_TOKENS),\n'
    '               "--v7-recovery-interval-tokens", str(V7_RECOVERY_INTERVAL_TOKENS)]\n'
)

INITIAL_OLD = "    if not args.resume:save()\n"
INITIAL_NEW = "    if not args.resume and not args.v7_recovery_interval_tokens:save()\n"

ENDPOINT_OLD = '''        if args.resume_save_interval_tokens:
            if done==args.new_tokens or done>=next_resume_save:
                save()
                while next_resume_save and next_resume_save<=done:next_resume_save+=args.resume_save_interval_tokens
        elif step%20==0 or done==args.new_tokens:
            save()
'''
ENDPOINT_NEW = '''        if args.resume_save_interval_tokens:
            if done==args.new_tokens or done>=next_resume_save:
                if done==args.new_tokens and args.v7_recovery_interval_tokens:
                    save(include_training_state=False)
                else:
                    save()
                while next_resume_save and next_resume_save<=done:next_resume_save+=args.resume_save_interval_tokens
        elif step%20==0 or done==args.new_tokens:
            save()
'''


def patch_controller(text: str) -> str:
    if CTRL_NEW in text:
        return text
    if text.count(CTRL_OLD) != 1:
        raise RuntimeError(f"V7.2 checkpoint cadence anchor mismatch: {text.count(CTRL_OLD)}")
    if "STAGE_TOKENS = 5_000_000" not in text:
        raise RuntimeError("unexpected stage token contract")
    if "V7_RECOVERY_INTERVAL_TOKENS = 1_000_000" not in text:
        raise RuntimeError("unexpected V7 recovery cadence")
    return text.replace(CTRL_OLD, CTRL_NEW, 1)


def patch_trainer(text: str) -> str:
    if "--v7-recovery-interval-tokens" not in text:
        raise RuntimeError("V7 exact recovery argument missing")
    if INITIAL_NEW not in text:
        if text.count(INITIAL_OLD) != 1:
            raise RuntimeError(f"V7.2 initial-save anchor mismatch: {text.count(INITIAL_OLD)}")
        text = text.replace(INITIAL_OLD, INITIAL_NEW, 1)
    if ENDPOINT_NEW not in text:
        if text.count(ENDPOINT_OLD) != 1:
            raise RuntimeError(f"V7.2 endpoint-save anchor mismatch: {text.count(ENDPOINT_OLD)}")
        text = text.replace(ENDPOINT_OLD, ENDPOINT_NEW, 1)
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

    print(
        "V7.2 checkpoint I/O fixed: 1M exact recovery retained; legacy interval full saves disabled; "
        "initial redundant full save skipped; 5M endpoint model checkpoint omits redundant optimizer/master state"
    )


if __name__ == "__main__":
    main()
