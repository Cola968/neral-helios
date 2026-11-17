"""Read-only-model GPU smoke of the opt-in local Luna/Terra/Helios HF runtime.

Only synthetic prompts, frozen weights and existing Docker. No training, model
downloads, personal memory access, or raw chat responses in artifacts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time

from neral_three_tier_runtime import FrozenQwen35HF, ThreeTierRuntime
from scripts.neral_three_tier_runtime_inventory import frozen_v2_full


def run(base4: Path, base9: Path, full: Path) -> dict:
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 4070" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Real isolated RTX4070 required")
    if not frozen_v2_full(full):
        raise RuntimeError("Original frozen v2-full adapter receipt missing")
    low = FrozenQwen35HF(base4)
    high = FrozenQwen35HF(base9, full_root=full)
    runtime = ThreeTierRuntime(low, high)
    stats = {
        "case_count": 0, "luna_warm_s": None, "luna_under_3_s": False,
        "luna_model_unchanged_4b": False,
        "terra_source_python_verified": False,
        "helios_frozen_v2_full_model_generated": False,
        "helios_source_z3_verified": False,
        "switch_back_luna_confirmed": False,
        "helios_adapter_sha256_receipt_checked": True,
        "single_real_rtx4070": True, "models_loaded_sequentially": True,
        "raw_outputs_persisted": False, "personal_chat_data_accessed": False,
        "models_downloaded": False, "training_performed": False,
        "full_browser_end_to_end_proven": False,
    }
    def ask(tier, question):
        start = time.perf_counter()
        reply = runtime.answer([
            {"role": "system", "content": "Antworte auf Deutsch knapp und sachlich."},
            {"role": "user", "content": question}], tier)
        stats["case_count"] += 1
        return reply, round(time.perf_counter() - start, 4)
    try:
        # Two real generations on the unchanged 4B snapshot. The first call
        # includes model loading; only the second is a warm latency sample.
        first, first_s = ask("Luna", "Antworte exakt mit einem Wort: Hallo.")
        second, warm_s = ask("Luna", "Antworte exakt mit einem Wort: Hallo.")
        stats["luna_first_cold_s"] = first_s
        stats["luna_warm_s"] = warm_s
        stats["luna_under_3_s"] = warm_s < 3.0
        stats["luna_model_unchanged_4b"] = (
            first["model"] == second["model"] == "qwen3.5:4b"
            and first.get("adapter_loaded") is False
            and second.get("model_generated") is True)

        # The trusted arithmetic compiler must execute inside real Docker,
        # and its independently parsed source must match the sandbox stdout.
        source, source_s = ask(
            "Terra", "How many positive divisors does the integer 120 have?")
        stats["terra_source_python_verified"] = (
            source.get("verified") is True
            and source.get("proof_backend") == "trusted_source_arithmetic"
            and source.get("text") == "16")
        stats["terra_source_s"] = source_s

        # Force actual 9B PEFT inference rather than accepting a 24-case
        # compiler-only green result as proof of adapter activation.
        answer, high_s = ask(
            "Helios", "Nenne einen einzigen kurzen Unterschied zwischen Stern und Planet.")
        stats["helios_frozen_v2_full_model_generated"] = (
            answer.get("model_generated") is True
            and answer.get("adapter_loaded") is True
            and answer.get("adapter_sha256") == high.adapter_sha256
            and answer.get("model") == "qwen3.5:9b+neral-v2-full")
        stats["helios_model_cold_s"] = high_s

        # Independent, full-source Z3 proof on a known DEV24 development
        # fixture; never a claimed blind generalization result.
        from scripts.neral_1_1_postverify_dev24_cases import build
        case = next(x for x in build() if x["family"] == "graph_coloring_z3")
        proof, z3_s = ask("Helios", case["question"])
        stats["helios_source_z3_verified"] = (
            proof.get("verified") is True and
            proof.get("proof_backend") == "z3_unsat" and
            proof.get("text") == str(case["gold_integer"]))
        stats["helios_source_z3_s"] = z3_s

        # This must genuinely release the 9B model before reloading 4B.
        last, back_s = ask("Luna", "Antworte exakt mit einem Wort: Hallo.")
        stats["switch_back_luna_confirmed"] = (
            last.get("model") == "qwen3.5:4b"
            and last.get("model_generated") is True
            and not high.loaded and low.loaded)
        stats["luna_after_switch_cold_s"] = back_s
        stats["status"] = (
            "GPU_TIERS_VERIFIED"
            if (stats["luna_model_unchanged_4b"]
                and stats["terra_source_python_verified"]
                and stats["helios_frozen_v2_full_model_generated"]
                and stats["helios_source_z3_verified"]
                and stats["switch_back_luna_confirmed"])
            else "GPU_TIERS_NOT_RELEASE_READY")
        return stats
    finally:
        low.unload()
        high.unload()


def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--base4", required=True, type=Path)
    parser.add_argument("--base9", required=True, type=Path)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args=parser.parse_args(argv)
    try:
        stats=run(args.base4,args.base9,args.full)
    except Exception as exc:
        stats={
            "status":"GPU_TIERS_FAILED",
            "error_class":type(exc).__name__,
            "no_training":True,
            "no_model_download":True,
            "raw_outputs_persisted":False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(stats,f,sort_keys=True,indent=2)
    print("NERAL_THREE_TIER_GPU_SCALAR "+json.dumps(stats,sort_keys=True),flush=True)
    return 0 if stats["status"]=="GPU_TIERS_VERIFIED" else 1


if __name__=="__main__":
    raise SystemExit(main())
