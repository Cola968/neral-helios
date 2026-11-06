"""Generate a pinned, one-run NERAL v3 R32 trainer.

Never modify the original trainer, run training or create a checkpoint here.
The user must authorize the separate Windows workflow before any GPU work.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from scripts.neral_rank32_copy import (
    patched_trainer, EXPECTED_SOURCE_GIT_BLOB, PatchError,
)

AUTH = "TRAIN_NERAL_V3_R32_120_STEPS"
REPLACEMENTS = [
    (
        '    if args.lora_rank == 32 and args.steps != 3:\n'
        '        parser.error("LoRA rank 32 is limited to the authorized 3-step smoke; longer training needs separate approval")\n',
        '    if (args.lora_rank != 32 or args.steps != 120 or\n'
        '        os.environ.get("NERAL_V3_TRAIN_AUTHORIZATION") != "TRAIN_NERAL_V3_R32_120_STEPS"):\n'
        '        parser.error("Unapproved v3 training configuration")\n',
    ),
    ('learning_rate=2e-4, bf16=True', 'learning_rate=5e-5, bf16=True'),
    ('dataloader_num_workers=0, seed=42,', 'dataloader_num_workers=0, seed=6112026,'),
    (
        '            report["optimizer_step_seconds"].append(time.perf_counter()-self._step_start)\n',
        '            report["optimizer_step_seconds"].append(time.perf_counter()-self._step_start)\n'
        '            free,total=torch.cuda.mem_get_info(0)\n'
        '            gpu_used=total-free\n'
        '            reserved=torch.cuda.memory_reserved(0)\n'
        '            report["last_step_gpu_used_bytes"]=gpu_used\n'
        '            if max(gpu_used,reserved)>=int(11.5 * 1024 ** 3):\n'
        '                raise RuntimeError("NERAL VRAM safety stop at 11.5 GiB")\n',
    ),
]


def patch(original: str) -> str:
    result = patched_trainer(original, expected_blob_sha=EXPECTED_SOURCE_GIT_BLOB)
    for before, after in REPLACEMENTS:
        if result.count(before) != 1:
            raise PatchError("Pinned R32 training modification anchor missing/duplicated")
        result = result.replace(before, after, 1)
    required = [
        "80_740_352", "r=args.lora_rank", "learning_rate=5e-5",
        "optim=\"paged_adamw_8bit\"", "gradient_checkpointing=True",
        "per_device_train_batch_size=1", "gradient_accumulation_steps=4",
        "NERAL_V3_TRAIN_AUTHORIZATION", "max_steps=args.steps",
        "int(11.5 * 1024 ** 3)", "from neral_adapter_candidate_r32 import seal_adapter",
    ]
    for name in required:
        if name not in result:
            raise PatchError("Missing v3 training contract: " + name)
    if "args.lora_rank == 32 and args.steps != 3" in result:
        raise PatchError("Old 3-step gate still present")
    return result


def write_checked(source: Path, target: Path) -> dict:
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 64_000:
        raise PatchError("Original pinned trainer invalid")
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise PatchError("Destination must be a fresh scratch file")
    content = patch(source.read_text(encoding="utf-8"))
    with target.open("x", encoding="utf-8", newline="\n") as out:
        out.write(content)
    return {
        "status": "SOURCE_PATCH_ONLY_NOT_TRAINED",
        "original_git_blob": EXPECTED_SOURCE_GIT_BLOB,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "rank": 32, "steps": 120, "learning_rate": 0.00005,
    }


def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(write_checked(a.source, a.output))


if __name__ == "__main__":
    main()
