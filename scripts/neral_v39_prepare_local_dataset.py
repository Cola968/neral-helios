"""CPU-only, idempotent merge of the committed V3.9 seed into gitignored data/."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from scripts.neral_qlora_verified_v39 import load_verified_dataset


def merge_seed(seed: Path, target: Path) -> int:
    source = load_verified_dataset(seed)
    if target.exists() and target.is_symlink():
        raise ValueError("Refuse symlink target")
    live = load_verified_dataset(target) if target.is_file() else []
    merged = list(live)
    keys = {x["meta"]["instruction_sha256"] for x in live}
    for item in source:
        key = item["meta"]["instruction_sha256"]
        if key not in keys:
            merged.append(item)
            keys.add(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement leaves previously logged verified examples intact.
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=target.parent,
                                     prefix=".neral-training-",delete=False) as handle:
        temp=Path(handle.name)
        for row in merged:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp,target)
    finally:
        if temp.exists():
            temp.unlink()
    return len(merged)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=Path,default=Path("datasets/neral_v39_seed97.jsonl"))
    p.add_argument("--out",type=Path,default=Path("data/verified_training_dataset.jsonl"))
    args=p.parse_args(argv)
    count=merge_seed(args.seed,args.out)
    print(json.dumps({"status":"local_dataset_ready_no_gpu","records":count,
                      "gitignored":True}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
