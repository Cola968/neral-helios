from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np

from tokenizer import ByteTokenizer


def iter_documents(path: Path, text_field: str) -> Iterator[str]:
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No .txt or .jsonl files found under {path}")

    for file in files:
        if file.suffix.lower() == ".jsonl":
            with file.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get(text_field)
                    if isinstance(text, str) and text.strip():
                        yield text
        else:
            # Paragraph-oriented streaming keeps memory use small for huge text files.
            buf = []
            with file.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        buf.append(line.rstrip("\n"))
                    elif buf:
                        yield "\n".join(buf)
                        buf.clear()
                if buf:
                    yield "\n".join(buf)


def stable_is_val(text: str, val_fraction: float) -> bool:
    digest = hashlib.blake2b(text[:4096].encode("utf-8", errors="ignore"), digest_size=8).digest()
    value = int.from_bytes(digest, "little") / 2**64
    return value < val_fraction


def append_ids(handle, ids, buffer_limit=1_000_000):
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(handle)
    return len(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Text/JSONL file or directory")
    ap.add_argument("--out", default="data")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--val-fraction", type=float, default=0.005)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = ByteTokenizer()

    counts = {"train": 0, "val": 0, "docs_train": 0, "docs_val": 0}
    with (out / "train.bin").open("wb") as f_train, (out / "val.bin").open("wb") as f_val:
        for i, text in enumerate(iter_documents(src, args.text_field), start=1):
            ids = tok.encode(text, bos=True, eos=True)
            split = "val" if stable_is_val(text, args.val_fraction) else "train"
            target = f_val if split == "val" else f_train
            counts[split] += append_ids(target, ids)
            counts[f"docs_{split}"] += 1
            if i % 10000 == 0:
                print(f"processed {i:,} docs | train tokens={counts['train']:,} | val={counts['val']:,}")

    (out / "meta.json").write_text(json.dumps({
        "dtype": "uint16",
        "vocab_size": tok.vocab_size,
        "tokenizer": "Neral byte260",
        **counts,
    }, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    if counts["val"] < 10_000:
        print("WARNING: validation split is very small; use more data or a larger --val-fraction.")


if __name__ == "__main__":
    main()
