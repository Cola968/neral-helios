from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def evaluate_text(tokenizer: Tokenizer, text: str) -> dict:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
    byte_count = len(text.encode("utf-8"))
    token_count = len(encoded.ids)
    return {
        "bytes": byte_count,
        "tokens": token_count,
        "bytes_per_token": byte_count / max(token_count, 1),
        "legacy_byte_tokens": byte_count,
        "compression_vs_legacy_byte": byte_count / max(token_count, 1),
        "roundtrip_ok": decoded == text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        default="artifacts/helios1b/tokenizer.json",
    )
    ap.add_argument(
        "--sample",
        action="append",
        required=True,
        help="category=path, e.g. de=data/eval/de.txt",
    )
    ap.add_argument(
        "--out",
        default="artifacts/helios1b/tokenizer_benchmark.json",
    )
    args = ap.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    report = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.get_vocab_size(),
        "categories": {},
    }

    for spec in args.sample:
        if "=" not in spec:
            raise ValueError("--sample must use category=path")
        category, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        report["categories"][category] = evaluate_text(tokenizer, text)

    if not all(v["roundtrip_ok"] for v in report["categories"].values()):
        raise SystemExit("FAIL: tokenizer round-trip error")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
