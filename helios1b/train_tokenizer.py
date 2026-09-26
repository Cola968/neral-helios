from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end|>",
]


def iter_text_files(paths: list[Path]) -> Iterable[str]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        text = obj
                    elif isinstance(obj, dict):
                        text = obj.get("text")
                    else:
                        text = None
                    if not isinstance(text, str):
                        raise ValueError(
                            f"{path}:{line_no}: JSONL row must be a string "
                            "or contain a string field named 'text'"
                        )
                    if text:
                        yield text
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line:
                        yield line


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=True,
    )
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def train_tokenizer(
    inputs: list[Path],
    output: Path,
    vocab_size: int = 32_768,
    min_frequency: int = 2,
) -> None:
    if vocab_size < 1_024:
        raise ValueError("vocab_size is unrealistically small for Helios")
    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        max_token_length=64,
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_text_files(inputs), trainer=trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab > vocab_size:
        raise RuntimeError(
            f"trained vocabulary {actual_vocab} exceeds target {vocab_size}"
        )

    smoke = [
        "Berlin ist eine Stadt.",
        "The model should preserve English text.",
        "def f(x: int) -> int:\n    return x * x\n",
        "E = mc²; \\int_0^1 x^2 dx = 1/3",
        "emoji ✓ 🚀 äöü ÄÖÜ ß",
    ]
    for sample in smoke:
        decoded = tokenizer.decode(
            tokenizer.encode(sample, add_special_tokens=False).ids,
            skip_special_tokens=False,
        )
        if decoded != sample:
            raise RuntimeError(
                f"tokenizer round-trip failed: {sample!r} -> {decoded!r}"
            )

    meta = {
        "format": "Neral Helios tokenizer",
        "type": "byte-level BPE",
        "target_vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab,
        "min_frequency": min_frequency,
        "special_tokens": SPECIAL_TOKENS,
        "inputs": [str(p) for p in inputs],
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the Neral Helios 32k ByteLevel BPE tokenizer."
    )
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="UTF-8 .txt or .jsonl file; repeat for multiple corpus shards",
    )
    ap.add_argument(
        "--output",
        default="artifacts/helios1b/tokenizer.json",
    )
    ap.add_argument("--vocab-size", type=int, default=32_768)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    train_tokenizer(
        [Path(p) for p in args.input],
        Path(args.output),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
