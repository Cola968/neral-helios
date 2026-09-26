from pathlib import Path

from tokenizers import Tokenizer

from helios1b.benchmark_tokenizer import evaluate_text
from helios1b.train_tokenizer import train_tokenizer


def test_tokenizer_training_roundtrip(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "Berlin ist eine Stadt mit öffentlichem Verkehr.\n"
            "The model learns from English technical text.\n"
            "def square(x):\n    return x * x\n"
            "E = mc² und int_0^1 x^2 dx = 1/3.\n"
        )
        * 20,
        encoding="utf-8",
    )
    out = tmp_path / "tokenizer.json"
    train_tokenizer([corpus], out, vocab_size=1024, min_frequency=1)

    tokenizer = Tokenizer.from_file(str(out))
    samples = [
        "Grüße aus Berlin ✓",
        "def f(x):\n    return x + 1\n",
        "E = mc² 🚀",
    ]
    for text in samples:
        result = evaluate_text(tokenizer, text)
        assert result["roundtrip_ok"]
        assert result["tokens"] > 0
        assert result["bytes_per_token"] > 0
