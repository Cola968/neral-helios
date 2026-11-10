from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
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

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d ()\-]{7,}\d)(?!\d)")
SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*"
    r"['\"]?[A-Za-z0-9_\-/.+=]{12,}"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    config: str | None
    split: str
    revision: str
    license: str
    share: float
    language: str
    kind: str


# Revisions were already admitted by the earlier Neral Frontier data audit.
# Helios reuses only the immutable source contract, never the old tokenizer
# vocabulary or merges.
SOURCES = (
    SourceSpec(
        "fineweb_edu_en",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "train",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "odc-by",
        0.52,
        "en",
        "education",
    ),
    SourceSpec(
        "fineweb_en",
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        "train",
        "9bb295ddab0e05d785b879661af7260fed5140fc",
        "odc-by",
        0.20,
        "en",
        "general",
    ),
    SourceSpec(
        "fineweb2_de",
        "HuggingFaceFW/fineweb-2",
        "deu_Latn",
        "train",
        "af9c13333eb981300149d5ca60a8e9d659b276b9",
        "odc-by",
        0.13,
        "de",
        "general",
    ),
    SourceSpec(
        "finemath",
        "HuggingFaceTB/finemath",
        "finemath-3plus",
        "train",
        "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "odc-by",
        0.15,
        "multi",
        "math",
    ),
)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def quality_ok(text: str) -> bool:
    if not 160 <= len(text) <= 500_000:
        return False
    if EMAIL_RE.search(text) or PHONE_RE.search(text) or SECRET_RE.search(text):
        return False
    if text.count("�") / max(1, len(text)) > 0.001:
        return False
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(
        1, len(text)
    )
    if printable < 0.985:
        return False
    lines = [line for line in text.split("\n") if line]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35:
        return False
    words = text.split()
    if len(words) >= 80 and len(set(words)) / len(words) < 0.08:
        return False
    return True


def content_hash(text: str) -> str:
    compact = " ".join(text.lower().split())
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def stream_rows(spec: SourceSpec) -> Iterable[dict]:
    from datasets import load_dataset

    kwargs = {
        "path": spec.repo,
        "split": spec.split,
        "streaming": True,
        "revision": spec.revision,
    }
    if spec.config:
        kwargs["name"] = spec.config
    return load_dataset(**kwargs)


def iter_source(
    spec: SourceSpec,
    char_budget: int,
    global_seen: set[str],
) -> Iterable[str]:
    emitted = 0
    for row in stream_rows(spec):
        value = row.get("text", "")
        text = normalize(value if isinstance(value, str) else str(value or ""))
        if not quality_ok(text):
            continue
        digest = content_hash(text)
        if digest in global_seen:
            continue
        global_seen.add(digest)

        remaining = char_budget - emitted
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        if text:
            emitted += len(text)
            yield text
        if emitted >= char_budget:
            break

    if emitted < char_budget:
        raise RuntimeError(
            f"source {spec.name} exhausted at {emitted:,}/{char_budget:,} chars"
        )


def training_iterator(total_chars: int, training_hashes: set[str]) -> Iterable[str]:
    assigned = 0
    for index, spec in enumerate(SOURCES):
        budget = int(total_chars * spec.share)
        if index == len(SOURCES) - 1:
            budget = total_chars - assigned
        assigned += budget
        yield from iter_source(spec, budget, training_hashes)


def heldout_texts(
    spec: SourceSpec,
    training_hashes: set[str],
    examples: int,
) -> Iterable[str]:
    accepted = 0
    local_seen: set[str] = set()
    for row in stream_rows(spec):
        value = row.get("text", "")
        text = normalize(value if isinstance(value, str) else str(value or ""))
        if not quality_ok(text):
            continue
        digest = content_hash(text)
        if digest in training_hashes or digest in local_seen:
            continue
        local_seen.add(digest)
        yield text
        accepted += 1
        if accepted >= examples:
            return
    raise RuntimeError(
        f"source {spec.name} supplied only {accepted}/{examples} holdout examples"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(
    tokenizer: Tokenizer,
    training_hashes: set[str],
    per_source: int,
) -> dict:
    by_source: dict[str, dict] = {}
    total_bytes = 0
    total_tokens = 0
    total_examples = 0

    for spec in SOURCES:
        row = {"bytes": 0, "tokens": 0, "examples": 0}
        for text in heldout_texts(spec, training_hashes, per_source):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if decoded != text:
                raise RuntimeError(
                    f"heldout roundtrip failed for source {spec.name}"
                )
            nbytes = len(text.encode("utf-8"))
            row["bytes"] += nbytes
            row["tokens"] += len(ids)
            row["examples"] += 1

        row["bytes_per_token"] = row["bytes"] / row["tokens"]
        row["sequence_reduction_vs_bytes_pct"] = 100.0 * (
            1.0 - row["tokens"] / row["bytes"]
        )
        by_source[spec.name] = row
        total_bytes += row["bytes"]
        total_tokens += row["tokens"]
        total_examples += row["examples"]

    return {
        "examples": total_examples,
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "sequence_reduction_vs_bytes_pct": 100.0
        * (1.0 - total_tokens / total_bytes),
        "by_source": by_source,
    }


def privacy_audit(tokenizer: Tokenizer) -> dict:
    base = set(ByteLevel.alphabet())
    suspicious: list[dict] = []
    for token, token_id in tokenizer.get_vocab().items():
        if token in SPECIAL_TOKENS:
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        reason = None
        for text in (token, decoded):
            if EMAIL_RE.search(text):
                reason = "email"
            elif SECRET_RE.search(text):
                reason = "secret_like"
            elif re.search(r"\d{12,}", text):
                reason = "long_digit_sequence"
            elif token not in base and CONTROL_RE.search(text):
                reason = "learned_control_character"
            if reason:
                break
        if reason:
            suspicious.append({"id": int(token_id), "reason": reason})
    return {
        "mandatory_bytelevel_base_symbols": len(base),
        "suspicious_entry_count": len(suspicious),
        "suspicious_ids_only": suspicious[:100],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=32_768)
    ap.add_argument("--training-chars", type=int, default=50_000_000)
    ap.add_argument("--eval-per-source", type=int, default=100)
    ap.add_argument("--min-total-reduction-pct", type=float, default=40.0)
    args = ap.parse_args()

    if args.vocab_size != 32_768:
        raise SystemExit("Helios production tokenizer gate requires vocab_size=32768")
    if args.training_chars < 20_000_000:
        raise SystemExit("Helios tokenizer training sample must be >=20M chars")
    if args.eval_per_source < 25:
        raise SystemExit("Helios tokenizer holdout requires >=25 examples/source")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=True,
    )
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        max_token_length=64,
        show_progress=True,
    )

    training_hashes: set[str] = set()
    tokenizer.train_from_iterator(
        training_iterator(args.training_chars, training_hashes),
        trainer=trainer,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != args.vocab_size:
        raise RuntimeError(
            f"expected exact 32768 vocabulary, got {actual_vocab}"
        )
    missing = [
        token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None
    ]
    if missing:
        raise RuntimeError(f"missing special tokens: {missing}")

    smoke = [
        "Grüße aus Berlin – Physik, Mathematik und 🚀.",
        "The model should preserve English technical prose.",
        "def f(x: int) -> int:\n    return x * x + 2\n",
        "ΔE = m c²; \\int_0^1 x^2 dx = 1/3",
        "https://example.org/a?x=42&id=abc-123",
    ]
    roundtrip = []
    for text in smoke:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        roundtrip.append({"tokens": len(ids), "ok": decoded == text})
    if not all(row["ok"] for row in roundtrip):
        raise RuntimeError("UTF-8/code/math tokenizer roundtrip failed")

    evaluation = evaluate(
        tokenizer,
        training_hashes,
        args.eval_per_source,
    )
    privacy = privacy_audit(tokenizer)

    failures: list[str] = []
    if evaluation["sequence_reduction_vs_bytes_pct"] < args.min_total_reduction_pct:
        failures.append("insufficient_total_compression")
    if privacy["mandatory_bytelevel_base_symbols"] != 256:
        failures.append("bytelevel_fallback_incomplete")
    if privacy["suspicious_entry_count"]:
        failures.append("suspicious_atomic_vocabulary_entries")

    report = {
        "schema": 1,
        "tokenizer": "neral-helios-byte-bpe-32k-v1",
        "foreign_vocab_or_merges_loaded": False,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab,
        "training_characters": args.training_chars,
        "training_unique_documents": len(training_hashes),
        "source_contract": [
            {
                "name": spec.name,
                "repo": spec.repo,
                "config": spec.config,
                "revision": spec.revision,
                "license": spec.license,
                "share": spec.share,
                "language": spec.language,
                "kind": spec.kind,
            }
            for spec in SOURCES
        ],
        "special_token_ids": {
            token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
        },
        "roundtrip": roundtrip,
        "holdout": evaluation,
        "privacy_audit": privacy,
        "tokenizer_sha256": sha256(tokenizer_path),
        "minimum_total_reduction_pct": args.min_total_reduction_pct,
        "pass": not failures,
        "failures": failures,
    }
    (out / "TOKENIZER_GATE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True), flush=True)
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
