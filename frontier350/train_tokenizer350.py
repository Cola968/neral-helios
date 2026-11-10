from __future__ import annotations

"""Train and evaluate Neral Frontier's own 16K Byte-BPE tokenizer.

No foreign tokenizer vocabulary or merges are loaded. Text is streamed from the same pinned,
admitted foundation sources, passed through Frontier quality/PII filters, then used to learn
merges from scratch. The tokenizer remains local until its vocabulary receives a release review.
"""

import argparse
import hashlib
import json
from pathlib import Path

from build_frontier_data import DEFAULT_SOURCES, content_hash, extract_text, normalize, quality_ok, resolve_revision, stream_rows

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|sep|>", "<|pad|>"]


def source_iterator(spec, revision: str, char_budget: int):
    seen: set[str] = set()
    emitted = 0
    for row in stream_rows(spec, revision):
        text = normalize(extract_text(spec, row))
        ok, _ = quality_ok(text)
        if not ok:
            continue
        digest = content_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
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
        raise RuntimeError(f"tokenizer source {spec.name} exhausted at {emitted}/{char_budget} chars")


def combined_iterator(revisions: dict[str, str], total_chars: int):
    assigned = 0
    for i, spec in enumerate(DEFAULT_SOURCES):
        budget = int(total_chars * spec.share)
        if i == len(DEFAULT_SOURCES) - 1:
            budget = total_chars - assigned
        assigned += budget
        yield from source_iterator(spec, revisions[spec.name], budget)


def evaluation_texts(revisions: dict[str, str], per_source: int = 200):
    for spec in DEFAULT_SOURCES:
        accepted = 0
        for index, row in enumerate(stream_rows(spec, revisions[spec.name])):
            if index < 1000:
                continue
            text = normalize(extract_text(spec, row))
            ok, _ = quality_ok(text)
            if not ok:
                continue
            yield spec.name, text
            accepted += 1
            if accepted >= per_source:
                break


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=16_384)
    ap.add_argument("--training-chars", type=int, default=50_000_000)
    ap.add_argument("--eval-per-source", type=int, default=200)
    args = ap.parse_args()
    if not 8_192 <= args.vocab_size <= 32_768:
        raise SystemExit("Frontier tokenizer pilot vocabulary must stay in [8192,32768]")
    if args.training_chars < 5_000_000:
        raise SystemExit("tokenizer training sample is too small")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    revisions = {spec.name: resolve_revision(spec.repo) for spec in DEFAULT_SOURCES}
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(combined_iterator(revisions, args.training_chars), trainer=trainer)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out / "tokenizer.json"
    tok.save(str(tokenizer_path))

    vocab_size = tok.get_vocab_size()
    if vocab_size > args.vocab_size or vocab_size < 8192:
        raise RuntimeError(f"unexpected trained vocab size: {vocab_size}")
    for token in SPECIAL_TOKENS:
        if tok.token_to_id(token) is None:
            raise RuntimeError(f"missing special token: {token}")

    test_strings = [
        "Grüße aus Berlin – Physik, Mathematik und 🚀.",
        "def f(x):\n    return x * x + 2",
        "ΔE = m c²; 2 + 2 = 4.",
    ]
    roundtrip = []
    for text in test_strings:
        enc = tok.encode(text, add_special_tokens=False)
        dec = tok.decode(enc.ids, skip_special_tokens=True)
        roundtrip.append({"text": text, "tokens": len(enc.ids), "ok": dec == text})
    if not all(x["ok"] for x in roundtrip):
        raise RuntimeError(f"Byte-BPE UTF-8 roundtrip failed: {roundtrip}")

    by_source: dict[str, dict[str, int]] = {}
    total_bytes = total_tokens = examples = 0
    for source, text in evaluation_texts(revisions, args.eval_per_source):
        nbytes = len(text.encode("utf-8"))
        ntokens = len(tok.encode(text, add_special_tokens=False).ids)
        if ntokens <= 0:
            continue
        rec = by_source.setdefault(source, {"bytes": 0, "bpe_tokens": 0, "examples": 0})
        rec["bytes"] += nbytes
        rec["bpe_tokens"] += ntokens
        rec["examples"] += 1
        total_bytes += nbytes
        total_tokens += ntokens
        examples += 1

    if not examples or not total_tokens:
        raise RuntimeError("no tokenizer evaluation samples")
    report = {
        "architecture": "frontier350-v1",
        "tokenizer_candidate": "byte-bpe-16k-v1",
        "foreign_vocab_or_merges_loaded": False,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": vocab_size,
        "training_characters": args.training_chars,
        "source_revisions": revisions,
        "special_token_ids": {token: tok.token_to_id(token) for token in SPECIAL_TOKENS},
        "roundtrip": roundtrip,
        "eval_examples": examples,
        "eval_utf8_bytes": total_bytes,
        "eval_bpe_tokens": total_tokens,
        "bytes_per_bpe_token": total_bytes / total_tokens,
        "sequence_reduction_vs_raw_bytes_pct": 100.0 * (1.0 - total_tokens / total_bytes),
        "by_source": {
            key: {
                **value,
                "bytes_per_bpe_token": value["bytes"] / value["bpe_tokens"],
                "sequence_reduction_vs_raw_bytes_pct": 100.0 * (1.0 - value["bpe_tokens"] / value["bytes"]),
            }
            for key, value in by_source.items()
        },
        "tokenizer_sha256": sha256(tokenizer_path),
        "release_review_required": True,
    }
    (out / "tokenizer_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    # ASCII-escaped stdout is intentional on Windows runners whose console may still be CP1252.
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
