from __future__ import annotations

"""Fail-closed privacy/integrity audit for a locally trained Frontier tokenizer.

The tokenizer is model architecture, not training data. Before it can become a locked release
asset, its learned vocabulary is inspected for obvious PII/secret-like atomic tokens and its
special IDs, vocabulary size and file digest must match the measured tokenizer report.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

EMAIL_RE = re.compile(r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}")
SECRET_RE = re.compile(r"(?i)(?:sk-|api[_-]?key|secret|password|passwd|bearer)[A-Za-z0-9_\-/.+=]{8,}")
LONG_DIGITS_RE = re.compile(r"\d{12,}")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EXPECTED_SPECIALS = ["<|bos|>", "<|eos|>", "<|sep|>", "<|pad|>"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tokenizers import Tokenizer, pre_tokenizers

    tokenizer_path = Path(args.tokenizer)
    report = json.loads(Path(args.report).read_text(encoding="utf8"))
    tok = Tokenizer.from_file(str(tokenizer_path))
    failures: list[str] = []

    digest = sha256(tokenizer_path)
    if digest != report.get("tokenizer_sha256"):
        failures.append("tokenizer_sha256_mismatch")
    if tok.get_vocab_size() != 16_384 or int(report.get("actual_vocab_size", -1)) != 16_384:
        failures.append("vocab_size_not_16384")
    if report.get("foreign_vocab_or_merges_loaded") is not False:
        failures.append("foreign_tokenizer_components_not_excluded")

    special_ids = report.get("special_token_ids") or {}
    for token in EXPECTED_SPECIALS:
        tid = tok.token_to_id(token)
        if tid is None or special_ids.get(token) != tid:
            failures.append(f"special_id_mismatch:{token}")

    # A ByteLevel BPE deliberately includes the complete 256-symbol byte fallback alphabet.
    # Some of those symbols decode to ASCII control bytes; they are required for lossless UTF-8
    # coverage and are not learned from corpus content. Only learned/non-base tokens are subject
    # to the control-character privacy rule. PII/secret checks still apply to every non-special token.
    base_alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    if len(base_alphabet) != 256:
        failures.append("bytelevel_base_alphabet_not_256")

    suspicious: list[dict] = []
    learned_checked = 0
    for token, tid in tok.get_vocab().items():
        if token in EXPECTED_SPECIALS:
            continue
        decoded = tok.decode([tid], skip_special_tokens=True)
        candidates = {token, decoded}
        reason = None
        for text in candidates:
            if EMAIL_RE.search(text):
                reason = "email"
            elif SECRET_RE.search(text):
                reason = "secret_like"
            elif LONG_DIGITS_RE.search(text):
                reason = "long_digit_sequence"
            elif token not in base_alphabet and CONTROL_RE.search(text):
                reason = "learned_control_character"
            if reason:
                break
        if token not in base_alphabet:
            learned_checked += 1
        if reason:
            suspicious.append({"id": int(tid), "reason": reason})

    # Never write suspicious token text itself into report artifacts.
    if suspicious:
        failures.append(f"suspicious_vocab_entries:{len(suspicious)}")

    roundtrip = report.get("roundtrip") or []
    if not roundtrip or not all(x.get("ok") is True for x in roundtrip):
        failures.append("utf8_roundtrip_failed")
    compression = float(report.get("sequence_reduction_vs_raw_bytes_pct", -1))
    if compression < 40.0:
        failures.append("insufficient_sequence_compression")

    result = {
        "pass": not failures,
        "failures": failures,
        "tokenizer_sha256": digest,
        "vocab_size": tok.get_vocab_size(),
        "mandatory_bytelevel_base_symbols": len(base_alphabet),
        "learned_tokens_privacy_checked": learned_checked,
        "suspicious_entry_count": len(suspicious),
        "suspicious_ids_only": suspicious[:50],
        "sequence_reduction_vs_raw_bytes_pct": compression,
        "policy": "Tokenizer promotion requires integrity, UTF-8 roundtrip, >=40% measured sequence reduction, complete lossless ByteLevel fallback, and no obvious PII/secret-like learned atomic vocabulary entries.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
