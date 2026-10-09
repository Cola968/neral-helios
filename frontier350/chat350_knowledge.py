"""Offline, retrieval-only knowledge for Neral 350M.

All content comes from explicitly selected local .txt/.md documents. It is
returned as attributed excerpts, NEVER executed as code or as instructions.
This module uses only the standard library and makes zero network calls.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math
import re
import unicodedata


_STOP = frozenset("""
a an and are as at be but by can could der die das dem den des
ein eine einem einen einer eines er es für from have how ich im in
is ist it its mit nach of on oder or our over sie so that the their
them there these this to und was were what when where which who
wie wir wird with you your zu zum zur auf auch dass
""".split())
_ALLOWED = frozenset({".txt", ".md"})
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(text: str) -> list[str]:
    cleaned = unicodedata.normalize("NFKC", text).casefold()
    return [word for word in _WORD.findall(cleaned)
            if len(word) >= 2 and word not in _STOP and not word.isdecimal()]


@dataclass(frozen=True)
class Passage:
    source: str
    line: int
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeHit:
    source: str
    line: int
    excerpt: str
    matched_terms: tuple[str, ...]
    score: float


class KnowledgeIndex:
    """Bounded, deterministic BM25-style search over plain-text passages."""

    def __init__(self, passages: list[Passage]) -> None:
        self._passages = tuple(passages)
        self._freqs = [Counter(p.tokens) for p in passages]
        self._avg_len = max(1, sum(len(p.tokens) for p in passages) / max(1, len(passages)))
        self._df = Counter()
        for freq in self._freqs:
            self._df.update(freq.keys())

    @property
    def passage_count(self) -> int:
        return len(self._passages)

    @classmethod
    def from_directory(cls, root: str | Path, *,
                       max_files: int = 500,
                       max_file_bytes: int = 1_000_000,
                       max_total_bytes: int = 12_000_000,
                       lines_per_passage: int = 10,
                       overlap_lines: int = 2) -> "KnowledgeIndex":
        folder = Path(root).expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise ValueError("knowledge root must be a directory")
        if not (1 <= max_files <= 5000 and 1024 <= max_file_bytes <= 8_000_000
                and 1024 <= max_total_bytes <= 50_000_000):
            raise ValueError("unsafe indexing limits")
        if not (2 <= lines_per_passage <= 50 and 0 <= overlap_lines < lines_per_passage):
            raise ValueError("invalid passage window")

        import os
        eligible = []
        for base, dirs, files in os.walk(folder, followlinks=False):
            basepath = Path(base)
            dirs[:] = sorted(d for d in dirs if not (basepath / d).is_symlink())
            for name in sorted(files):
                path = basepath / name
                if path.suffix.lower() not in _ALLOWED or path.is_symlink() or not path.is_file():
                    continue
                if not path.resolve().is_relative_to(folder):
                    continue
                eligible.append(path)
        if len(eligible) > max_files:
            raise ValueError("too many knowledge files")

        passages = []
        total = 0
        for file in sorted(eligible):
            size = file.stat().st_size
            if size > max_file_bytes:
                raise ValueError(f"knowledge file too large: {file.name}")
            total += size
            if total > max_total_bytes:
                raise ValueError("total knowledge size exceeded")
            # Strict UTF-8 prevents platform-dependent indexing and hidden replacements.
            content = file.read_text(encoding="utf-8")
            lines = content.splitlines()
            step = lines_per_passage - overlap_lines
            source = file.relative_to(folder).as_posix()
            for start in range(0, len(lines), step):
                window = lines[start:start + lines_per_passage]
                if not window:
                    continue
                excerpt = "\n".join(window).strip()
                if not excerpt:
                    continue
                tokens = _terms(excerpt)
                if tokens:
                    passages.append(Passage(source, start + 1, excerpt[:2400], tuple(tokens)))
        return cls(passages)

    def search(self, query: str, *, limit: int = 3) -> list[KnowledgeHit]:
        if not isinstance(query, str) or len(query) > 500:
            return []
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        q = tuple(dict.fromkeys(_terms(query)))
        if not q or not self._passages:
            return []
        results = []
        count = len(self._passages)
        for idx, (p, freq) in enumerate(zip(self._passages, self._freqs)):
            shared = tuple(t for t in q if t in freq)
            # Never claim a matching passage for a generic one-word overlap
            # in a longer question. Single-keyword searches remain supported.
            if len(shared) < min(2, len(q)):
                continue
            score = 0.0
            for term in shared:
                idf = math.log(1 + (count - self._df[term] + 0.5) / (self._df[term] + 0.5))
                f = freq[term]
                score += idf * (f * 2.2) / (f + 1.2 * (0.25 + 0.75 * len(p.tokens) / self._avg_len))
            if score > 0:
                results.append(KnowledgeHit(p.source, p.line, p.text, shared, round(score, 5)))
        results.sort(key=lambda h: (-h.score, h.source, h.line))
        return results[:limit]


def render_knowledge_hits(hits: list[KnowledgeHit]) -> str:
    """Give the user grounded excerpts, not a synthesized model assertion."""
    if not hits:
        return ""
    parts = ["Lokale Quellen (Textauszüge, keine KI-Schlussfolgerung):"]
    for hit in hits:
        parts.append(f"[{hit.source}:Zeile {hit.line}] {hit.excerpt}")
    return "\n\n".join(parts)
