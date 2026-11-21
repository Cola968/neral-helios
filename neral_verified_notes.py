"""Local read-only exact-match recall for verified/batch-graded training pairs.

A note lookup is NOT learning model weights, a fresh source verification, or an
independent evaluation. Never retrieve approximate matches or cached web facts.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import threading
from neral_verified_training import DATASET_PATH, _safe_text

_CACHE_LOCK=threading.Lock()
_CACHE={}
_MAX_BYTES=64*1024*1024


def exact_note_lookup(question: str, path: Path = DATASET_PATH) -> dict | None:
    if type(question) is not str or not question.strip() or len(question)>12000:
        return None
    # The answer to a time-dependent question requires current tools, not memory.
    lower=question.casefold()
    if any(word in lower for word in ('aktuell', 'heute', 'morgen', 'jetzt',
                                      'neueste', 'diese woche', 'dieses jahr',
                                      'such im web', 'recherchiere', 'quellen')):
        return None
    source=Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size>_MAX_BYTES:
        return None
    stat=source.stat()
    cache_key=(str(source.resolve()),stat.st_size,stat.st_mtime_ns)
    with _CACHE_LOCK:
        if cache_key not in _CACHE:
            records={}
            with source.open('r',encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    if (type(row) is not dict or type(row.get('meta')) is not dict or
                            not _safe_text(row.get('instruction'),row.get('response'))):
                        continue
                    meta=row['meta']
                    name=meta.get('source')
                    if meta.get('category')=='web' or meta.get('category')=='web_verified':
                        continue
                    if (name=='v39_benchmark' and
                            (meta.get('verification_kind')!='benchmark_strict_e2e' or
                             meta.get('evaluation_excluded') is not True or
                             meta.get('verifier_ok') is not False)):
                        continue
                    if (name=='live_verified' and
                            (meta.get('verification_kind')!='independent_runtime_proof' or
                             meta.get('verifier_ok') is not True or
                             meta.get('category')!='offline_verified')):
                        continue
                    if name not in ('v39_benchmark','live_verified'):
                        continue
                    key=sha256(' '.join(row['instruction'].split()).encode('utf-8')).hexdigest()
                    if meta.get('instruction_sha256')!=key:
                        continue
                    if key in records:
                        records[key]=None  # duplicates are ambiguous, never pick one
                    else:
                        records[key]={'text':row['response'], 'source':name,
                                      'verifier_ok':name=='live_verified' and meta.get('verifier_ok') is True}
            _CACHE.clear()
            _CACHE[cache_key]=records
        key=sha256(' '.join(question.split()).encode('utf-8')).hexdigest()
        return _CACHE[cache_key].get(key)


# V4.1: three non-interchangeable memory scopes. Do not save raw user
# dialogue as generalized model knowledge.
from dataclasses import dataclass
import os
import re
from neral_verified_training import verifier_ok

_STRATEGY_PATH = Path(__file__).resolve().parent / "data" / "verified_strategies.jsonl"
_STRATEGY_LOCK = threading.Lock()
_STRATEGY_LIMIT = 128 * 1024
_TEMPORAL = re.compile(
    r"(?i)\b(?:heute|aktuell|gestern|morgen|neueste|jetzt|"
    r"dieses jahr|diese woche|preis|kurs|flugplan|wetter|news|"
    r"today|current|latest|stock|exchange rate)\b"
)
_ARITHMETIC = re.compile(
    r"(?i)(?:\b(?:berechne|rechne|rechne aus|mathematik|gleichung|"
    r"ausrechnen|calculate|compute|solve|percentage)\b|[\d]\s*[*+/=-]\s*[\d])"
)
_CONSTRAINTS = re.compile(
    r"(?i)\b(?:schicht|zuordnung|bedingungen|optimier|"
    r"constraint|satisfiable|kombinatorik|belegung|z3)\w*\b"
)
_STRATEGIES = {
    "arithmetic": "Rechenaufgaben: zuerst die Bedingungen exakt erfassen, "
                  "mit Python oder SymPy rechnen und durch Einsetzen prüfen.",
    "constraints": "Formale Zuordnung: Bedingungen explizit modellieren, "
                   "mit Z3 prüfen und jede ursprüngliche Regel vergleichen.",
}


class SessionConversation:
    """Ephemeral, caller-owned conversation history; no file persistence."""
    def __init__(self, *, max_turns: int = 12):
        if type(max_turns) is not int or not 1 <= max_turns <= 80:
            raise ValueError("Invalid session limit")
        self._limit = max_turns
        self._turns: list[dict] = []

    def add(self, role: str, content: str) -> None:
        if role not in ("user", "assistant") or type(content) is not str or (
                not content.strip() or len(content.encode("utf-8")) > 4096):
            raise ValueError("Invalid session turn")
        self._turns.append({"role": role, "content": content.strip()})
        self._turns = self._turns[-self._limit:]

    def history(self) -> list[dict]:
        return [dict(t) for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()


class PersonalMemory:
    """Consent-only user memory delegating to existing login-scoped SQLite."""
    def __init__(self, user_db_path: Path):
        from neral_cpu_foundation.memory import LocalMemory
        self._store = LocalMemory(user_db_path)

    def save(self, key: str, value: str, *, kind="fact", consent=False) -> bool:
        return self._store.upsert("user", key, value, kind=kind, consent=consent)

    def list(self) -> list[dict]:
        return self._store.list("user")

    def delete(self, key: str) -> bool:
        return self._store.forget("user", key)

    def close(self) -> None:
        self._store.close()


class VerifiedStrategyStore:
    """General verified TOOL STRATEGIES only; no raw questions or answers.

    Do not confuse a past proof with proof of a new question. Stored hints
    merely bias model tool selection; they cannot authorize a proof badge.
    """
    def __init__(self, path: Path = _STRATEGY_PATH):
        self.path = Path(path)

    @staticmethod
    def _kind(question: str) -> str | None:
        if type(question) is not str or _TEMPORAL.search(question):
            return None
        if _CONSTRAINTS.search(question):
            return "constraints"
        if _ARITHMETIC.search(question):
            return "arithmetic"
        return None

    def learn(self, question: str, result: dict) -> bool:
        if (not verifier_ok(result) or result.get("verified") is not True
                or result.get("sources") or type(question) is not str
                or not question.strip()):
            return False
        category = self._kind(question)
        proof = result.get("proof_backend")
        if not category or proof not in (
            "trusted_source_arithmetic", "z3_unsat",
            "independent_enumeration", "independent_unique_enumeration",
            "independent_source_parser_z3", "independent_user_source_docker"):
            return False
        if category == "arithmetic" and proof not in (
                "trusted_source_arithmetic", "independent_user_source_docker"):
            return False
        row = {"category": category, "strategy": _STRATEGIES[category],
               "proof_backend": proof, "schema_version": 1}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _STRATEGY_LOCK:
            if self.path.is_symlink() or (
                    self.path.exists() and self.path.stat().st_size > _STRATEGY_LIMIT):
                return False
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        previous = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        return False
                    if previous.get("category") == category and previous.get("proof_backend") == proof:
                        return True
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        return True

    def hint(self, question: str) -> str:
        kind = self._kind(question)
        if not kind or not self.path.is_file() or self.path.is_symlink():
            return ""
        if self.path.stat().st_size > _STRATEGY_LIMIT:
            return ""
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if (row.get("category") == kind and
                        row.get("strategy") == _STRATEGIES[kind]):
                    return _STRATEGIES[kind]
        except (OSError, ValueError, AttributeError):
            return ""
        return ""

    def accept_correction(self, question: str, corrected: str, result: dict) -> bool:
        """A correction needs a proof bound to the exact current question."""
        if (type(question) is not str or type(corrected) is not str
                or type(result) is not dict or
                result.get("verified_question_sha256") !=
                sha256(" ".join(question.split()).encode("utf-8")).hexdigest()
                or corrected.strip() != str(result.get("text", "")).strip()):
            return False
        return self.learn(question, result)
