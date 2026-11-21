"""V4.1 conservative multi-turn context resolution.

Only *user-authored* earlier turns may clarify an elliptical question. This
module never promotes the reconstructed question into an independent proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Resolution:
    work_question: str
    is_followup: bool
    independent_proof_allowed: bool


_FOLLOWUP_START = re.compile(
    r"(?i)^\s*(?:und\b|aber\b|was ist mit\b|wie sieht es (?:damit|bei)\b|"
    r"wie viel (?:\w+\s+){0,3}(?:er|sie|es|das)\b|"
    r"wie lange (?:\w+\s+){0,3}(?:er|sie|es|das)\b|"
    r"(?:kann|braucht|schafft|kostet|verbraucht) (?:er|sie|es|das)\b|"
    r"(?:was|wie) (?:wäre|ist) (?:es|das) (?:dann|damit)\b)"
)
_REFERENCE = re.compile(
    r"(?i)\b(?:er|sie|es|ihn|ihm|diese|dieser|dieses|damit|dabei|"
    r"davon|dafür|dort|gleiche|dasselbe|vorherigen|the same|it|that one)\b"
)
_CHANGE = re.compile(r"(?i)^\s*(?:neues thema|themenwechsel|andere frage|unabhängig davon)\b")
_MAX_USER_CONTEXT = 3
_MAX_QUESTION_BYTES = 4096


def resolve_question(turns: list[dict]) -> Resolution:
    if (type(turns) is not list or not turns or
            turns[-1].get("role") != "user" or
            type(turns[-1].get("content")) is not str):
        raise ValueError("A current user question is required")
    current = turns[-1]["content"].strip()
    if not current or len(current.encode("utf-8")) > _MAX_QUESTION_BYTES:
        raise ValueError("Invalid user question")
    previous_users = [
        t["content"].strip() for t in turns[:-1] if
        type(t) is dict and t.get("role") == "user" and
        type(t.get("content")) is str and t["content"].strip()
    ][-_MAX_USER_CONTEXT:]
    followup = bool(previous_users) and not _CHANGE.search(current) and (
        bool(_FOLLOWUP_START.search(current)) or
        (len(current.split()) <= 10 and bool(_REFERENCE.search(current)))
    )
    if not followup:
        return Resolution(current, False, True)
    # Enforce predictable finite input; do not include previous assistant
    # assertions, tool output, saved memories, or older unrelated chats.
    user_context = "\n".join(
        "- " + p.replace("\n", " ").strip()[:500] for p in previous_users
    )
    work = (
        "Kontext ausschließlich aus vorherigen Nutzerfragen "
        "(keine unabhängig geprüften Fakten; nicht als zusätzliche "
        "Bedingungen einer neuen Aufgabe verwenden):\n"
        + user_context
        + "\nAktuelle Anschlussfrage: " + current
        + "\nBeantworte nur die Anschlussfrage. Bei uneindeutigem Bezug "
          "nachfragen; keine alten Modellantworten als Fakten übernehmen."
    )
    if len(work.encode("utf-8")) > _MAX_QUESTION_BYTES:
        return Resolution(current, False, True)
    return Resolution(work, True, False)
