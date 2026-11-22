"""NERAL V4.2 answer-format and evidence contracts.

No private gold values are ever inserted into model prompts. Source and
checker policies fail closed; no general "relax verifier" switch.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit


FINAL_LABEL = "ENDERGEBNIS:"
_EXACT_KINDS = frozenset(("integer", "yesno", "web_integer", "string"))


def prompt_for_exact_case(case):
    """Return an ephemeral benchmark prompt; never mutate the sealed case."""
    original = case["prompt"]
    if case.get("grader") != "independent_exact":
        return original
    ref = case.get("sealed_reference") or {}
    if ref.get("kind") not in _EXACT_KINDS:
        return original
    return (original +
            "\n\nAntwortformat für diesen Einzeltest: Du darfst den Rechenweg "
            "kurz erklären. Schreibe als allerletzte Zeile genau "
            "ENDERGEBNIS: <isolierter Wert>. Danach keine weiteren Zeichen. "
            "Bei Ja/Nein-Fragen nutze nur Ja oder Nein als Wert. "
            "Bei einer Zeichenkette nur die verlangte Zeichenkette.")


def final_marked_value(text):
    """Only one explicitly marked, FINAL nonempty line can be a result."""
    if type(text) is not str:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    matched = [i for i, line in enumerate(lines)
               if re.match(r"(?i)^ENDERGEBNIS\s*:", line)]
    if matched != [len(lines)-1]:
        return None
    ending = re.fullmatch(r"(?i)ENDERGEBNIS\s*:\s*(.+)", lines[-1])
    if ending is None:
        return None
    value = ending.group(1).strip()
    return value if value and len(value) <= 200 else None


def forced_repair_tool(question, *, local=False, web=False):
    """Request a permitted real tool on second/third Deep pass only."""
    if not isinstance(question, str):
        return None
    q = question.casefold()
    if re.search(r"\b(?:schicht\w*|zuordn\w*|ordne|z3|constraint\w*|"
                 r"belegungs\w*|kombinator\w*)\b", q):
        return "z3_solver"
    if (re.search(r"\b(?:berechn\w*|rechne\w*|rechnung\w*|"
                  r"gesam?tgewicht|wiegt|python|programmier\w*|"
                  r"implementier\w*|funktion\w*)\b", q)
            or re.search(r"\d\s*[×*/+\-]\s*\d", q)):
        return "python_sandbox"
    if local or web:
        return "direct_reasoning"
    return None


def evidence_veto_is_generic(issue):
    """Only the explicitly known mistaken 'no quote' wording qualifies."""
    if not isinstance(issue, str):
        return False
    norm = " ".join(issue.casefold().strip().split()).rstrip(".!")
    return norm in {
        "wörtlicher quellenbeleg fehlt",
        "kein wörtlicher quellenbeleg vorhanden",
        "zitierter quellentext fehlt",
    }


def _clean(value):
    return " ".join(str(value).split()).casefold()


def grounded_literal_citations(packet, evidence, sources, wiki_hits, *, passages=None):
    """A *verbatim* bounded answer with provenance can clear generic quote veto.

    Never accepts a paraphrase, unsupported extra claim, un-read URL,
    unverified source, or a possibly-current offline fact as certified.
    """
    if not isinstance(packet, dict) or set(packet) != {"answer", "citations"}:
        return False
    answer = packet["answer"]
    citations = packet["citations"]
    if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 1800:
        return False
    if not isinstance(citations, list) or not 1 <= len(citations) <= 6:
        return False
    allowed = []
    for item in citations:
        if isinstance(item, str) and isinstance(passages, dict):
            value = passages.get(item)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                return False
            key, quote = value
        elif type(item) is dict and set(item) == {"source", "quote"}:
            key, quote = item["source"], item["quote"]
        else:
            return False
        if not isinstance(key, str) or not isinstance(quote, str) or not quote.strip():
            return False
        evidence_text = evidence.get(key)
        if not isinstance(evidence_text, str) or _clean(quote) not in _clean(evidence_text):
            return False
        if key.startswith("wiki") and re.fullmatch(r"wiki[1-9]\d*", key):
            idx = int(key[4:])-1
            if idx >= len(wiki_hits):
                return False
            hit = wiki_hits[idx]
            url = hit.get("url", "")
            if (not isinstance(url, str) or
                    not url.startswith("http://127.0.0.1:9691/content/") or
                    _clean(quote) not in _clean(hit.get("context", ""))):
                return False
        elif key.startswith("web") and re.fullmatch(r"web[1-9]\d*", key):
            idx = int(key[3:])-1
            if idx >= len(sources):
                return False
            source = sources[idx]
            url = source.get("url","")
            parsed = urlsplit(url) if isinstance(url,str) else None
            if (source.get("page_read") is not True or parsed is None or
                    parsed.scheme != "https" or not parsed.hostname):
                return False
        else:
            return False
        allowed.append(_clean(quote))
    # A literal answer must equal a quoted span; no ungrounded extra claims.
    return _clean(answer) in allowed


def checker_veto_class(issue, *, sources=(), wiki_hits=(), executed=None):
    """Privacy-safe bounded code, never the arbitrary model's checker wording."""
    if evidence_veto_is_generic(issue):
        return "generic_quote_veto"
    if isinstance(issue,str) and re.search(r"(?i)widerspruch|contradiction",issue):
        return "contradiction_reported"
    if isinstance(issue,str) and re.search(r"(?i)webseite|seite|quelle|beleg|zitat|quote",issue):
        return "source_or_quote_issue"
    if isinstance(issue,str) and re.search(r"(?i)tool|werkzeug|docker|sandbox|z3",issue):
        return "tool_issue"
    if not sources and not wiki_hits:
        return "no_external_evidence"
    if executed is None:
        return "tool_not_executed"
    return "other_checker_issue"
