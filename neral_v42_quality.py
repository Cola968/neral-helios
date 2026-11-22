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
    # Permit presentation-only wrappers, never competing answers or content
    # after the terminal marker. Keep all other kinds untouched.
    for fence in ("**", "__", "`"):
        if value.startswith(fence) and value.endswith(fence) and len(value) > 2 * len(fence):
            value = value[len(fence):-len(fence)].strip()
            break
    if re.fullmatch(r"[-+]?\d+\.", value):
        value = value[:-1]
    return value if value and len(value) <= 200 else None


def _explicit_two_stage_stock_steps(prompt):
    """Parse only an explicit quantity-times-unit calculation followed by subtraction.

    Source-level consistency, not a benchmark answer key or general NL solver.
    Ambiguous inputs never qualify for the multi-output shortcut.
    """
    if type(prompt) is not str:
        return None
    statement = prompt.split("ENDERGEBNIS:", 1)[0]
    q = statement.casefold()
    if (not re.search(r"\b(?:zuerst|zunächst)\b", q)
            or not re.search(r"\b(?:danach|anschließend)\b", q)
            or not re.search(r"\b(?:je|jeweils|pro)\b", q)
            or not re.search(r"\b(?:gibt|abgibt|entnimmt|zieht|verbraucht)\b", q)
            or not re.search(r"\b(?:ab|weg|rest\w*|übrig)\b", q)):
        return None
    tokens = re.findall(r"(?<![\w])(?:0|[1-9]\d{0,6})(?![\w])", statement)
    if len(tokens) != 3:
        return None
    count, per_unit, removed = (int(token) for token in tokens)
    if count <= 0 or per_unit <= 0 or removed < 0:
        return None
    total = count * per_unit
    if removed > total:
        return None
    return total, total - removed


def _source_consistent_stock_stdout(stdout):
    """Two ordered numeric steps from actual tool stdout; accept only known labels.

    No extra numbers, competing final answers or model-generated explanations.
    The caller checks both values against the user-source calculation.
    """
    if type(stdout) is not str or not 1 <= len(stdout) <= 240:
        return None
    rows = stdout.strip().splitlines()
    if len(rows) == 1 and rows[0].count(";") == 1:
        rows = rows[0].split(";")
    if len(rows) != 2:
        return None
    units = r"(?:\s+(?:Bauteile?|Teile?|St(?:ü|ue)ck))?"
    prefixes = (
        r"(?:Gesamtlieferung|Gesamtmenge|Gesamtzahl|Lieferung|"
        r"Zwischenergebnis|Zwischensumme|Subtotal|Total)",
        r"(?:Restmenge|Restbestand|Rest|Verbleibend|Uebrig|Übrig|Endmenge)",
    )
    values = []
    for row, label in zip(rows, prefixes):
        text = row.strip()
        pattern = rf"(?iu)(?:(?:{label})\s*[:=]\s*)?([-+]?\d{{1,18}}){units}"
        matched = re.fullmatch(pattern, text)
        if matched is None:
            return None
        values.append(int(matched.group(1)))
    return tuple(values)


def exact_executed_numeric_tail(stdout, *, succeeded, prompt):
    """One canonical final only from successful, source-consistent tool stdout.

    Multi-line outputs must match both independently parsed arithmetic steps.
    Never infer the result from an arbitrary last line or benchmark answer key.
    """
    if (succeeded is not True or type(stdout) is not str or
            type(prompt) is not str or
            re.search(r"(?i)\b(?:berechn\w*|rechne|rechnung\w*|"
                      r"ermittle|bestimme|ENDERGEBNIS)\b", prompt) is None):
        return None
    value = stdout.strip()
    if re.fullmatch(r"[-+]?\d{1,18}", value):
        return "ENDERGEBNIS: " + str(int(value))
    expected = _explicit_two_stage_stock_steps(prompt)
    actual = _source_consistent_stock_stdout(value)
    if expected is None or actual != expected:
        return None
    return "ENDERGEBNIS: " + str(expected[-1])



def forced_repair_tool(question, *, local=False, web=False):
    """Request a permitted real tool on second/third Deep pass only."""
    if not isinstance(question, str):
        return None
    q = question.casefold()
    if re.search(r"\b(?:schicht\w*|zuordn\w*|ordne|z3|constraint\w*|"
                 r"belegungs\w*|kombinator\w*)\b", q):
        return "z3_solver"
    if re.search(r"\b(?:python|programmier\w*|implementier\w*|funktion\w*|code|algorithmus)\b",q):
        return "python_code_sandbox"
    if (re.search(r"\b(?:berechn\w*|rechne\w*|rechnung\w*|"
                  r"gesam?tgewicht|wiegt)\b", q)
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


_YEAR_TOKEN = re.compile(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)")
_YEAR_QUESTION = re.compile(r"(?iu)\b(?:in\s+welchem\s+jahr|wann)\b")
_YEAR_EVENT = {
    "land": re.compile(r"(?iu)\b(?:land\w*|mondland\w*|moon\s+landing)\b"),
    "start": re.compile(r"(?iu)\b(?:start\w*|launch\w*)\b"),
    "erschein": re.compile(r"(?iu)\b(?:erschein\w*|publish\w*|releas\w*)\b"),
}


def _year_event_subject(question):
    """Extract a narrow public event and its named subject, never a target year."""
    if type(question) is not str or not _YEAR_QUESTION.search(question):
        return None
    match = re.search(
        r"(?iu)\b(landete|startete|erschien)\s+([^\W\d_][\w.-]*(?:\s+\d{1,3})?)\b",
        question)
    if match is None:
        return None
    stem = ("land" if match[1].casefold() == "landete"
            else "start" if match[1].casefold() == "startete" else "erschein")
    return stem, match[2]


def _year_quote_supports(question, quote, year):
    task = _year_event_subject(question)
    if task is None or not isinstance(quote, str):
        return False
    event, subject = task
    if subject.casefold() not in quote.casefold() or not _YEAR_EVENT[event].search(quote):
        return False
    years = _YEAR_TOKEN.findall(quote)
    if not years or set(years) != {year}:
        return False
    if re.search(r"(?iu)\b(?:nicht|never|did\s+not|not\s+land|geplant|planned|sollte)\b", quote):
        return False
    return True


def unique_read_year_source(question, sources):
    """Find one non-contradictory event-year quote in real read HTTPS pages.

    Return nothing if several years conflict, all matches are only snippets,
    or a quote merely describes plans/denials. Never infer a benchmark key.
    """
    if _year_event_subject(question) is None or type(sources) is not list:
        return None
    candidates = []
    for index, source in enumerate(sources, 1):
        if type(source) is not dict or source.get("page_read") is not True:
            continue
        url = source.get("url")
        parsed = urlsplit(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme != "https" or not parsed.hostname:
            continue
        content = source.get("page_text")
        if type(content) is not str or not 1 <= len(content) <= 120000:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
            quote = sentence.strip()
            if not 25 <= len(quote) <= 400:
                continue
            years = set(_YEAR_TOKEN.findall(quote))
            if len(years) != 1:
                continue
            year = next(iter(years))
            if _year_quote_supports(question, quote, year):
                candidates.append({"year": year, "source": f"web{index}",
                                   "quote": quote, "url": url})
    if not candidates or len({row["year"] for row in candidates}) != 1:
        return None
    # Prefer the shortest literal sentence so a long page cannot bury the
    # specific event behind unrelated dates or boilerplate.
    return min(candidates, key=lambda row: len(row["quote"]))


def year_evidence_verdict(question, packet, evidence, sources, wiki_hits):
    """Read-page literal year matching, never truth or freshness certification.

    None preserves the standard model checker for unsupported task shapes.
    Search previews, unread pages and invalid citation IDs fail closed.
    """
    if _year_event_subject(question) is None:
        return None
    if (type(packet) is not dict or set(packet) != {"answer", "citations"}
            or not isinstance(packet["answer"], str)
            or not isinstance(packet["citations"], list)):
        return "invalid_quote"
    answer = packet["answer"].strip()
    years = _YEAR_TOKEN.findall(answer)
    if len(years) != 1 or not 1 <= len(answer) <= 170:
        return "unsupported_year"
    year = years[0]
    event, subject = _year_event_subject(question)
    bare_year = re.fullmatch(r"(?:19|20)\d{2}[.!]?", answer)
    if bare_year is None:
        if (subject.casefold() not in answer.casefold()
                or not _YEAR_EVENT[event].search(answer)
                or len(re.split(r"[.!?]", answer.rstrip(".!?"))) != 1):
            return None
    citations = packet["citations"]
    if not 1 <= len(citations) <= 6:
        return "invalid_quote"
    if not isinstance(sources, list) or not sources:
        return "no_read_page"
    supported = False
    for item in citations:
        if type(item) is not dict or set(item) != {"source", "quote"}:
            return "invalid_quote"
        key, quote = item["source"], item["quote"]
        if not isinstance(key, str) or not re.fullmatch(r"web[1-9]\d*", key):
            return None
        idx = int(key[3:]) - 1
        if idx >= len(sources) or type(sources[idx]) is not dict:
            return "no_read_page"
        source = sources[idx]
        if source.get("page_read") is not True:
            return "no_read_page"
        url = source.get("url")
        parsed = urlsplit(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme != "https" or not parsed.hostname:
            return "invalid_quote"
        if (not isinstance(quote, str) or len(quote) < 25 or
                quote not in evidence.get(key, "") or
                quote not in source.get("page_text", "")):
            return "invalid_quote"
        if not _year_quote_supports(question, quote, year):
            return "unsupported_year"
        supported = True
    return "supported" if supported else "invalid_quote"


def checker_veto_class(issue, *, sources=(), wiki_hits=(), executed=None):
    """Privacy-safe bounded code, never the arbitrary model's checker wording."""
    if isinstance(issue, str):
        low = issue.casefold()
        if "keine gelesene webseite" in low or "keine lesbare webseite" in low:
            return "no_read_page"
        if "unbelegte jahreszahl" in low or "jahreszahl ohne beleg" in low:
            return "unsupported_year"
        if "konkreter widerspruch" in low:
            return "critic_contradiction"
        if "wörtlicher quellenbeleg fehlt" in low or "ungültiger quellenbeleg" in low:
            return "invalid_quote"
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
