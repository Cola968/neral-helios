"""Conservative single-final numeric extraction for the NERAL HIGH quality gate.

Parses a clearly designated final answer, not arbitrary numbers in prose.
The same implementation is used for baseline and candidate scoring.
"""
from __future__ import annotations
import re

MAX_TEXT_LENGTH = 32768
_NUMBER = r"[+-]?\\d{1,25}(?:[.,]\\d{1,25})?(?:/\\d{1,25})?"
_VALUE = r"(?:x\\s*=\\s*)?" + _NUMBER
_MARKER = re.compile(
    r"^(?:FINAL(?:_ANSWER)?|ANTWORT|ANSWER|ERGEBNIS|RESULT)\\s*[:=]\\s*"
    r"(?P<value>" + _VALUE + r")$", re.IGNORECASE)
_HEAD = re.compile(
    r"^(?:FINAL(?:_ANSWER)?|ANTWORT|ANSWER|ERGEBNIS|RESULT)\\s*[:=]",
    re.IGNORECASE)
_NATURAL = re.compile(
    r"^(?:Die\\s+Antwort|Das\\s+Ergebnis|The\\s+answer|The\\s+result)"
    r"\\s+(?:ist|lautet|beträgt|is|equals)\\s+(?P<value>"
    + _VALUE + r")\\.?$", re.IGNORECASE)
_BARE = re.compile(r"^" + _VALUE + r"$", re.IGNORECASE)


def extract_numeric_final(text: str) -> str | None:
    """Extract one unambiguous, explicitly terminal numeric value.

    Allows a final standalone value, one last clearly marked numeric answer,
    or a short last line declaring the answer. Multiple explicit markers and
    any text after a marker are rejected; prose is never mined for digits.
    This is parsing, NOT correctness verification against the original task.
    """
    if not isinstance(text, str) or not 1 <= len(text) <= MAX_TEXT_LENGTH:
        return None
    if any((ord(c) < 32 and c not in "\\t\\n\\r") or ord(c) == 127
           for c in text):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) > 256:
        return None

    def unwrap(line: str) -> str:
        # One pair of Markdown bold delimiters; no general Markdown evaluation.
        if line.startswith("**") and line.endswith("**") and len(line) >= 5:
            return line[2:-2].strip()
        return line

    lines = [unwrap(line) for line in lines]
    markers = [i for i, line in enumerate(lines) if _HEAD.match(line)]
    if markers and (len(markers) != 1 or markers[0] != len(lines) - 1):
        return None
    last = lines[-1]
    if markers:
        match = _MARKER.fullmatch(last)
    else:
        match = _BARE.fullmatch(last) or _NATURAL.fullmatch(last)
    if match is None:
        return None
    return match.group("value") if "value" in match.groupdict() else match.group(0)
