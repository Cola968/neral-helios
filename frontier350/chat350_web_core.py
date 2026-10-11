"""Pure chat turn routing for the optional local Neral 350M browser interface.

Tool responses are explicitly attributed and never counted as native model
capabilities. No weights, network calls, or UI dependencies in this module.
"""
from __future__ import annotations

from collections.abc import Callable

try:
    from .chat350_tools import tool_answer
except ImportError:
    from chat350_tools import tool_answer

History = list[tuple[str, str]]
NativeGenerator = Callable[[History], tuple[str, History]]


def run_turn(message: str, history: History, *,
             native: NativeGenerator | None = None,
             enable_tools: bool = True) -> tuple[str, str, History]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Bitte eine Frage eingeben.")
    question = message.strip()
    if len(question) > 2000:
        raise ValueError("Nachricht zu lang (maximal 2000 Zeichen).")
    # Gradio session state is client-specific; keep only bounded chat history.
    previous = list(history)[-24:]
    pending = previous + [("user", question)]
    tool = tool_answer(question) if enable_tools else None
    if tool is not None:
        return tool.text, "tool:" + tool.kind, pending + [("assistant", tool.text)]
    if native is None:
        answer = ("Kein Neral-350M-Modell geladen. In diesem Modus sind nur "
                  "ausdrücklich unterstützte Rechenaufgaben verfügbar.")
        return answer, "no-model", pending + [("assistant", answer)]
    answer, trimmed = native(pending)
    if not isinstance(answer, str):
        raise ValueError("Modellantwort muss Text sein.")
    answer = answer.strip()
    # An empty generation must not be misrepresented as a model success.
    if not answer:
        answer = "[Das Modell hat keine Antwort erzeugt.]"
    if not trimmed or trimmed[-1] != ("user", question):
        raise ValueError("Ungültiger Chat-Verlauf aus der Modellgenerierung.")
    return answer, "native-model", trimmed + [("assistant", answer)]
