"""Convert frozen benchmark transcripts to native Arctus 350M chat token IDs.

Never encode an entire transcript in one BPE call: SFT, DPO and the actual
chat runtime tokenize alternating user/assistant segments independently.
"""
from __future__ import annotations

try:
    from .chat_prompt350 import prompt_ids, render_history
except ImportError:
    from chat_prompt350 import prompt_ids, render_history


def parse_benchmark_prefix(prefix: str) -> list[tuple[str, str]]:
    """Parse only canonical, complete User/Assistant role boundaries.

    The last turn must be a user prompt awaiting an Assistant completion.
    Reject, rather than silently reinterpret, malformed or multi-line data.
    """
    if not isinstance(prefix, str) or not prefix or len(prefix) > 8192:
        raise ValueError("benchmark prefix must be a nonempty bounded string")
    lines = prefix.split("\n")
    if len(lines) < 2 or len(lines) % 2:
        raise ValueError("benchmark prefix must contain alternating chat turns")
    history: list[tuple[str, str]] = []
    for i in range(0, len(lines), 2):
        user = lines[i]
        if not user.startswith("User: ") or not user[6:].strip():
            raise ValueError("expected nonempty User turn")
        history.append(("user", user[6:]))
        assistant = lines[i + 1]
        if i + 2 == len(lines):
            if assistant != "Assistant:":
                raise ValueError("benchmark must end with pending Assistant:")
        else:
            if not assistant.startswith("Assistant: ") or not assistant[11:].strip():
                raise ValueError("expected nonempty Assistant answer")
            history.append(("assistant", assistant[11:]))
    if render_history(history) != prefix:
        raise ValueError("noncanonical benchmark chat prefix")
    return history


def benchmark_prompt_ids(tokenizer, prefix: str) -> list[int]:
    """Exactly match the SFT/DPO and native CLI prompt tokenization."""
    return prompt_ids(tokenizer, parse_benchmark_prefix(prefix))
