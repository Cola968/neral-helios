"""Exact training/evaluation prompt boundaries and safe generation metadata.

Arctus 350M SFT learns each role segment separately. Whole-string tokenization
can merge across boundaries and score a different prompt than the chat runtime.
This module intentionally has no GPU / torch dependency.
"""
from __future__ import annotations

import re

try:
    from .chat_prompt350 import prompt_ids
except ImportError:  # Historical self-hosted workflows set PYTHONPATH=frontier350.
    from chat_prompt350 import prompt_ids


_HEADERS = re.compile(r"(?m)^(User|Assistant):")


def parse_chat_prefix(prefix: str) -> list[tuple[str, str]]:
    """Convert a frozen canonical chat prefix into complete prior role turns.

    Only the final Assistant: header may have no content. Avoid silently
    falling back to a differently tokenized benchmark on malformed cases.
    """
    if not isinstance(prefix, str):
        raise ValueError("chat case prefix must be text")
    matches = list(_HEADERS.finditer(prefix))
    if len(matches) < 2 or matches[0].start() != 0:
        raise ValueError("chat case must start with 'User:'")
    if not prefix.endswith("\nAssistant:") or matches[-1].group(1) != "Assistant":
        raise ValueError("chat case must end with an empty Assistant: header")
    if matches[-1].end() != len(prefix):
        raise ValueError("terminal Assistant: header must be empty")
    if len(matches) % 2 != 0:
        raise ValueError("malformed chat turn sequence")
    history = []
    for i in range(len(matches) - 1):
        match, following = matches[i], matches[i + 1]
        role = match.group(1).lower()
        if role != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("unexpected chat role order")
        content = prefix[match.end():following.start()].strip()
        if not content:
            raise ValueError("empty historical chat turn")
        history.append((role, content))
    return history


def case_prefix_ids(tokenizer, prefix: str) -> list[int]:
    """Use SFT-aligned per-segment BPE, not tokenization of the entire text."""
    return prompt_ids(tokenizer, parse_chat_prefix(prefix))


def generation_forbidden_ids(tokenizer) -> tuple[int, int, int]:
    specials = {}
    for text in ("<|bos|>", "<|eos|>", "<|sep|>", "<|pad|>"):
        value = tokenizer.token_to_id(text)
        if type(value) is not int or value < 0:
            raise ValueError("missing or invalid special token: " + text)
        specials[text] = value
    if len(set(specials.values())) != 4:
        raise ValueError("special tokens must have distinct IDs")
    return specials["<|bos|>"], specials["<|sep|>"], specials["<|pad|>"]


def trim_hallucinated_user_turn(text: str) -> tuple[str, bool]:
    """Prevent a fabricated User: turn from being scored as assistant output."""
    if not isinstance(text, str):
        raise ValueError("generation must be text")
    marker = "\nUser:"
    if marker not in text:
        return text, False
    return text.split(marker, 1)[0], True
