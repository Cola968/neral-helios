"""Pure, bounded, local multi-turn chat context and decoding for Neral 350M.

No weight access, network, persistence, or claim of improved model quality.
"""
from __future__ import annotations

import re

MAX_CONTEXT = 512
MAX_NEW = 64
MAX_MESSAGES = 13
MAX_MESSAGE_CHARS = 1024
ROLE_LINE = re.compile(r"(?im)^\s*(?:user|assistant|system|nutzer|benutzer)\s*:")
NEXT_ROLE = re.compile(r"(?im)(?:^|\n)\s*(?:user|assistant|system|nutzer|benutzer)\s*:")


def validate_messages(messages: object, budget: object) -> list[dict[str, str]]:
    if type(budget) is not int or not 1 <= budget <= MAX_NEW:
        raise ValueError("max_new_tokens must be an integer from 1 to 64")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError("messages must contain 1 to 13 turns")
    cleaned = []
    for index, item in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(item, dict) or set(item) != {"role", "content"} or item["role"] != role:
            raise ValueError("messages must alternate user/assistant and end with user")
        text = item["content"]
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            raise ValueError("message must contain 1 to 1024 characters")
        text = text.strip()
        if ROLE_LINE.search(text):
            raise ValueError("role separator at line start is not supported")
        cleaned.append({"role": role, "content": text})
    if len(cleaned) % 2 != 1:
        raise ValueError("last message must be a user message")
    return cleaned


def encode_context(tokenizer, bos: int, messages: object, budget: int) -> tuple[list[int], int]:
    """Drop complete oldest Q/A pairs; never silently shorten the latest prompt."""
    cleaned = validate_messages(messages, budget)
    discarded = 0
    while True:
        dialogue = "".join(("User: " if m["role"] == "user" else "Assistant: ") +
                           m["content"] + "\n" for m in cleaned)
        ids = [bos] + list(tokenizer.encode(dialogue + "Assistant:").ids)
        if len(ids) + budget <= MAX_CONTEXT:
            return ids, discarded
        if len(cleaned) == 1:
            raise ValueError("latest message exceeds available 512-token context")
        cleaned = cleaned[2:]
        discarded += 1


def _looping(text: str) -> bool:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 15:
        return False
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return grams.count(grams[-1]) >= 3 and (len(grams) - len(set(grams))) / len(grams) >= 0.28


def generate_turn(tokenizer, bos: int, eos: int, messages: object, budget: int,
                  next_token) -> dict:
    """next_token(ids, emitted) is the real model callback; tests may inject a fake."""
    if type(bos) is not int or type(eos) is not int or bos == eos:
        raise ValueError("invalid model BOS/EOS IDs")
    ids, discarded = encode_context(tokenizer, bos, messages, budget)
    used_context = len(ids)
    emitted: list[int] = []
    for step in range(1, budget + 1):
        token = next_token(ids, emitted)
        if type(token) is not int or token < 0:
            raise ValueError("model returned invalid token")
        if token == eos:
            return {"answer": tokenizer.decode(emitted, skip_special_tokens=True),
                    "generated_tokens": step, "finish_reason": "eos",
                    "dropped_history_pairs": discarded, "input_tokens": used_context}
        emitted.append(token)
        ids.append(token)
        text = tokenizer.decode(emitted, skip_special_tokens=True)
        boundary = NEXT_ROLE.search(text)
        if boundary:
            return {"answer": text[:boundary.start()].rstrip(),
                    "generated_tokens": step, "finish_reason": "role_boundary",
                    "dropped_history_pairs": discarded, "input_tokens": used_context}
        if _looping(text):
            return {"answer": text, "generated_tokens": step,
                    "finish_reason": "repetition_guard",
                    "dropped_history_pairs": discarded, "input_tokens": used_context}
    return {"answer": tokenizer.decode(emitted, skip_special_tokens=True),
            "generated_tokens": budget, "finish_reason": "max_tokens",
            "dropped_history_pairs": discarded, "input_tokens": used_context}
