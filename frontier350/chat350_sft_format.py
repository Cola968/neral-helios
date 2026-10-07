from __future__ import annotations

"""Tokenizer-agnostic assistant-only SFT formatting for the locked Neral 16K BPE.

The caller supplies the locally audited Hugging Face tokenizers.Tokenizer instance. No
foreign tokenizer, vocabulary or model weight is loaded by this module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EncodedChatExample:
    input_ids: list[int]
    targets: list[int]
    assistant_tokens: int


def _id(tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"missing required special token: {token}")
    return int(value)


def encode_prompt_answer(tokenizer, prompt: str, answer: str, *, max_seq_len: int) -> EncodedChatExample:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    prompt = prompt.strip()
    answer = answer.strip()
    if not prompt or not answer:
        raise ValueError("prompt and answer must be non-empty")

    bos = _id(tokenizer, "<|bos|>")
    eos = _id(tokenizer, "<|eos|>")

    prefix_text = f"User: {prompt}\nAssistant:"
    answer_text = f" {answer}\n"
    prefix = [bos] + list(tokenizer.encode(prefix_text).ids)
    answer_ids = list(tokenizer.encode(answer_text).ids) + [eos]
    ids = prefix + answer_ids

    if len(ids) < 2:
        raise ValueError("encoded example is too short")
    if len(ids) - 1 > max_seq_len:
        raise ValueError(f"encoded example exceeds max_seq_len={max_seq_len}: {len(ids)-1}")

    # Causal alignment: targets[t] predicts ids[t+1]. Prompt tokens are context-only.
    target_stream = [-100] * len(prefix) + answer_ids
    x = ids[:-1]
    y = target_stream[1:]
    if len(x) != len(y):
        raise AssertionError("causal alignment failure")
    assistant_tokens = sum(1 for value in y if value != -100)
    if assistant_tokens != len(answer_ids):
        raise AssertionError("assistant target accounting failure")
    return EncodedChatExample(x, y, assistant_tokens)


def render_multiturn(messages: list[dict[str, str]]) -> str:
    """Deterministic plain-text chat rendering for future multi-turn SFT data."""
    parts: list[str] = []
    expected = "user"
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role}")
        if role != expected:
            raise ValueError(f"expected {expected} turn, got {role}")
        if not content:
            raise ValueError("empty chat turn")
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content}")
        expected = "assistant" if role == "user" else "user"
    if not parts or messages[-1].get("role") != "assistant":
        raise ValueError("conversation must end with an assistant turn")
    return "\n".join(parts)
