from __future__ import annotations
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

def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text).ids)

def encode_messages(tokenizer, messages: list[dict[str, str]], *, max_seq_len: int) -> EncodedChatExample:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if not messages or messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        raise ValueError("conversation must start with user and end with assistant")
    bos = _id(tokenizer, "<|bos|>")
    eos = _id(tokenizer, "<|eos|>")
    ids = [bos]
    supervised = [False]
    expected = "user"
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role != expected or not content:
            raise ValueError(f"invalid {expected} turn")
        if role == "user":
            segment = _encode(tokenizer, f"User: {content}\nAssistant:")
            ids.extend(segment)
            supervised.extend([False] * len(segment))
            expected = "assistant"
        else:
            segment = _encode(tokenizer, f" {content}\n")
            ids.extend(segment)
            supervised.extend([True] * len(segment))
            expected = "user"
            if i != len(messages) - 1:
                spacer = _encode(tokenizer, "User:")
                # Keep the next literal User label as unsupervised context. The following
                # user turn content is appended by replacing the duplicate label below.
                # To avoid duplicated labels, remove it immediately; this branch exists only
                # to make the alternation intent explicit.
                if spacer:
                    pass
    ids.append(eos)
    supervised.append(True)
    # The generic loop above emits a complete User/Assistant pair each time. For multi-turn
    # chats the next user segment already includes its own "User:" label.
    if len(ids) - 1 > max_seq_len:
        raise ValueError(f"encoded example exceeds max_seq_len={max_seq_len}: {len(ids)-1}")
    target_stream = [tok if sup else -100 for tok, sup in zip(ids, supervised)]
    x = ids[:-1]
    y = target_stream[1:]
    assistant_tokens = sum(v != -100 for v in y)
    if len(x) != len(y) or assistant_tokens <= 0 or y[-1] != eos:
        raise AssertionError("assistant-only causal alignment failure")
    return EncodedChatExample(x, y, assistant_tokens)

def encode_prompt_answer(tokenizer, prompt: str, answer: str, *, max_seq_len: int) -> EncodedChatExample:
    return encode_messages(
        tokenizer,
        [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": answer.strip()}],
        max_seq_len=max_seq_len,
    )

def render_multiturn(messages: list[dict[str, str]]) -> str:
    parts = []
    expected = "user"
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role != expected or not content:
            raise ValueError(f"expected non-empty {expected} turn")
        parts.append(("User: " if role == "user" else "Assistant: ") + content)
        expected = "assistant" if role == "user" else "user"
    if not parts or messages[-1].get("role") != "assistant":
        raise ValueError("conversation must end with assistant")
    return "\n".join(parts)
