from __future__ import annotations

SPECIAL = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "end": "<|end|>",
}

def render_chat(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        if role not in SPECIAL:
            raise ValueError(f"unsupported role: {role}")
        content = msg.get("content", "").strip()
        parts.append(f"{SPECIAL[role]}\n{content}\n{SPECIAL['end']}")
    if add_generation_prompt:
        parts.append(f"{SPECIAL['assistant']}\n")
    return "\n".join(parts)
