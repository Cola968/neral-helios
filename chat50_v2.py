from __future__ import annotations

"""Efficient multi-turn chat encoding and hardware-aware batching for Neral-50M Chat V2.

Previous assistant turns appear as ordinary inference-style history, while the current assistant
answer is the only supervised span and ends in EOS. A multi-turn conversation is expanded to one
prefix->assistant example per assistant turn.

The sampler supports either a portable fixed token budget or an explicit per-bucket batch profile
measured on the target GPU. The latter matters because causal-attention memory/compute is not
linear in sequence length, so equal padded-token budgets are not equally efficient.

The pure encoding/sampling layer intentionally remains importable without PyTorch. This allows
GitHub-hosted CPU data audits to build and validate ChatMix manifests without downloading a large
ML runtime. Tensor collation/device transfer still fail explicitly if PyTorch is unavailable.
"""

from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

try:
    import torch
except ModuleNotFoundError:  # data-only hosted jobs do not need a full Torch installation
    torch = None

from tokenizer import ByteTokenizer

TOKENIZER = ByteTokenizer()
DEFAULT_BUCKETS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


@dataclass(frozen=True)
class EncodedChatExample:
    row_id: str
    source: str
    language: str
    domain: str
    assistant_turn: int
    input_ids: tuple[int, ...]
    targets: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_tokens(self) -> int:
        return sum(t != -100 for t in self.targets)


def _clean(value: str) -> str:
    return value.strip().replace("\r\n", "\n").replace("\r", "\n")


def validate_messages(messages: list[dict]) -> None:
    if len(messages) < 2 or len(messages) % 2:
        raise ValueError("chat must contain user/assistant pairs")
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise ValueError(f"message {i} expected role={expected}")
        content = message.get("content")
        if not isinstance(content, str) or not _clean(content):
            raise ValueError("chat message content must be non-empty text")


def format_history(messages: list[dict], assistant_index: int) -> tuple[str, str]:
    validate_messages(messages)
    if assistant_index <= 0 or assistant_index >= len(messages) or assistant_index % 2 != 1:
        raise ValueError("assistant_index must reference an assistant turn")
    lines: list[str] = []
    for message in messages[:assistant_index]:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {_clean(message['content'])}")
    prefix = "\n".join(lines) + "\nAssistant:"
    answer = _clean(messages[assistant_index]["content"])
    return prefix, answer


def encode_turn(row: dict, assistant_index: int, max_seq: int) -> EncodedChatExample:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row.messages must be a list")
    prefix, answer_text = format_history(messages, assistant_index)
    prefix_ids = TOKENIZER.encode(prefix, bos=True)
    answer_ids = TOKENIZER.encode(" " + answer_text + "\n", eos=True)
    ids = prefix_ids + answer_ids
    if len(ids) < 2:
        raise ValueError("encoded chat is empty")
    x = tuple(ids[:-1])
    raw_targets = [-100] * len(prefix_ids) + answer_ids
    y = tuple(raw_targets[1:])
    if len(x) != len(y):
        raise AssertionError("chat input/target shift mismatch")
    if len(x) > max_seq:
        raise ValueError(f"chat example exceeds max_seq={max_seq}: {len(x)}")
    eos_targets = sum(t == TOKENIZER.eos_id for t in y)
    if eos_targets != 1:
        raise AssertionError(f"each prefix example must supervise exactly one EOS, got {eos_targets}")
    rid = str(row.get("id", "unknown"))
    turn = (assistant_index + 1) // 2
    return EncodedChatExample(
        row_id=f"{rid}:turn{turn}",
        source=str(row.get("source", "unknown")),
        language=str(row.get("language", "unknown")),
        domain=str(row.get("domain", "chat")),
        assistant_turn=turn,
        input_ids=x,
        targets=y,
    )


def expand_conversation(row: dict, max_seq: int, include_all_turns: bool = True) -> list[EncodedChatExample]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row.messages must be a list")
    validate_messages(messages)
    indices = list(range(1, len(messages), 2))
    if not include_all_turns:
        indices = indices[-1:]
    out = []
    for idx in indices:
        try:
            out.append(encode_turn(row, idx, max_seq))
        except ValueError as exc:
            if "exceeds max_seq" not in str(exc):
                raise
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def preencode(rows: Iterable[dict], max_seq: int, include_all_turns: bool = True) -> list[EncodedChatExample]:
    examples: list[EncodedChatExample] = []
    seen: set[str] = set()
    for row in rows:
        for ex in expand_conversation(row, max_seq, include_all_turns):
            h = hashlib.sha256()
            h.update(str(ex.input_ids).encode("ascii"))
            h.update(str(ex.targets).encode("ascii"))
            key = h.hexdigest()
            if key in seen:
                continue
            seen.add(key)
            examples.append(ex)
    if not examples:
        raise ValueError("no encodable chat examples")
    return examples


def bucket_width(length: int, buckets=DEFAULT_BUCKETS) -> int:
    for width in buckets:
        if length <= width:
            return width
    raise ValueError(f"length {length} exceeds largest bucket {buckets[-1]}")


def build_buckets(examples: list[EncodedChatExample], buckets=DEFAULT_BUCKETS) -> dict[int, list[int]]:
    out = {int(width): [] for width in buckets}
    for i, ex in enumerate(examples):
        out[bucket_width(ex.length, buckets)].append(i)
    return {width: indices for width, indices in out.items() if indices}


def normalize_batch_profile(profile: dict | None, known_widths: set[int]) -> dict[int, int]:
    if profile is None:
        return {}
    normalized = {int(k): int(v) for k, v in profile.items()}
    if any(width not in known_widths for width in normalized):
        raise ValueError("batch profile contains unknown bucket width")
    if any(rows <= 0 for rows in normalized.values()):
        raise ValueError("batch profile row counts must be positive")
    return normalized


class TokenBudgetSampler:
    """Deterministic length-bucket sampler with optional measured rows-per-bucket profile."""

    def __init__(
        self,
        examples: list[EncodedChatExample],
        token_budget: int,
        seed: int,
        buckets=DEFAULT_BUCKETS,
        batch_profile: dict | None = None,
    ):
        if token_budget < min(buckets):
            raise ValueError("token_budget too small")
        self.examples = examples
        self.token_budget = int(token_budget)
        self.rng = random.Random(seed)
        self.buckets = build_buckets(examples, buckets)
        self.batch_profile = normalize_batch_profile(batch_profile, set(self.buckets))
        self.orders = {w: list(v) for w, v in self.buckets.items()}
        self.cursors = {w: 0 for w in self.buckets}
        for order in self.orders.values():
            self.rng.shuffle(order)
        self.width_population = [(w, len(v)) for w, v in self.buckets.items()]

    def rows_for_width(self, width: int) -> int:
        return self.batch_profile.get(width, max(1, self.token_budget // width))

    @property
    def maximum_padded_tokens(self) -> int:
        return max(width * self.rows_for_width(width) for width in self.buckets)

    def state_dict(self) -> dict:
        return {
            "token_budget": self.token_budget,
            "batch_profile": self.batch_profile,
            "orders": self.orders,
            "cursors": self.cursors,
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["token_budget"]) != self.token_budget:
            raise ValueError("sampler token budget changed")
        stored_profile = {int(k): int(v) for k, v in state.get("batch_profile", {}).items()}
        if stored_profile != self.batch_profile:
            raise ValueError("sampler hardware batch profile changed")
        orders = {int(k): list(v) for k, v in state["orders"].items()}
        cursors = {int(k): int(v) for k, v in state["cursors"].items()}
        if set(orders) != set(self.orders):
            raise ValueError("sampler bucket layout changed")
        for w in orders:
            if sorted(orders[w]) != sorted(self.orders[w]):
                raise ValueError("sampler examples changed")
        self.orders, self.cursors = orders, cursors
        self.rng.setstate(state["rng"])

    def _choose_width(self) -> int:
        total = sum(n for _, n in self.width_population)
        pick = self.rng.randrange(total)
        acc = 0
        for width, count in self.width_population:
            acc += count
            if pick < acc:
                return width
        return self.width_population[-1][0]

    def next_indices(self) -> tuple[int, list[int]]:
        width = self._choose_width()
        order = self.orders[width]
        count = self.rows_for_width(width)
        picked: list[int] = []
        while len(picked) < count:
            cursor = self.cursors[width]
            if cursor >= len(order):
                self.rng.shuffle(order)
                cursor = 0
            take = min(count - len(picked), len(order) - cursor)
            picked.extend(order[cursor:cursor + take])
            cursor += take
            self.cursors[width] = cursor
        return width, picked


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor collation/device transfer, not for pure ChatMix encoding")
    return torch


def collate_pinned(examples: list[EncodedChatExample], indices: list[int], width: int):
    t = _require_torch()
    if not indices:
        raise ValueError("empty chat batch")
    pin = bool(t.cuda.is_available())
    x = t.full((len(indices), width), TOKENIZER.pad_id, dtype=t.int16, pin_memory=pin)
    y = t.full((len(indices), width), -100, dtype=t.int16, pin_memory=pin)
    supervised = 0
    true_tokens = 0
    for r, idx in enumerate(indices):
        ex = examples[idx]
        if ex.length > width:
            raise ValueError("example exceeds selected bucket")
        n = ex.length
        x[r, :n] = t.tensor(ex.input_ids, dtype=t.int16)
        y[r, :n] = t.tensor(ex.targets, dtype=t.int16)
        supervised += ex.supervised_tokens
        true_tokens += n
    return x, y, {"width": width, "rows": len(indices), "true_tokens": true_tokens, "padded_tokens": x.numel(), "supervised_tokens": supervised}


def to_device(host_x, host_y, device):
    t = _require_torch()
    non_blocking = host_x.is_pinned() and device.type == "cuda"
    x = host_x.to(device=device, dtype=t.long, non_blocking=non_blocking)
    y = host_y.to(device=device, dtype=t.long, non_blocking=non_blocking)
    return x, y
