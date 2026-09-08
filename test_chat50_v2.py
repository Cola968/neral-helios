from __future__ import annotations

import torch
import torch.nn as nn

from chat50_v2 import TOKENIZER, TokenBudgetSampler, bucket_width, encode_turn, expand_conversation, format_history, preencode
from fetch_oasst2_multiturn50 import split_for_root, trace_chain
from fastpath50 import FusedSwiGLU50, canonical_legacy_state_dict
from prepare_chatmix50_v2 import HUMAN_TRAIN_FRACTION_MIN, human_majority_train


def chat_row():
    return {
        "id": "c1", "source": "oasst2-human-multiturn", "source_class": "human",
        "language": "de", "domain": "chat",
        "messages": [
            {"role": "user", "content": "Was ist Gravitation?"},
            {"role": "assistant", "content": "Eine Wechselwirkung zwischen Massen."},
            {"role": "user", "content": "Und auf dem Mond?"},
            {"role": "assistant", "content": "Dort ist die Fallbeschleunigung kleiner."},
        ],
    }


def test_multiturn_expands_one_terminal_example_per_assistant_turn():
    examples = expand_conversation(chat_row(), 2048)
    assert len(examples) == 2
    assert [x.assistant_turn for x in examples] == [1, 2]
    assert all(sum(t == TOKENIZER.eos_id for t in x.targets) == 1 for x in examples)


def test_second_turn_history_matches_inference_style_without_internal_eos():
    row = chat_row()
    prefix, answer = format_history(row["messages"], 3)
    assert prefix == (
        "User: Was ist Gravitation?\n"
        "Assistant: Eine Wechselwirkung zwischen Massen.\n"
        "User: Und auf dem Mond?\nAssistant:"
    )
    assert answer == "Dort ist die Fallbeschleunigung kleiner."
    ids = TOKENIZER.encode(prefix, bos=True)
    assert TOKENIZER.eos_id not in ids


def test_only_current_assistant_span_and_eos_are_supervised():
    row = chat_row()
    ex = encode_turn(row, 3, 2048)
    prefix, answer = format_history(row["messages"], 3)
    prefix_len = len(TOKENIZER.encode(prefix, bos=True))
    answer_len = len(TOKENIZER.encode(" " + answer + "\n", eos=True))
    assert all(t == -100 for t in ex.targets[: prefix_len - 1])
    assert ex.supervised_tokens == answer_len
    assert ex.targets[-1] == TOKENIZER.eos_id


def test_preencode_deduplicates_exact_prefix_examples():
    rows = [chat_row(), dict(chat_row(), id="duplicate-id")]
    examples = preencode(rows, 2048)
    assert len(examples) == 2


def test_bucket_sampler_uses_token_budget():
    examples = preencode([chat_row()], 2048)
    sampler = TokenBudgetSampler(examples, token_budget=1024, seed=1)
    width, indices = sampler.next_indices()
    assert width == bucket_width(examples[indices[0]].length)
    assert len(indices) == max(1, 1024 // width)


def test_human_majority_caps_targeted_rehearsal_deterministically():
    human = [{"source_class": "human", "source": "h", "chat_sha256": f"h{i:03}"} for i in range(60)]
    targeted = [{"source_class": "targeted-deterministic", "source": "t", "chat_sha256": f"t{i:03}"} for i in range(100)]
    selected, meta = human_majority_train(human + targeted)
    assert meta["human_fraction"] >= HUMAN_TRAIN_FRACTION_MIN
    assert meta["targeted_kept"] == 40
    assert len(selected) == 100


def human_message(mid, parent, role, text, rank=0):
    return {
        "message_id": mid, "parent_id": parent, "role": role, "text": text, "lang": "de",
        "deleted": False, "synthetic": False, "review_result": True, "model_name": None, "rank": rank,
    }


def test_oasst2_chain_requires_all_human_alternating_messages():
    rows = [
        human_message("u1", None, "prompter", "Frage"),
        human_message("a1", "u1", "assistant", "Antwort"),
        human_message("u2", "a1", "prompter", "Nachfrage"),
        human_message("a2", "u2", "assistant", "Antwort zwei"),
    ]
    messages = {r["message_id"]: r for r in rows}
    chain = trace_chain(messages, rows[-1])
    assert chain is not None and [r["role"] for r in chain] == ["prompter", "assistant", "prompter", "assistant"]
    rows[1]["synthetic"] = True
    assert trace_chain(messages, rows[-1]) is None


def test_tree_split_is_deterministic():
    assert split_for_root("same-root") == split_for_root("same-root")
    assert split_for_root("same-root") in {"train", "val", "test", "holdout"}


def test_fused_swiglu_preserves_weights_and_canonical_export_keys():
    class Legacy(nn.Module):
        def __init__(self):
            super().__init__(); self.gate = nn.Linear(8, 12, bias=False); self.up = nn.Linear(8, 12, bias=False); self.down = nn.Linear(12, 8, bias=False)
        def forward(self, x):
            return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))
    legacy = Legacy(); fused = FusedSwiGLU50.from_legacy(legacy)
    x = torch.randn(2, 3, 8)
    assert torch.allclose(legacy(x), fused(x), atol=1e-6, rtol=1e-6)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.blocks = nn.ModuleList([nn.Module()]); self.blocks[0].mlp = fused
    state = canonical_legacy_state_dict(Tiny())
    assert "blocks.0.mlp.gate.weight" in state
    assert "blocks.0.mlp.up.weight" in state
    assert "blocks.0.mlp.gate_up.weight" not in state
    assert torch.equal(state["blocks.0.mlp.gate.weight"], legacy.gate.weight)
    assert torch.equal(state["blocks.0.mlp.up.weight"], legacy.up.weight)
