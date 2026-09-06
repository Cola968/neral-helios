import torch

from config import NeralConfig
from model import Neral20M
from model50 import Neral50M


def tiny_config(head_dim=16, max_seq_len=64):
    return NeralConfig(
        vocab_size=260,
        d_model=head_dim * 2,
        n_layers=2,
        n_heads=2,
        ffn_hidden=head_dim * 4,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )


def assert_cached_matches_full(model, tokens, atol=2e-5, rtol=2e-5):
    model.eval()
    with torch.inference_mode():
        full = model(tokens)[0]
        prefix = min(5, tokens.size(1) - 1)
        cached, state = model.forward_cached(tokens[:, :prefix])
        assert torch.allclose(cached, full[:, :prefix], atol=atol, rtol=rtol)
        for position in range(prefix, tokens.size(1)):
            cached, state = model.forward_cached(tokens[:, position : position + 1], state)
            assert torch.allclose(
                cached[:, -1, :], full[:, position, :], atol=atol, rtol=rtol
            )


def test_20m_attention_cache_matches_full_forward():
    torch.manual_seed(9681)
    model = Neral20M(tiny_config())
    x = torch.randint(0, 260, (2, 19))
    assert_cached_matches_full(model, x)


def test_padded_52_dim_attention_cache_matches_full_forward():
    # Neral-50M's real head dimension is 52; exercise exactly that padding geometry while
    # keeping the CPU test model tiny (two heads, two layers).
    torch.manual_seed(9682)
    model = Neral50M(tiny_config(head_dim=52))
    x = torch.randint(0, 260, (1, 17))
    assert_cached_matches_full(model, x, atol=4e-5, rtol=4e-5)


def test_cached_greedy_generation_matches_legacy_path():
    torch.manual_seed(9683)
    model = Neral50M(tiny_config(head_dim=52, max_seq_len=32)).eval()
    prompt = torch.randint(4, 260, (1, 11))
    with torch.inference_mode():
        legacy = model.generate(
            prompt.clone(), max_new_tokens=12, temperature=0, eos_id=None, use_cache=False
        )
        cached = model.generate(
            prompt.clone(), max_new_tokens=12, temperature=0, eos_id=None, use_cache=True
        )
    assert torch.equal(cached, legacy)


def test_cached_sliding_window_matches_legacy_path():
    torch.manual_seed(9684)
    model = Neral20M(tiny_config(head_dim=16, max_seq_len=8)).eval()
    prompt = torch.randint(4, 260, (1, 8))
    with torch.inference_mode():
        legacy = model.generate(
            prompt.clone(), max_new_tokens=5, temperature=0, eos_id=None, use_cache=False
        )
        cached = model.generate(
            prompt.clone(), max_new_tokens=5, temperature=0, eos_id=None, use_cache=True
        )
    assert torch.equal(cached, legacy)


def test_runtime_caches_do_not_change_checkpoint_or_parameter_count():
    torch.manual_seed(9685)
    model = Neral20M(tiny_config()).eval()
    before = tuple(model.state_dict().keys())
    params = sum(p.numel() for p in model.parameters())
    x = torch.randint(0, 260, (1, 7))
    with torch.inference_mode():
        model.forward_cached(x)
    assert tuple(model.state_dict().keys()) == before
    assert sum(p.numel() for p in model.parameters()) == params
    assert not any("cache" in key.lower() for key in model.state_dict())
