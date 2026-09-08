import torch

from architectures100 import HELIX_100_EXPECTED_PARAMETERS, expected_parameters100
from model100 import HELIX_100, Neral100Config, Neral100M


def tiny():
    return Neral100Config(
        vocab_size=260,
        d_model=64,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=64,
        rope_theta=10000.0,
    )


def test_exact_helix100_parameter_contract():
    assert expected_parameters100(HELIX_100) == HELIX_100_EXPECTED_PARAMETERS == 99_783_168
    model = Neral100M(HELIX_100)
    assert model.parameter_report()["total"] == HELIX_100_EXPECTED_PARAMETERS


def test_compact_gqa_cache_uses_only_kv_heads():
    model = Neral100M(tiny()).eval()
    idx = torch.randint(0, 260, (1, 5))
    _, cache = model.forward_cached(idx)
    assert len(cache) == 2
    k, v = cache[0]
    assert k.shape == v.shape == (1, 2, 5, 8)


def test_cached_logits_match_full_forward():
    torch.manual_seed(7)
    model = Neral100M(tiny()).eval()
    idx = torch.randint(0, 260, (1, 6))
    full, _ = model(idx)
    _, cache = model.forward_cached(idx[:, :-1])
    nxt, _ = model.forward_cached(idx[:, -1:], cache)
    torch.testing.assert_close(full[:, -1], nxt[:, -1], rtol=2e-4, atol=2e-4)


def test_causal_prefix_is_unchanged_by_future_bytes():
    torch.manual_seed(8)
    model = Neral100M(tiny()).eval()
    prefix = torch.randint(0, 260, (1, 4))
    suffix = torch.randint(0, 260, (1, 3))
    short, _ = model(prefix)
    long, _ = model(torch.cat((prefix, suffix), dim=1))
    torch.testing.assert_close(short, long[:, :4], rtol=2e-4, atol=2e-4)


def test_activation_checkpointing_backward_is_finite():
    torch.manual_seed(9)
    model = Neral100M(tiny()).train()
    model.set_gradient_checkpointing(True)
    idx = torch.randint(0, 260, (2, 12))
    target = torch.randint(0, 260, (2, 12))
    _, loss = model(idx, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_helix_kv_cache_is_smaller_than_legacy_50m_at_same_4k_context():
    model = Neral100M(HELIX_100)
    helix = model.estimated_kv_cache_bytes(4096)
    legacy_50m = 2 * 8 * 12 * 52 * 4096 * 2
    assert helix == 64 * 1024 * 1024
    assert helix < legacy_50m
