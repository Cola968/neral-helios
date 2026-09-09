import torch

from orion100 import NeralOrion, expected_parameters
from orion_a5 import ORION_A5, a5_report


def test_orion_a5_exact_contract():
    assert ORION_A5.d_model == 768
    assert ORION_A5.n_layers == 12
    assert ORION_A5.n_heads == 12
    assert ORION_A5.n_kv_heads == 2
    assert ORION_A5.head_dim == 64
    assert ORION_A5.ffn_hidden == 3008
    assert ORION_A5.max_seq_len == 8192
    assert expected_parameters(ORION_A5) == 99_900_672


def test_orion_a5_report_and_kv_budget():
    report = a5_report()
    assert report["parameters"] == 99_900_672
    assert report["linear_gemm_calls_per_forward"] == 48
    # 12 layers * K/V * 2 KV heads * 4096 positions * 64 head dim * bf16 bytes
    assert report["kv_cache_bf16_4096_bytes"] == 25_165_824
    assert report["kv_cache_bf16_8192_bytes"] == 50_331_648


def test_orion_a5_small_proxy_backward_cpu():
    # Same attention/FFN ratios scaled down for cheap CPU contract coverage.
    from orion100 import OrionConfig
    proxy = OrionConfig(
        d_model=96,
        n_layers=2,
        n_heads=3,
        n_kv_heads=1,
        ffn_hidden=376,
        max_seq_len=64,
    )
    model = NeralOrion(proxy).train()
    x = torch.randint(0, proxy.vocab_size, (2, 16))
    y = torch.randint(0, proxy.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, proxy.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
