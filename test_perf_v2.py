import torch

from config import NeralConfig
from fastpath50 import FusedSwiGLU50
from model import SwiGLU
from orion100 import ORION_A3, ORION_A4, OrionConfig, NeralOrion, expected_parameters


def test_fused_swiglu_preserves_weights_and_output_cpu():
    torch.manual_seed(7)
    cfg = NeralConfig(d_model=16, n_heads=4, ffn_hidden=32, max_seq_len=32)
    legacy = SwiGLU(cfg)
    fused = FusedSwiGLU50.from_legacy(legacy)
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        a = legacy(x)
        b = fused(x)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in fused.parameters())


def test_orion_a3_contract():
    assert ORION_A3.d_model == 768
    assert ORION_A3.n_layers == 14
    assert ORION_A3.n_heads == 12
    assert ORION_A3.n_kv_heads == 2
    assert ORION_A3.head_dim == 64
    assert ORION_A3.max_seq_len == 8192
    assert expected_parameters(ORION_A3) == 100_002_304


def test_orion_a4_contract():
    assert ORION_A4.d_model == 768
    assert ORION_A4.n_layers == 16
    assert ORION_A4.head_dim == 64
    assert expected_parameters(ORION_A4) == 100_103_936


def test_orion_small_forward_and_gradient_checkpoint_path_cpu():
    cfg = OrionConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        ffn_hidden=128,
        max_seq_len=64,
    )
    model = NeralOrion(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)
    model.set_gradient_checkpointing(True)
    logits2, loss2 = model(x, y)
    loss2.backward()
    assert logits2.shape == logits.shape


def test_orion_compact_kv_formula():
    cfg = OrionConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=1, ffn_hidden=128, max_seq_len=64)
    model = NeralOrion(cfg)
    expected = 1 * 2 * 2 * 1 * 64 * 16 * 2
    assert model.estimated_kv_cache_bytes(64) == expected
