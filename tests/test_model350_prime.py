import torch

from frontier350.model350_prime import (
    ArctusPrime350Config,
    ArctusPrime350M,
    PRIME_350,
    PRIME_350_PARAMETER_COUNT,
)


def test_prime350_locked_shape_contract():
    assert PRIME_350.d_model == 1024
    assert PRIME_350.n_layers == 28
    assert PRIME_350.n_heads == 16
    assert PRIME_350.n_kv_heads == 4
    assert PRIME_350.head_dim == 64
    assert PRIME_350.ffn_hidden == 3008
    assert PRIME_350.vocab_size == 16_384
    assert PRIME_350.max_seq_len == 8192
    assert PRIME_350.gqa_groups == 4
    assert PRIME_350.analytical_parameter_count() == PRIME_350_PARAMETER_COUNT
    assert PRIME_350_PARAMETER_COUNT == 348_975_616


def test_prime350_tiny_forward_and_tied_embeddings():
    cfg = ArctusPrime350Config(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=256,
        max_seq_len=32,
    )
    model = ArctusPrime350M(cfg)
    assert model.lm_head.weight is model.tok_emb.weight

    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_prime350_analytical_count_matches_tiny_model():
    cfg = ArctusPrime350Config(
        vocab_size=128,
        d_model=128,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=256,
        max_seq_len=16,
    )
    model = ArctusPrime350M(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == cfg.analytical_parameter_count()
