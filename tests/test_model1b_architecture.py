import torch

from helios1b import HeliosConfig, HeliosForCausalLM


def test_production_contract_is_about_one_billion_parameters():
    cfg = HeliosConfig()
    assert cfg.n_layers == 32
    assert cfg.n_heads == 12
    assert cfg.n_kv_heads == 4
    assert cfg.head_dim == 128
    assert cfg.d_model == 1536
    assert cfg.vocab_size == 32768
    assert cfg.parameter_count() == 1_006_740_992


def test_baseline_uses_rope_everywhere():
    cfg = HeliosConfig()
    assert all(cfg.layer_uses_rope(i) for i in range(cfg.n_layers))


def test_nope_is_opt_in_ablation():
    cfg = HeliosConfig().with_nope_ablation(4)
    assert cfg.layer_uses_rope(0)
    assert cfg.layer_uses_rope(2)
    assert not cfg.layer_uses_rope(3)
    assert not cfg.layer_uses_rope(7)


def test_tiny_proxy_forward_backward_is_finite():
    cfg = HeliosConfig.tiny_proxy()
    model = HeliosForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, y)

    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert model.tok_emb.weight.grad is not None


def test_tiny_proxy_analytical_count_matches_materialized_model():
    cfg = HeliosConfig.tiny_proxy()
    model = HeliosForCausalLM(cfg)
    report = model.parameter_report()
    assert report["actual"] == report["analytical"]


def test_gradient_checkpointing_preserves_finite_backward():
    cfg = HeliosConfig.tiny_proxy()
    model = HeliosForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, y)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert model.blocks[0].attn.q_proj.weight.grad is not None
    assert torch.isfinite(model.blocks[0].attn.q_proj.weight.grad).all()
