"""CPU-only RED/GREEN contract tests. No model downloads, trained weights or GPU use."""
import math
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.neral_weightspace_probe import (
    Moments, capture_mlp_input, contrastive_direction,
    temporary_directional_projection, resolve_down_proj,
)


class OneBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 0., 2.]]))


class TinyQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([OneBlock(), OneBlock()])
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.model.layers[0].mlp.down_proj(x)


class FakePackedDownProj(torch.nn.Module):
    """Synthetic CPU stand-in for a 4-bit Linear4bit packed Params4bit.

    The packed weight.shape is NOT [out_features, in_features]. The original
    weight dimensions are declared by in_features, out_features and quant_state.
    This fake simulates that structural property; it does not test real NF4.
    """

    def __init__(self, *, state_shape=(2, 3), storage_dtype=torch.uint8):
        super().__init__()
        self.in_features, self.out_features = 3, 2
        self.weight = torch.nn.Parameter(torch.zeros((3, 1), dtype=storage_dtype),
                                         requires_grad=False)
        if state_shape is not None:
            self.weight.quant_state = SimpleNamespace(shape=state_shape)
        self.register_buffer('reference_matrix', torch.tensor([[1., 2., 3.], [-1., 0., 2.]]))

    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.reference_matrix)


class FakePeftWrapper(torch.nn.Module):
    """Simple proxy that exposes packed weight only through base_layer."""

    def __init__(self):
        super().__init__()
        self.base_layer = FakePackedDownProj()

    def forward(self, x):
        return self.base_layer(x)


class WeightspaceTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyQwen()
        self.x = torch.tensor([[2.0, 3.0, -2.0], [-1.0, 1.0, 2.0]])
        self.u = torch.tensor([3.0, 4.0, 0.0])  # normalized internally

    def test_reversible_and_exact_rank_one_equivalence(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        original = self.model(self.x).clone()
        with temporary_directional_projection(
            self.model, layer_index=0, direction=self.u, strength=0.1,
            experimental_opt_in=True
        ):
            candidate = self.model(self.x)
            weight = self.model.model.layers[0].mlp.down_proj.weight.detach()
            unit = self.u / self.u.norm()
            projected_weight = weight - 0.1 * torch.outer(weight @ unit, unit)
            torch.testing.assert_close(candidate, self.x @ projected_weight.T)
            self.assertFalse(torch.allclose(original, candidate))
        torch.testing.assert_close(self.model(self.x), original)
        for k, v in before.items():
            torch.testing.assert_close(self.model.state_dict()[k], v)

    def test_exception_restores_baseline(self):
        base = self.model(self.x).clone()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with temporary_directional_projection(self.model, layer_index=0, direction=self.u,
                                                  strength=0.05, experimental_opt_in=True):
                raise RuntimeError('intentional')
        torch.testing.assert_close(self.model(self.x), base)

    def test_explicit_opt_in_required(self):
        with self.assertRaises(PermissionError):
            with temporary_directional_projection(self.model, layer_index=0,
                                                  direction=self.u, strength=0.05):
                pass

    def test_rejects_invalid_indices_and_direction_before_any_hook(self):
        block = resolve_down_proj(self.model, layer_index=0)
        before = len(block._forward_hooks)
        for idx, vector in [(2, self.u), (-1, self.u), (0, torch.ones(2)),
                            (0, torch.tensor([math.nan, 1., 0.])), (0, torch.zeros(3))]:
            with self.assertRaises((ValueError, IndexError)):
                with temporary_directional_projection(self.model, layer_index=idx,
                         direction=vector, strength=0.05, experimental_opt_in=True):
                    pass
        self.assertEqual(len(block._forward_hooks), before)

    def test_rejects_high_strength_and_training_mode(self):
        for strength in [-0.01, 0, 0.11, float('nan')]:
            with self.assertRaises(ValueError):
                with temporary_directional_projection(self.model, layer_index=0,
                    direction=self.u, strength=strength, experimental_opt_in=True):
                    pass
        self.model.train()
        with self.assertRaises(ValueError):
            with temporary_directional_projection(self.model, layer_index=0,
                direction=self.u, strength=0.05, experimental_opt_in=True):
                pass

    def test_capture_moments_only_not_raw_activations(self):
        with capture_mlp_input(self.model, layer_index=0, token_limit=10) as moments:
            self.model(self.x)
        self.assertEqual(moments.n_tokens, 2)
        torch.testing.assert_close(moments.mean(), self.x.mean(dim=0))
        self.assertEqual(moments.total.shape, (3,))
        self.assertFalse(hasattr(moments, 'raw_activations'))
        self.assertEqual(len(resolve_down_proj(self.model, 0)._forward_pre_hooks), 0)

    def test_capture_token_cap_fails_closed_and_unhooks(self):
        with self.assertRaises(ValueError):
            with capture_mlp_input(self.model, layer_index=0, token_limit=1):
                self.model(self.x)
        self.assertEqual(len(resolve_down_proj(self.model, 0)._forward_pre_hooks), 0)

    def test_contrastive_direction_fits_and_rejects_degenerate(self):
        good = Moments.from_summaries([4., 2., 0.], 2)
        bad = Moments.from_summaries([6., 2., 0.], 2)
        direction = contrastive_direction(bad, good)
        torch.testing.assert_close(direction, torch.tensor([1., 0., 0.]))
        with self.assertRaises(ValueError):
            contrastive_direction(good, good)
        with self.assertRaises(ValueError):
            contrastive_direction(Moments.from_summaries([1., 2.], 1), good)

    def test_reject_biased_down_projection_to_preserve_rank_one_equivalence(self):
        layer = self.model.model.layers[0].mlp.down_proj
        biased = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            biased.weight.copy_(layer.weight)
            biased.bias.fill_(0.25)
        biased.requires_grad_(False)
        biased.eval()
        self.model.model.layers[0].mlp.down_proj = biased
        with self.assertRaisesRegex(ValueError, 'bias'):
            with temporary_directional_projection(
                self.model, layer_index=0, direction=self.u,
                strength=0.05, experimental_opt_in=True):
                pass
        self.assertEqual(len(biased._forward_hooks), 0)

    def test_reject_capture_during_active_weight_patch(self):
        with temporary_directional_projection(
            self.model, layer_index=0, direction=self.u,
            strength=0.05, experimental_opt_in=True):
            with self.assertRaisesRegex(ValueError, 'active'):
                with capture_mlp_input(self.model, layer_index=0):
                    pass
        self.assertEqual(len(resolve_down_proj(self.model, 0)._forward_hooks), 0)

    def test_packed_4bit_weight_shape_uses_original_quantization_dimensions(self):
        model = TinyQwen()
        layer = FakePackedDownProj()
        model.model.layers[0].mlp.down_proj = layer
        model.eval()
        self.assertEqual(tuple(layer.weight.shape), (3, 1))
        self.assertEqual(layer.in_features, 3)
        self.assertEqual(layer.out_features, 2)
        self.assertIs(resolve_down_proj(model, 0), layer)
        with capture_mlp_input(model, layer_index=0, token_limit=4) as moments:
            output_before = model(self.x)
        torch.testing.assert_close(moments.mean(), self.x.mean(dim=0))
        with temporary_directional_projection(model, layer_index=0,
                                              direction=self.u, strength=0.10,
                                              experimental_opt_in=True):
            candidate = model(self.x)
        unit = self.u / self.u.norm()
        expected_weight = layer.reference_matrix - 0.10 * torch.outer(layer.reference_matrix @ unit, unit)
        torch.testing.assert_close(candidate, self.x @ expected_weight.T, rtol=0.006, atol=0.006)
        torch.testing.assert_close(model(self.x), output_before)

    def test_rejects_packed_quantized_weight_without_original_shape(self):
        model = TinyQwen()
        model.model.layers[0].mlp.down_proj = FakePackedDownProj(state_shape=None)
        with self.assertRaisesRegex(ValueError, 'quant'):
            resolve_down_proj(model, 0)

    def test_rejects_corrupted_quantized_shape(self):
        model = TinyQwen()
        model.model.layers[0].mlp.down_proj = FakePackedDownProj(state_shape=(4, 3))
        with self.assertRaisesRegex(ValueError, 'shape'):
            resolve_down_proj(model, 0)

    def test_floating_point_packed_quant_storage_still_uses_quant_state(self):
        model = TinyQwen()
        layer = FakePackedDownProj(storage_dtype=torch.bfloat16)
        model.model.layers[0].mlp.down_proj = layer
        model.eval()
        self.assertIs(resolve_down_proj(model, 0), layer)
        with temporary_directional_projection(model, layer_index=0,
                                              direction=self.u, strength=0.05,
                                              experimental_opt_in=True):
            candidate = model(self.x)
            self.assertEqual(tuple(candidate.shape), (2, 2))

    def test_peft_wrapper_reads_packed_base_layer_dimensions(self):
        model = TinyQwen()
        layer = FakePeftWrapper()
        model.model.layers[0].mlp.down_proj = layer
        model.eval()
        baseline = model(self.x).clone()
        self.assertIs(resolve_down_proj(model, 0), layer)
        with temporary_directional_projection(model, layer_index=0,
                                              direction=self.u, strength=0.05,
                                              experimental_opt_in=True):
            changed = model(self.x)
            self.assertFalse(torch.allclose(changed, baseline))
        torch.testing.assert_close(model(self.x), baseline)

    def test_base_without_edit_is_unchanged_and_no_io(self):
        base = self.model(self.x).clone()
        _ = contrastive_direction(Moments.from_summaries([1., 1., 0.], 2),
                                  Moments.from_summaries([1., 0., 0.], 2))
        torch.testing.assert_close(self.model(self.x), base)


if __name__ == '__main__':
    unittest.main()
