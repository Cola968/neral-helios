"""Experimental, reversible weight-space direction probe for Qwen2.5-Coder.

CPU-only contract by default. Importing this module never loads a model, trains,
opens the network, writes artifacts or changes weights. The optional projection
is a short-lived rank-one linear correction, algebraically equivalent to
W' = W - strength * (W @ u) @ u.T for a unit direction u.

NOT a reproduction of MIT CSAIL's Qwen3.5-0.8B result, NOT a tested 7B gain.
Only use with explicitly reviewed local direction vectors, external baselines,
and a dedicated, permissioned GPU evaluation. Never upload model artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch


ALLOWED_STRENGTH_MAX = 0.10


def _projection_weight(module: torch.nn.Module) -> torch.Tensor:
    """Access the real (possibly packed) parameter of a PEFT-wrapped projection."""
    weight = getattr(module, 'weight', None)
    if weight is None:
        weight = getattr(getattr(module, 'base_layer', None), 'weight', None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError('projection has no tensor weight')
    return weight


def _projection_shape(module: torch.nn.Module) -> tuple[int, int]:
    """Verify original (out, in), never derive features from packed NF4 storage.

    bitsandbytes Params4bit frequently uses a packed tensor shape such as
    (ceil(out*in/2), 1), unrelated to the input/output feature dimensions.
    quant_state.shape records the *original* matrix shape, including when
    quant_storage is floating point. A missing/inconsistent state fails closed.
    """
    base = getattr(module, 'base_layer', module)
    declared = []
    for obj in (module, base):
        n_in, n_out = getattr(obj, 'in_features', None), getattr(obj, 'out_features', None)
        if type(n_in) is int and type(n_out) is int and n_in > 0 and n_out > 0:
            declared.append((n_out, n_in))
    if not declared or len(set(declared)) != 1:
        raise ValueError('unverified projection feature dimensions')
    dimensions = declared[0]
    weight = _projection_weight(module)
    state = getattr(weight, 'quant_state', None)
    if state is None:
        state = getattr(base, 'quant_state', None)
    if state is not None:
        try:
            original = tuple(state.shape)
        except (TypeError, AttributeError):
            raise ValueError('invalid quantization shape') from None
        if original != dimensions:
            raise ValueError('quantization shape disagrees with projection features')
    elif not weight.dtype.is_floating_point or tuple(weight.shape) != dimensions:
        raise ValueError('packed/quantized projection without verified quantization shape')
    return dimensions


def resolve_down_proj(model, layer_index: int):
    """Resolve one explicit Qwen2/Qwen2.5 down-projection, optionally via PEFT.

    Only recognized paths can be used. This deliberately refuses arbitrary
    dotted module paths and all non-down_proj tensors.
    """
    if type(layer_index) is not int or layer_index < 0:
        raise IndexError('layer_index must be a nonnegative integer')
    for names in (
        ('model', 'layers'),
        ('base_model', 'model', 'model', 'layers'),
        ('base_model', 'model', 'layers'),
    ):
        current = model
        for name in names:
            current = getattr(current, name, None)
            if current is None:
                break
        if not isinstance(current, (torch.nn.ModuleList, list, tuple)):
            continue
        if layer_index >= len(current):
            raise IndexError('layer index exceeds model depth')
        mlp = getattr(current[layer_index], 'mlp', None)
        module = getattr(mlp, 'down_proj', None)
        if not isinstance(module, torch.nn.Module):
            raise ValueError('Qwen down_proj not found on selected layer')
        _projection_shape(module)
        base = getattr(module, 'base_layer', module)
        if getattr(module, 'bias', None) is not None or getattr(base, 'bias', None) is not None:
            raise ValueError('biased down_proj is not covered by the linear rank-one identity')
        return module
    raise ValueError('not a recognized Qwen2/Qwen2.5/PEFT layer structure')


@dataclass
class Moments:
    """Bounded aggregate only; no raw prompts or per-token activations stored."""

    total: torch.Tensor
    n_tokens: int

    @classmethod
    def from_summaries(cls, summed, n_tokens: int) -> 'Moments':
        if type(n_tokens) is not int or n_tokens <= 0:
            raise ValueError('nonempty token count required')
        total = torch.as_tensor(summed, dtype=torch.float32, device='cpu').detach().clone()
        if total.ndim != 1 or total.numel() == 0 or not bool(torch.isfinite(total).all()):
            raise ValueError('invalid aggregate vector')
        return cls(total, n_tokens)

    def mean(self) -> torch.Tensor:
        if self.n_tokens < 1:
            raise ValueError('no observed tokens')
        return self.total / self.n_tokens

    def add(self, input_tensor: torch.Tensor, *, token_limit: int):
        if input_tensor.ndim not in (2, 3) or input_tensor.shape[-1] != self.total.numel():
            raise ValueError('unexpected activation shape')
        flat = input_tensor.detach().reshape(-1, input_tensor.shape[-1])
        if self.n_tokens + flat.shape[0] > token_limit:
            raise ValueError('activation capture token limit exceeded')
        # Keep only the CPU aggregate; do not retain per-token data.
        tensor_sum = flat.float().sum(dim=0).cpu()
        if not bool(torch.isfinite(tensor_sum).all()):
            raise ValueError('non-finite activation')
        self.total += tensor_sum
        self.n_tokens += flat.shape[0]


@contextmanager
def capture_mlp_input(model, *, layer_index: int, token_limit: int = 512):
    """Read-only capture of means at one down-projection.

    The caller must decide independently which inputs are looping/failing and
    which are complete/successful. This utility never creates those labels.
    A protected/private prompt must not be uploaded to GitHub.
    """
    if type(token_limit) is not int or not 1 <= token_limit <= 4096:
        raise ValueError('token_limit must be 1..4096')
    if model.training:
        raise ValueError('capture requires evaluation mode')
    layer = resolve_down_proj(model, layer_index)
    if layer.training:
        raise ValueError('projection layer must be in evaluation mode')
    if layer._forward_hooks or layer._forward_pre_hooks:
        raise ValueError('capture cannot overlap an active projection hook')
    _, in_features = _projection_shape(layer)
    moments = Moments.from_summaries(torch.zeros(in_features), 1)
    moments.n_tokens = 0

    def hook(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError('missing tensor activation')
        moments.add(args[0], token_limit=token_limit)

    handle = layer.register_forward_pre_hook(hook)
    try:
        yield moments
    finally:
        handle.remove()


def contrastive_direction(failing: Moments, successful: Moments) -> torch.Tensor:
    """Normalize contrast between labeled failure and success means.

    A contrast vector is a *hypothesis* for later causal testing, not evidence
    that an internal circuit caused the observed behavior.
    """
    if failing.total.shape != successful.total.shape or failing.n_tokens < 1 or successful.n_tokens < 1:
        raise ValueError('incompatible or empty observed cohorts')
    delta = failing.mean() - successful.mean()
    length = torch.linalg.vector_norm(delta)
    if not bool(torch.isfinite(length)) or float(length) < 1e-8:
        raise ValueError('no stable nonzero contrast direction')
    return delta / length


@contextmanager
def temporary_directional_projection(model, *, layer_index: int,
                                     direction: torch.Tensor,
                                     strength: float,
                                     experimental_opt_in: bool = False):
    """Apply a small, temporary rank-one delta to a Qwen down_proj *output*.

    Mathematically same as W' = W - alpha*(W u)*u.T for this linear layer.
    Nothing in the model.state_dict() or original HF/safetensors changes.
    Never use on a shared active serving model. Always test quality and rollback.
    """
    if not experimental_opt_in:
        raise PermissionError('explicit experimental opt-in required')
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 < strength <= ALLOWED_STRENGTH_MAX:
        raise ValueError('strength must be in (0, 0.10]')
    if model.training:
        raise ValueError('edit is only allowed during evaluation')
    module = resolve_down_proj(model, layer_index)
    if module.training:
        raise ValueError('down_proj must be in evaluation mode')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('all base/adapter parameters must be frozen')
    if module._forward_hooks or module._forward_pre_hooks:
        raise ValueError('selected layer already has a hook; refuse unsafe stacking')
    vector = torch.as_tensor(direction, dtype=torch.float32, device='cpu').detach().clone()
    out_features, in_features = _projection_shape(module)
    if vector.ndim != 1 or vector.numel() != in_features:
        raise ValueError('direction size does not match down_proj input')
    if not bool(torch.isfinite(vector).all()):
        raise ValueError('direction is not finite')
    norm = torch.linalg.vector_norm(vector)
    if float(norm) < 1e-8:
        raise ValueError('direction must be nonzero')
    unit = vector / norm
    # A 4-bit BitsAndBytes weight often reports uint8 storage. Its compute path
    # still accepts BF16 inputs in the verified NERAL inference setup.
    real_weight = _projection_weight(module)
    weight_dtype = real_weight.dtype
    probe_dtype = weight_dtype if weight_dtype.is_floating_point else torch.bfloat16
    unit_on_device = unit.to(device=real_weight.device, dtype=probe_dtype)
    # Evaluate W@u once *before* registering the hook, using the same quantized
    # forward computation as the actual module. No full 7B dequantization.
    with torch.inference_mode():
        output_direction = module(unit_on_device.unsqueeze(0)).squeeze(0).detach().clone()
    if output_direction.ndim != 1 or output_direction.numel() != out_features:
        raise ValueError('invalid down_proj output dimension')
    if not bool(torch.isfinite(output_direction.float()).all()):
        raise ValueError('nonfinite projected output direction')

    def edit(_module, args, output):
        if not args or not isinstance(args[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ValueError('projection hook received unexpected inputs')
        x = args[0]
        local_u = unit_on_device.to(dtype=x.dtype, device=x.device)
        local_v = output_direction.to(dtype=output.dtype, device=output.device)
        # shape (..., input_features) -> (..., 1)
        coeff = torch.sum(x * local_u, dim=-1, keepdim=True)
        return output - strength * coeff.to(dtype=output.dtype) * local_v

    handle = module.register_forward_hook(edit)
    try:
        yield
    finally:
        handle.remove()
