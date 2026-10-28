"""Explicit opt-in full precision master weights with BF16 CUDA compute.

The native Arctus350 old SFT mode is preserved by default. No pretrained
third-party weights, optimizer execution, or GPU job is started on import.
"""
from __future__ import annotations


def validate_precision_mode(mode: str) -> str:
    if type(mode) is not str or mode not in ("legacy-bf16", "fp32-master"):
        raise ValueError("precision mode must be legacy-bf16 or fp32-master")
    return mode


def forward_with_amp(model, inputs, targets=None, *, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    param = next(model.parameters())
    master = param.dtype == torch_module.float32 and param.device.type == "cuda"
    # FP32 master parameters/AdamW states remain FP32. The forward and backward
    # activations use BF16 CUDA compute; no GradScaler required for BF16.
    with torch_module.autocast("cuda", dtype=torch_module.bfloat16, enabled=master):
        if targets is None:
            return model(inputs)
        return model(inputs, targets)


def checkpoint_model_state(model, mode: str, *, resume: bool = False, torch_module=None):
    mode=validate_precision_mode(mode)
    if mode=="legacy-bf16":
        # Same state_dict semantics as the historical model_payload.
        return model.state_dict()
    if torch_module is None:
        import torch as torch_module
    state=model.state_dict()
    # Preserve the FP32 masters in the optimizer-resume checkpoint. BF16 is
    # exported only for the immutable selected inference candidate.
    dtype=torch_module.float32 if resume else torch_module.bfloat16
    return {name: value.detach().to(dtype).cpu() for name,value in state.items()}
