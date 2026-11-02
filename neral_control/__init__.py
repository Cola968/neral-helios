"""Opt-in NERAL control API. Import does not load weights or run tools."""
from .controller import ReasoningController, ControllerRunResult, FAST, THINK, DEEP

__all__ = ['ReasoningController', 'ControllerRunResult', 'FAST', 'THINK', 'DEEP']


def create_shared_controller(registry=None, provider=None, *, allow_model_inference=False,
                             context_manager=None, **controller_options):
    """Construct only; no registration, model load, query or task execution.

    An explicit provider needs an explicit inference opt-in. Artifact checks are
    owned by the provider constructor; paths never come from untrusted model text.
    """
    if type(allow_model_inference) is not bool:
        raise ValueError('Model opt-in must be a boolean.')
    if provider is not None and not allow_model_inference:
        raise ValueError('Explicit model inference opt-in required.')
    if registry is None:
        from .registry import default_registry
        registry = default_registry()
    return ReasoningController(registry, provider, context_manager=context_manager, **controller_options)


__all__.append('create_shared_controller')
