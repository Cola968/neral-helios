from .config import HeliosConfig

__all__ = ["HeliosConfig", "HeliosForCausalLM"]


def __getattr__(name: str):
    """Keep data/tokenizer tooling usable in environments without PyTorch."""
    if name == "HeliosForCausalLM":
        from .model import HeliosForCausalLM

        return HeliosForCausalLM
    raise AttributeError(name)
