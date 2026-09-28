from .config import Helios180Config

__all__ = ["Helios180Config", "Helios180ForCausalLM"]

def __getattr__(name):
    if name == "Helios180ForCausalLM":
        from .model import Helios180ForCausalLM
        return Helios180ForCausalLM
    raise AttributeError(name)
