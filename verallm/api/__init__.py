"""
VeraLLM API — HTTP layer for miner-validator communication.

Provides:
- server: FastAPI miner server (python -m verallm.api.server)
- client: ValidatorClient (python -m verallm.api.client)
- serialization: JSON protocol serialization
"""


def __getattr__(name):
    """Lazy imports to avoid circular/order issues with python -m."""
    if name == "ValidatorClient":
        from verallm.api.client import ValidatorClient
        return ValidatorClient
    if name in ("to_dict", "from_dict", "model_spec_to_dict", "dict_to_model_spec"):
        from verallm.api import serialization
        return getattr(serialization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ValidatorClient",
    "to_dict",
    "from_dict",
    "model_spec_to_dict",
    "dict_to_model_spec",
]
