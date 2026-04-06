"""
VeraLLM Validator — proof verification orchestration.

Provides the Validator class for verifying inference proofs
against on-chain model specifications.
"""


def __getattr__(name):
    if name == "Validator":
        from verallm.validator.core import Validator
        return Validator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Validator"]
