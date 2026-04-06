"""
litellm-verathos -- LiteLLM custom provider for Verathos verified inference.

Usage::

    import litellm
    from litellm_verathos import VerathosProvider

    # Register once at startup
    VerathosProvider.register()

    # Then use the verathos/ prefix
    response = litellm.completion(
        model="verathos/auto",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key="vrt_sk_...",
    )
"""

from litellm_verathos.provider import VerathosProvider

__all__ = ["VerathosProvider"]
__version__ = "0.1.0"
