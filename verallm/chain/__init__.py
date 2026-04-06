"""On-chain interaction layer for Verathos.

Provides typed Python clients for the ModelRegistry and MinerRegistry
Solidity contracts deployed on Bittensor EVM.

Usage::

    from verallm.chain import ChainConfig, ModelRegistryClient, MinerRegistryClient

    config = ChainConfig.from_env()
    model_client = ModelRegistryClient(config)
    spec = model_client.get_model_spec("Qwen/Qwen3-8B")
"""

from verallm.chain.config import ChainConfig
from verallm.chain.model_registry import ModelRegistryClient
from verallm.chain.miner_registry import MinerRegistryClient
from verallm.chain.subnet_config import SubnetConfigClient

__all__ = [
    "ChainConfig",
    "ModelRegistryClient",
    "MinerRegistryClient",
    "SubnetConfigClient",
]
