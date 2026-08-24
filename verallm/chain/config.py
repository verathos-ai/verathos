"""Chain configuration for Verathos contracts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Map of field names to their expected Python types (avoids fragile string comparison)
_FIELD_TYPES = {
    "chain_id": int,
    "netuid": int,
    "max_retries": int,
    "retry_delay": float,
    "model_spec_cache_ttl": float,
    "miner_list_cache_ttl": float,
    "mock": bool,
}


def _validate_address(value: str, field_name: str) -> None:
    """Raise ValueError if value is not a valid EVM address."""
    if value and not _ADDRESS_RE.match(value):
        raise ValueError(
            f"Invalid {field_name}: {value!r}. "
            f"Expected 0x-prefixed 40-hex-char EVM address."
        )


@dataclass
class ChainConfig:
    """Configuration for connecting to Bittensor EVM and interacting with contracts.

    All fields can be overridden via environment variables prefixed with ``VERATHOS_``.
    """

    rpc_url: str = "https://test.chain.opentensor.ai"
    chain_id: int = 945  # Bittensor testnet
    netuid: int = 0

    model_registry_address: str = ""
    miner_registry_address: str = ""
    payment_gateway_address: str = ""
    validator_registry_address: str = ""
    checkpoint_registry_address: str = ""
    subnet_config_address: str = ""

    # Public content-addressed proof-v2 artifact stores. The release chain
    # configs may provide a primary URL followed by read-only mirrors.
    proof_v2_artifact_base_urls: tuple[str, ...] = ()
    proof_v2_artifact_cache_dir: str = ""
    proof_v3_artifact_base_urls: tuple[str, ...] = ()
    proof_v3_artifact_cache_dir: str = ""

    # Content-addressed mesh tensor-manifest distribution (same style and,
    # by default, the same hosting directory as the GLEIPNIR artifact
    # store: the two stores use different index filenames, so they coexist
    # under one base URL). Discovery metadata only; every download is
    # verified against the on-chain committed tensor manifest root.
    mesh_manifest_base_urls: tuple = ()

    # EVM private key for signing transactions (hex string, no 0x prefix)
    evm_private_key: str = ""

    # Retry / resilience
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds, doubled on each retry

    # Cache TTLs
    model_spec_cache_ttl: float = 3600.0  # 1 hour (specs change rarely)
    miner_list_cache_ttl: float = 300.0  # 5 minutes

    # Use mock provider for dev/testing (no chain needed)
    mock: bool = False

    def __post_init__(self):
        if isinstance(self.proof_v2_artifact_base_urls, str):
            raise ValueError(
                "proof_v2_artifact_base_urls must be a list of URLs"
            )
        try:
            artifact_urls = tuple(self.proof_v2_artifact_base_urls)
        except TypeError as exc:
            raise ValueError(
                "proof_v2_artifact_base_urls must be a list of URLs"
            ) from exc
        if any(not isinstance(item, str) or not item for item in artifact_urls):
            raise ValueError(
                "proof_v2_artifact_base_urls must contain non-empty URLs"
            )
        self.proof_v2_artifact_base_urls = artifact_urls
        if not isinstance(self.proof_v2_artifact_cache_dir, str):
            raise ValueError("proof_v2_artifact_cache_dir must be a string")
        if isinstance(self.proof_v3_artifact_base_urls, str):
            raise ValueError(
                "proof_v3_artifact_base_urls must be a list of URLs"
            )
        try:
            proof_v3_urls = tuple(self.proof_v3_artifact_base_urls)
        except TypeError as exc:
            raise ValueError(
                "proof_v3_artifact_base_urls must be a list of URLs"
            ) from exc
        if any(not isinstance(item, str) or not item for item in proof_v3_urls):
            raise ValueError(
                "proof_v3_artifact_base_urls must contain non-empty URLs"
            )
        self.proof_v3_artifact_base_urls = proof_v3_urls
        if not isinstance(self.proof_v3_artifact_cache_dir, str):
            raise ValueError("proof_v3_artifact_cache_dir must be a string")
        _validate_address(self.model_registry_address, "model_registry_address")
        _validate_address(self.miner_registry_address, "miner_registry_address")
        _validate_address(self.payment_gateway_address, "payment_gateway_address")
        _validate_address(self.validator_registry_address, "validator_registry_address")
        _validate_address(self.checkpoint_registry_address, "checkpoint_registry_address")
        _validate_address(self.subnet_config_address, "subnet_config_address")
        urls = self.mesh_manifest_base_urls
        if isinstance(urls, str):
            raise ValueError("mesh_manifest_base_urls must be a list of URLs")
        urls = tuple(str(url).strip().rstrip("/") for url in (urls or ()))
        for url in urls:
            if not url.startswith(("https://", "http://")):
                raise ValueError(
                    f"mesh_manifest_base_urls entries must be http(s): {url!r}"
                )
        object.__setattr__(self, "mesh_manifest_base_urls", urls)

    def require_addresses(self) -> None:
        """Raise if contract addresses are not configured.

        Call this before creating registry clients.
        """
        if not self.model_registry_address:
            raise ValueError(
                "model_registry_address not set. "
                "Deploy contracts first (scripts/deploy_testnet.py) and provide "
                "--chain-config or set VERATHOS_MODEL_REGISTRY."
            )
        if not self.miner_registry_address:
            raise ValueError(
                "miner_registry_address not set. "
                "Deploy contracts first (scripts/deploy_testnet.py) and provide "
                "--chain-config or set VERATHOS_MINER_REGISTRY."
            )

    @classmethod
    def from_env(cls, **overrides) -> ChainConfig:
        """Build config from environment variables with optional overrides."""
        env_map = {
            "rpc_url": "VERATHOS_RPC_URL",
            "chain_id": "VERATHOS_CHAIN_ID",
            "netuid": "VERATHOS_NETUID",
            "model_registry_address": "VERATHOS_MODEL_REGISTRY",
            "miner_registry_address": "VERATHOS_MINER_REGISTRY",
            "payment_gateway_address": "VERATHOS_PAYMENT_GATEWAY",
            "validator_registry_address": "VERATHOS_VALIDATOR_REGISTRY",
            "checkpoint_registry_address": "VERATHOS_CHECKPOINT_REGISTRY",
            "subnet_config_address": "VERATHOS_SUBNET_CONFIG",
            "evm_private_key": "VERATHOS_EVM_PRIVATE_KEY",
            "max_retries": "VERATHOS_MAX_RETRIES",
            "model_spec_cache_ttl": "VERATHOS_MODEL_SPEC_CACHE_TTL",
            "miner_list_cache_ttl": "VERATHOS_MINER_LIST_CACHE_TTL",
            "mock": "VERATHOS_MOCK",
        }

        kwargs: dict = {}
        for attr, env_var in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                target_type = _FIELD_TYPES.get(attr)
                if target_type is int:
                    kwargs[attr] = int(val)
                elif target_type is float:
                    kwargs[attr] = float(val)
                elif target_type is bool:
                    kwargs[attr] = val.lower() in ("1", "true", "yes")
                else:
                    kwargs[attr] = val

        kwargs.update(overrides)
        return cls(**kwargs)

    # Bundled config filenames (relative to repo root)
    _NETWORK_CONFIGS = {
        "test": "chain_config_testnet.json",
        "finney": "chain_config_mainnet.json",
    }

    # Default EVM RPC URLs per network (used when rpc_url not in JSON or CLI).
    _NETWORK_RPC_URLS = {
        "test": "https://test.chain.opentensor.ai",
        "finney": "https://lite.chain.opentensor.ai",
    }

    _PUBLIC_RPC_HOSTS = frozenset({
        "lite.chain.opentensor.ai",
        "test.chain.opentensor.ai",
        "entrypoint-finney.opentensor.ai",
        "entrypoint-test-finney.opentensor.ai",
    })

    @classmethod
    def resolve_config_path(cls, chain_config: str | None, subtensor_network: str | None) -> str | None:
        """Resolve a chain config path from explicit path or network name.

        Priority: explicit ``--chain-config`` > ``--subtensor.network`` > None.
        Network names (``test``, ``finney``) are mapped to bundled JSON files
        in the repo root.

        Returns the resolved file path, or None if neither is provided.
        """
        from pathlib import Path

        if chain_config:
            return chain_config

        if subtensor_network:
            filename = cls._NETWORK_CONFIGS.get(subtensor_network)
            if filename is None:
                raise ValueError(
                    f"Unknown network '{subtensor_network}'. "
                    f"Expected: {', '.join(cls._NETWORK_CONFIGS.keys())}. "
                    f"For local subtensor, use --subtensor-network test/finney "
                    f"+ --subtensor-chain-endpoint http://localhost:9944"
                )
            # Search: cwd first, then repo root (3 levels up from this file)
            for base in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
                candidate = base / filename
                if candidate.exists():
                    return str(candidate)
            raise FileNotFoundError(
                f"Chain config '{filename}' not found. "
                f"Ensure {filename} is in the repo root or current directory."
            )

        return None

    @classmethod
    def resolve_rpc_url(
        cls,
        chain_endpoint: str | None,
        subtensor_network: str | None,
    ) -> str | None:
        """Resolve EVM RPC URL from CLI flags.

        Priority: explicit HTTP ``--subtensor-chain-endpoint`` >
        auto-translated ``ws[s]://`` of the same form >
        ``--subtensor-network`` network default >
        None (use JSON default).

        Modern Bittensor subtensor exposes BOTH Substrate (WS) and EVM
        (HTTP) on the same host:port.  When the operator passes
        ``ws://host:port`` we translate to ``http://host:port`` for the
        EVM client — saves operators from having to specify two endpoints
        for what is one process.  The ``--evm-rpc-url`` flag (where the
        caller exposes one) still wins over this auto-translation.
        """
        if chain_endpoint and chain_endpoint.startswith(("http://", "https://")):
            return chain_endpoint
        if chain_endpoint and chain_endpoint.startswith(("ws://", "wss://")):
            # ws:// → http://, wss:// → https://  (same host:port — Bittensor
            # subtensor exposes both protocols on one port).
            return ("http://" + chain_endpoint[len("ws://"):]) \
                if chain_endpoint.startswith("ws://") \
                else ("https://" + chain_endpoint[len("wss://"):])
        if subtensor_network:
            return cls._NETWORK_RPC_URLS.get(subtensor_network)
        return None

    @classmethod
    def is_public_rpc_endpoint(cls, endpoint: str | None) -> bool:
        """Return whether an explicitly configured endpoint is public RPC."""
        if not endpoint:
            return False
        from urllib.parse import urlparse

        host = (urlparse(endpoint).hostname or "").lower()
        return host in cls._PUBLIC_RPC_HOSTS

    @classmethod
    def should_warn_public_rpc(cls, endpoint: str | None) -> bool:
        """Return whether the miner will use the public network RPC path."""
        return not endpoint or cls.is_public_rpc_endpoint(endpoint)

    @classmethod
    @classmethod
    def from_json(cls, path: str, **overrides) -> ChainConfig:
        """Load config from a JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        data.update(overrides)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


MAINNET_CHAIN_ID = 964


def validate_registration_endpoint_scheme(endpoint: str, chain_id: object) -> None:
    """Refuse plain-http registration endpoints on mainnet.

    The public mainnet proxy only routes https endpoints, so an http
    registration produces a miner the validators score but no user can
    ever reach — an invisible-to-traffic state the operator cannot
    diagnose from their side. Refusing at registration keeps that state
    unrepresentable. Test networks keep accepting http: their fleets run
    inside trusted rigs and endpoint TLS adds nothing to the proofs.
    """
    try:
        is_mainnet = int(chain_id) == MAINNET_CHAIN_ID
    except (TypeError, ValueError):
        is_mainnet = False
    if is_mainnet and not str(endpoint or "").strip().lower().startswith("https://"):
        raise ValueError(
            "mainnet registration requires an https:// endpoint "
            f"(got {endpoint!r}): front the miner with TLS "
            "(scripts/setup_https.sh) and register the https URL"
        )
