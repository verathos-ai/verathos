"""Python client for the UsageCheckpointRegistry Solidity contract.

Records Merkle roots of user balances on-chain for disaster recovery.
One event per checkpoint — constant gas cost regardless of user count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from verallm.chain.config import ChainConfig
from verallm.chain.provider import Web3Provider

logger = logging.getLogger(__name__)


@dataclass
class UsageCheckpointEvent:
    """A Merkle root checkpoint event from the UsageCheckpointRegistry contract."""

    validator: str  # validator EVM address (checksummed)
    merkle_root: bytes  # 32-byte Merkle root of user balances
    user_count: int
    total_consumed_wei: int
    total_consumed_usd_micros: int
    timestamp: int  # block.timestamp
    block_number: int
    tx_hash: str


class UsageCheckpointClient:
    """Read/write interface to the on-chain UsageCheckpointRegistry.

    Used by the validator proxy to:
    - Record a Merkle root of user balances periodically (write)
    - Verify backup integrity against on-chain root (read)
    """

    def __init__(
        self,
        config: ChainConfig,
        provider: Optional[Web3Provider] = None,
        private_key: Optional[str] = None,
    ):
        self._config = config
        if not config.checkpoint_registry_address:
            raise ValueError(
                "checkpoint_registry_address not set in chain config. "
                "Deploy UsageCheckpointRegistry and update chain_config."
            )
        self._provider = provider or Web3Provider(config)
        self._contract = self._provider.get_contract(
            config.checkpoint_registry_address, "UsageCheckpointRegistry"
        )
        self._private_key = private_key or config.evm_private_key

    # ── EVM registration ───────────────────────────────────────────

    def is_evm_registered(self, evm_address: str) -> bool:
        """Check if an EVM address has called registerEvm() on this contract."""
        from web3 import Web3
        addr = Web3.to_checksum_address(evm_address)
        return self._contract.functions.evmRegistered(addr).call()

    def register_evm(
        self,
        uid: int,
        hotkey_seed: bytes,
        netuid: int,
        private_key: str | None = None,
    ) -> str:
        """Register EVM address → UID mapping with SR25519 hotkey proof."""
        from verallm.chain.wallet import sign_evm_registration
        from eth_account import Account

        pk = private_key or self._private_key
        if not pk:
            raise ValueError("No private key for registerEvm()")

        contract_addr = self._contract.address
        evm_address = Account.from_key(pk).address

        sig_r, sig_s = sign_evm_registration(
            hotkey_seed, evm_address, uid, netuid, contract_addr,
        )
        tx_hash = self._provider.send_transaction(
            self._contract.functions.registerEvm(uid, sig_r, sig_s),
            private_key=pk,
        )
        logger.info("registerEvm(%d) on UsageCheckpointRegistry (SR25519 verified): %s", uid, tx_hash)
        return tx_hash

    # ── Write ─────────────────────────────────────────────────────

    def record_checkpoint(
        self,
        users: list[str],
        consumed_tao: list[int],
        consumed_usd: list[int],
    ) -> str:
        """Build a Merkle tree of user balances and record the root on-chain.

        Args:
            users: User EVM addresses.
            consumed_tao: Total consumed wei per user (cumulative).
            consumed_usd: Total consumed microdollars per user (cumulative).

        Returns:
            Transaction hash hex string.
        """
        from web3 import Web3

        assert len(users) == len(consumed_tao) == len(consumed_usd), "Length mismatch"

        if not self._private_key:
            raise ValueError("No private key configured for signing transactions")

        # Build Merkle tree: leaf = keccak256(abi.encodePacked(user, taoWei, usdMicros))
        leaves = []
        total_tao = 0
        total_usd = 0
        for user, tao, usd in zip(users, consumed_tao, consumed_usd):
            addr = Web3.to_checksum_address(user)
            leaf = Web3.keccak(
                Web3.to_bytes(hexstr=addr) +
                tao.to_bytes(32, "big") +
                usd.to_bytes(32, "big")
            )
            leaves.append(leaf)
            total_tao += tao
            total_usd += usd

        root = _compute_merkle_root(leaves) if leaves else b"\x00" * 32

        tx_hash = self._provider.send_transaction(
            self._contract.functions.recordCheckpoint(
                root, len(users), total_tao, total_usd,
            ),
            private_key=self._private_key,
        )

        logger.info(
            "Recorded checkpoint: %d users, root=%s, tao=%d, usd=%d: %s",
            len(users), root.hex()[:16], total_tao, total_usd, tx_hash,
        )
        return tx_hash

    # ── Read (for disaster recovery) ──────────────────────────────

    def get_checkpoint_events(
        self,
        from_block: int = 0,
        to_block: str | int = "latest",
        validator: Optional[str] = None,
    ) -> List[UsageCheckpointEvent]:
        """Query UsageCheckpoint events from the contract."""
        from web3 import Web3

        kwargs: dict = {
            "from_block": from_block,
            "to_block": to_block,
        }
        if validator:
            kwargs["argument_filters"] = {
                "validator": Web3.to_checksum_address(validator)
            }

        logs = self._provider.call_with_retry(
            lambda: self._contract.events.UsageCheckpoint.get_logs(**kwargs)
        )

        events = []
        for log in logs:
            events.append(UsageCheckpointEvent(
                validator=log.args["validator"],
                merkle_root=bytes(log.args["merkleRoot"]),
                user_count=log.args["userCount"],
                total_consumed_wei=log.args["totalConsumedWei"],
                total_consumed_usd_micros=log.args["totalConsumedUsdMicros"],
                timestamp=log.args["timestamp"],
                block_number=log.blockNumber,
                tx_hash=log.transactionHash.hex(),
            ))

        events.sort(key=lambda e: e.block_number)
        return events

    def get_latest_checkpoint(
        self,
        from_block: int = 0,
    ) -> Optional[UsageCheckpointEvent]:
        """Get the most recent checkpoint event."""
        events = self.get_checkpoint_events(from_block=from_block)
        return events[-1] if events else None


def _compute_merkle_root(leaves: list[bytes]) -> bytes:
    """Compute a simple Merkle root from a list of 32-byte leaves.

    Uses keccak256 for internal nodes. Odd layers are padded by
    duplicating the last leaf (standard binary Merkle tree).
    """
    from web3 import Web3

    if not leaves:
        return b"\x00" * 32
    if len(leaves) == 1:
        return leaves[0]

    # Sort leaves for deterministic ordering
    layer = sorted(leaves)

    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            # Sort pair for canonical ordering (prevents order-dependent roots)
            if left > right:
                left, right = right, left
            next_layer.append(Web3.keccak(left + right))
        layer = next_layer

    return layer[0]
