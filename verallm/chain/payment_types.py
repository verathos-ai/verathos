"""Data types for the PaymentGateway contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DepositEvent:
    """A deposit event from the PaymentGateway contract."""

    user: str  # EVM address (checksummed)
    validator_hotkey: bytes  # 32-byte hotkey
    tao_amount: int  # wei
    block_number: int
    tx_hash: str
