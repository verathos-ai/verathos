"""EVM address derivation and mirror address helpers for Bittensor hotkeys."""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


def sign_evm_registration(
    hotkey_seed: bytes,
    evm_address: str,
    uid: int,
    netuid: int,
    contract_address: str,
) -> tuple[bytes, bytes]:
    """Sign an EVM registration message with the SR25519 hotkey.

    The contract verifies this signature via the Sr25519Verify precompile
    (0x403) to prove the caller owns the hotkey for the claimed UID.

    The message is: ``keccak256(abi.encodePacked(evm_address, uid, netuid, contract_address))``
    matching the Solidity: ``keccak256(abi.encodePacked(msg.sender, uid, netuid, address(this)))``

    Args:
        hotkey_seed: 32-byte hotkey seed (Mini secret key).
        evm_address: Caller's EVM address (checksummed hex with 0x prefix).
        uid: UID to claim on the subnet.
        netuid: Subnet UID.
        contract_address: Address of the contract (MinerRegistry, etc.).

    Returns:
        ``(sig_r, sig_s)`` — each 32 bytes, ready for the contract call.
    """
    from substrateinterface import Keypair
    from web3 import Web3

    # Replicate Solidity: keccak256(abi.encodePacked(msg.sender, uid, netuid, address(this)))
    # abi.encodePacked types: address(20) + uint16(2) + uint16(2) + address(20) = 44 bytes
    evm_bytes = bytes.fromhex(evm_address[2:])  # 20 bytes
    uid_bytes = uid.to_bytes(2, "big")           # uint16
    netuid_bytes = netuid.to_bytes(2, "big")     # uint16
    contract_bytes = bytes.fromhex(contract_address[2:])  # 20 bytes

    packed = evm_bytes + uid_bytes + netuid_bytes + contract_bytes
    message = Web3.keccak(packed)  # bytes32

    # Sign with SR25519
    keypair = Keypair.create_from_seed(hotkey_seed[:32].hex())
    signature = keypair.sign(message)

    # signature is 64 bytes: first 32 = R, last 32 = S
    sig_bytes = signature if isinstance(signature, bytes) else bytes.fromhex(
        signature.replace("0x", "")
    )
    assert len(sig_bytes) == 64, f"Expected 64-byte SR25519 signature, got {len(sig_bytes)}"

    return sig_bytes[:32], sig_bytes[32:]


def derive_evm_private_key(hotkey_seed: bytes) -> str:
    """Derive an EVM-compatible private key from a Bittensor hotkey seed.

    Bittensor derives the EVM private key by taking the keccak256 hash
    of the Ed25519 secret key bytes. This matches the convention used by
    the Bittensor SDK's ``wallet.get_evm_key()`` method.

    Args:
        hotkey_seed: The 32-byte Ed25519 private key seed.

    Returns:
        Hex-encoded EVM private key (64 chars, no 0x prefix).
    """
    from web3 import Web3
    # keccak256 of the seed to derive EVM private key
    evm_pk = Web3.keccak(hotkey_seed)
    return evm_pk.hex()


def derive_evm_address(hotkey_seed: bytes) -> str:
    """Derive the EVM address from a Bittensor hotkey seed.

    Returns:
        Checksummed EVM address string.
    """
    from eth_account import Account
    pk = derive_evm_private_key(hotkey_seed)
    account = Account.from_key(pk)
    return account.address


def h160_to_ss58_mirror(h160_address: str, ss58_format: int = 42) -> str:
    """Compute the SS58 mirror address for an EVM H160 address.

    Bittensor uses Substrate Frontier's ``HashedAddressMapping<BlakeTwo256>``
    to map EVM addresses to Substrate accounts. The mirror is:

        AccountId32 = blake2b_256(b"evm:" + h160_bytes)

    Transferring TAO to this SS58 mirror address makes it available as
    the native balance of the corresponding EVM H160 wallet.

    Args:
        h160_address: EVM address (hex string, with or without 0x prefix).
        ss58_format: SS58 network prefix (42 = Bittensor).

    Returns:
        SS58-encoded mirror address string.
    """
    addr = h160_address.lower().removeprefix("0x")
    if len(addr) != 40:
        raise ValueError(f"Invalid H160 address length: {len(addr)}")
    h160_bytes = bytes.fromhex(addr)

    # HashedAddressMapping: blake2b_256("evm:" || h160)
    account_id = hashlib.blake2b(
        b"evm:" + h160_bytes, digest_size=32
    ).digest()

    return _ss58_encode(account_id, ss58_format)


def _ss58_encode(public_key: bytes, ss58_format: int = 42) -> str:
    """Minimal SS58 encoding (no external dependency required).

    Uses the SS58 checksum algorithm: blake2b_512(b"SS58PRE:" + prefix + key)[:2].
    """
    if ss58_format < 64:
        prefix = bytes([ss58_format])
    else:
        # Two-byte prefix for formats >= 64
        first = ((ss58_format & 0b0000000011111100) >> 2) | 0b01000000
        second = (ss58_format >> 8) | ((ss58_format & 0b0000000000000011) << 6)
        prefix = bytes([first, second])

    payload = prefix + public_key
    checksum = hashlib.blake2b(b"SS58PRE" + payload, digest_size=64).digest()[:2]
    return _base58_encode(payload + checksum)


_BASE58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_encode(data: bytes) -> str:
    """Bitcoin-style base58 encoding."""
    n = int.from_bytes(data, "big")
    result = bytearray()
    while n > 0:
        n, r = divmod(n, 58)
        result.append(_BASE58_ALPHABET[r])
    # Preserve leading zeros
    for byte in data:
        if byte == 0:
            result.append(_BASE58_ALPHABET[0])
        else:
            break
    return bytes(reversed(result)).decode("ascii")


def ss58_decode(address: str) -> bytes:
    """Decode an SS58 address to its 32-byte AccountId.

    Raises:
        ValueError: If the address is invalid.
    """
    decoded = _base58_decode(address)
    if len(decoded) < 35:
        raise ValueError(f"SS58 address too short: {len(decoded)} bytes")
    # Determine prefix length
    first = decoded[0]
    if first < 64:
        prefix_len = 1
    elif first < 128:
        prefix_len = 2
    else:
        raise ValueError(f"Invalid SS58 prefix byte: {first}")
    account_id = decoded[prefix_len:prefix_len + 32]
    if len(account_id) != 32:
        raise ValueError("Could not extract 32-byte AccountId from SS58 address")
    return account_id


def _base58_decode(s: str) -> bytes:
    """Bitcoin-style base58 decoding."""
    n = 0
    for char in s:
        n *= 58
        idx = _BASE58_ALPHABET.find(char.encode())
        if idx < 0:
            raise ValueError(f"Invalid base58 character: {char!r}")
        n += idx
    # Count leading '1's → leading zero bytes
    leading = sum(1 for c in s if c == "1")
    result = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\x00" * leading + result


def derive_deposit_private_key(
    validator_seed: bytes,
    index: int,
    domain: bytes = b"tao_deposit",
) -> str:
    """Derive a per-user deposit address private key via HD derivation.

    Each user gets a unique deposit address derived deterministically from
    the validator's hotkey seed, a user index, and a domain separator.

    Args:
        validator_seed: The 32-byte validator hotkey seed.
        index: Per-user address index (auto-incremented).
        domain: Domain separator — ``b"tao_deposit"`` for Bittensor chain,
            ``b"base_deposit"`` for Base L2.

    Returns:
        Hex-encoded EVM private key (64 chars, no 0x prefix).
    """
    from web3 import Web3

    data = validator_seed + index.to_bytes(4, "big") + domain
    return Web3.keccak(data).hex()


def derive_deposit_address(
    validator_seed: bytes,
    index: int,
    domain: bytes = b"tao_deposit",
) -> tuple[str, str]:
    """Derive EVM address and SS58 mirror for a per-user deposit address.

    Args:
        validator_seed: The 32-byte validator hotkey seed.
        index: Per-user address index.
        domain: Domain separator (``b"tao_deposit"`` or ``b"base_deposit"``).

    Returns:
        ``(evm_address, ss58_mirror)`` tuple. The SS58 mirror is only
        meaningful for ``tao_deposit`` domain (Bittensor chain).
    """
    from eth_account import Account

    pk = derive_deposit_private_key(validator_seed, index, domain)
    account = Account.from_key(pk)
    evm_addr = account.address
    ss58 = h160_to_ss58_mirror(evm_addr)
    return evm_addr, ss58


def get_evm_credentials_from_bittensor_wallet(
    wallet_name: str = "default",
    hotkey_name: str = "default",
) -> tuple[str, str]:
    """Load a Bittensor wallet and derive EVM credentials.

    Returns:
        (evm_private_key_hex, evm_address) tuple.

    Raises:
        ImportError: If bittensor is not installed.
    """
    try:
        import bittensor as bt
    except ImportError:
        raise ImportError(
            "bittensor is required for wallet operations. "
            "Install with: pip install bittensor"
        )

    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
    hotkey_seed = wallet.hotkey.private_key[:32]  # Ed25519 seed

    evm_pk = derive_evm_private_key(hotkey_seed)
    evm_addr = derive_evm_address(hotkey_seed)

    logger.info(
        "Derived EVM address %s from wallet %s/%s",
        evm_addr, wallet_name, hotkey_name,
    )
    return evm_pk, evm_addr
