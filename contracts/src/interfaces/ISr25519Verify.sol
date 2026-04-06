// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ISr25519Verify — Bittensor SR25519 signature verification precompile
/// @notice Precompile at 0x0000000000000000000000000000000000000403 (INDEX 1027).
///         Verifies SR25519 (Schnorr on Ristretto25519) signatures.
///         Bittensor hotkeys are SR25519 — this is the correct scheme.
///
///         Input (ABI-encoded, 132 bytes = 4-byte selector + 4×bytes32):
///           selector:  keccak256("verify(bytes32,bytes32,bytes32,bytes32)")[:4]
///           message:   bytes32 (typically a keccak256 hash)
///           publicKey: bytes32 (SR25519 public key, e.g. from META.getHotkey())
///           r:         bytes32 (first 32 bytes of the 64-byte SR25519 signature)
///           s:         bytes32 (last 32 bytes of the 64-byte SR25519 signature)
///
///         Output: bool (true if signature is valid)
///
///         Source: github.com/opentensor/subtensor/blob/main/precompiles/src/sr25519.rs
interface ISr25519Verify {
    function verify(
        bytes32 message,
        bytes32 publicKey,
        bytes32 r,
        bytes32 s
    ) external view returns (bool);
}
