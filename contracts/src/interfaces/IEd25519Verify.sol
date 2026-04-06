// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IEd25519Verify — Bittensor Ed25519 signature verification precompile
/// @notice Verifies Substrate Ed25519 signatures from within EVM contracts.
///         Deployed at 0x0000000000000000000000000000000000000402 on Bittensor.
interface IEd25519Verify {
    /// @param message   keccak256 hash of the signed data
    /// @param publicKey Ed25519 public key (hotkey)
    /// @param sigR      Signature R component
    /// @param sigS      Signature S component
    function verify(
        bytes32 message,
        bytes32 publicKey,
        bytes32 sigR,
        bytes32 sigS
    ) external view returns (bool);
}
