// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IAddressMapping — Bittensor AddressMappingPrecompile interface
/// @notice Precompile at 0x000000000000000000000000000000000000080C.
///         Converts an Ethereum H160 address to a Substrate AccountId32
///         using the runtime's HashedAddressMapping (Blake2b).
///
///         The mapping is: blake2b_256("evm:" || h160_bytes) -> AccountId32
///         This is deterministic and one-way.
interface IAddressMapping {

    /// @notice Convert an EVM address to its Substrate AccountId32.
    /// @param target_address  The EVM H160 address.
    /// @return The Substrate AccountId32 (SS58-encodable bytes32).
    function addressMapping(address target_address) external view returns (bytes32);
}
