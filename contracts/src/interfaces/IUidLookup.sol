// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IUidLookup — Bittensor UidLookupPrecompile interface
/// @notice Precompile at 0x0000000000000000000000000000000000000806.
///         Queries the Substrate `AssociatedEvmAddress` storage to find
///         which UIDs are associated with a given EVM address on a subnet.
///
///         Associations are created via the `associate_evm_key` Substrate
///         extrinsic (called by the hotkey, signed by the EVM key).
interface IUidLookup {

    struct LookupItem {
        uint16 uid;
        uint64 block_associated;
    }

    /// @notice Look up UIDs associated with an EVM address on a subnet.
    /// @param netuid  The subnet to query.
    /// @param evm_address  The EVM H160 address to look up.
    /// @param limit  Maximum number of results to return.
    /// @return Array of (uid, block_associated) pairs.
    function uidLookup(uint16 netuid, address evm_address, uint16 limit)
        external view returns (LookupItem[] memory);
}
