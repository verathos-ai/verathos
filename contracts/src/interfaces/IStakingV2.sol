// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IStakingV2 — Bittensor StakingV2 precompile interface
/// @notice Precompile at 0x0000000000000000000000000000000000000805.
///         Allows EVM contracts to stake/unstake TAO as alpha on subnets.
///         The contract address acts as the coldkey for staking operations.
interface IStakingV2 {
    /// @notice Stake TAO as alpha on a subnet for a specific hotkey.
    ///         V2 deducts from the caller's substrate balance directly
    ///         (Bittensor unifies EVM and substrate balances via
    ///         HashedAddressMapping). Do NOT forward msg.value — that would
    ///         reduce the balance twice.
    /// @param hotkey The validator/miner hotkey Sr25519 pubkey (bytes32).
    /// @param amount The amount to stake in RAO (1e9 per TAO).
    /// @param netuid The subnet to stake on.
    function addStake(bytes32 hotkey, uint256 amount, uint256 netuid) external payable;

    /// @notice Remove staked alpha from a subnet.
    /// @param hotkey The hotkey to unstake from.
    /// @param amount The amount to unstake (in alpha units, RAO-scale).
    /// @param netuid The subnet to unstake from.
    function removeStake(bytes32 hotkey, uint256 amount, uint256 netuid) external payable;

    /// @notice Query staked amount for a (hotkey, coldkey, netuid) triple.
    /// @param hotkey The hotkey.
    /// @param coldkey The coldkey (AccountId32 as bytes32).
    /// @param netuid The subnet.
    /// @return The staked amount.
    function getStake(bytes32 hotkey, bytes32 coldkey, uint256 netuid) external view returns (uint256);

    /// @notice Get total alpha staked to a hotkey on a specific subnet (all coldkeys).
    /// @param hotkey The hotkey.
    /// @param netuid The subnet.
    /// @return Total alpha staked in RAO.
    function getTotalAlphaStaked(bytes32 hotkey, uint256 netuid) external view returns (uint256);

    /// @notice Get total stake for a hotkey across all subnets.
    /// @param hotkey The hotkey.
    /// @return Total staked amount in RAO.
    function getTotalHotkeyStake(bytes32 hotkey) external view returns (uint256);
}
