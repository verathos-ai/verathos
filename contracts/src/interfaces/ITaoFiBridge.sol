// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ITaoFiBridge — interface for TaoFi SwapBridgeAndCallFromMain on Base
/// @notice Bridges USDC from Base to Bittensor EVM via Hyperlane Warp Routes,
///         then executes remote calls (swap + stake) via Interchain Accounts.
///
///         Struct layouts match TaoFi's deployed contract:
///         https://github.com/TaoFi-0x/taofi-swap/blob/main/contracts/SwapBridgeAndCallFromMain.sol
interface ITaoFiBridge {

    /// @notice Parameters for an optional LiFi swap before bridging.
    struct SwapParams {
        address fromToken;        // Token to bridge (e.g. USDC on Base)
        uint256 fromAmount;       // Amount to bridge
        address approvalAddress;  // LiFi approval target (0 if no swap)
        address target;           // LiFi swap contract (0 if no swap)
        bytes data;               // LiFi calldata (empty if no swap)
    }

    /// @notice A single remote call to execute on Bittensor EVM via Hyperlane ICA.
    ///         Matches Hyperlane's IInterchainAccountRouter.Call struct.
    struct Call {
        bytes32 to;     // Target contract (bytes32 for non-EVM compatibility)
        uint256 value;  // Native currency amount (0 for token calls)
        bytes data;     // Encoded function call
    }

    /// @notice Parameters for cross-chain remote calls via Hyperlane ICA.
    struct RemoteCallsParams {
        bytes32 router;        // ICA router address on destination chain
        bytes32 ism;           // Interchain Security Module override
        Call[] calls;          // Remote calls to execute on Bittensor EVM
        bytes hookMetadata;    // Hyperlane hook metadata
    }

    /// @notice Bridge tokens from Base to Bittensor EVM and execute remote calls.
    /// @param swapParams   LiFi swap parameters (pass-through if no swap needed).
    /// @param params       Remote calls to execute on Bittensor EVM via ICA.
    /// @param bridgeCost   ETH amount for Hyperlane messaging fees.
    function lifiSwapBridgeAndCall(
        SwapParams calldata swapParams,
        RemoteCallsParams calldata params,
        uint256 bridgeCost
    ) external payable;
}
