// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/X402Gateway.sol";

/// @title DeployBase — deploys X402Gateway on Base L2
/// @notice Usage:
///   forge script script/DeployBase.s.sol \
///       --rpc-url $BASE_RPC_URL --broadcast --private-key $PK
///
///   Environment variables:
///     USDC_ADDRESS            — Base USDC (default: mainnet)
///     TAOFI_ADDRESS           — TaoFi SwapBridgeAndCallFromMain on Base
///     NETUID                  — target Bittensor subnet
///     TREASURY_ADDRESS        — owner treasury for USDC cut
///     OWNER_CUT_BPS           — basis points (default: 1000 = 10%)
///     VALIDATOR_HOTKEY        — bytes32 hotkey for alpha staking
///     MIN_BRIDGE_AMOUNT       — min USDC to bridge (default: 10e6 = 10 USDC)
///     REMOTE_SWAP_AND_STAKE   — TaoFi SwapAndStake on Bittensor EVM
///     REMOTE_WTAO             — WTAO on Bittensor EVM
///     REMOTE_USDC             — Bridged USDC on Bittensor EVM
contract DeployBase is Script {

    // Base mainnet USDC
    address constant DEFAULT_USDC = 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913;

    function run() external {
        address usdc = vm.envOr("USDC_ADDRESS", DEFAULT_USDC);
        address taofi = vm.envAddress("TAOFI_ADDRESS");
        uint16 netuid = uint16(vm.envUint("NETUID"));
        address treasury = vm.envAddress("TREASURY_ADDRESS");
        uint16 ownerCutBps = uint16(vm.envOr("OWNER_CUT_BPS", uint256(0)));
        bytes32 validatorHotkey = vm.envBytes32("VALIDATOR_HOTKEY");
        uint256 minBridgeAmount = vm.envOr("MIN_BRIDGE_AMOUNT", uint256(10_000_000));

        vm.startBroadcast();

        X402Gateway gateway = new X402Gateway(
            usdc,
            taofi,
            netuid,
            treasury,
            ownerCutBps,
            validatorHotkey,
            minBridgeAmount
        );

        // Set remote config if provided
        address remoteSwapAndStake = vm.envOr("REMOTE_SWAP_AND_STAKE", address(0));
        if (remoteSwapAndStake != address(0)) {
            gateway.setRemoteConfig(
                remoteSwapAndStake,
                vm.envAddress("REMOTE_WTAO"),
                vm.envAddress("REMOTE_USDC")
            );
        }

        vm.stopBroadcast();

        console.log("X402Gateway deployed at:", address(gateway));
        console.log("  USDC:", usdc);
        console.log("  TaoFi:", taofi);
        console.log("  netuid:", netuid);
        console.log("  treasury:", treasury);
        console.log("  ownerCutBps:", ownerCutBps);
        console.log("  validatorHotkey:");
        console.logBytes32(validatorHotkey);
        console.log("  minBridgeAmount:", minBridgeAmount);
    }
}
