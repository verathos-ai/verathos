// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/ModelRegistry.sol";
import "../src/MinerRegistry.sol";
import "../src/PaymentGateway.sol";
import "../src/ValidatorRegistry.sol";
import "../src/UsageCheckpointRegistry.sol";

/// @title Deploy — deploys all Verathos contracts as UUPS proxies
/// @notice IMPORTANT: Must be deployed from the subnet owner wallet.
///         The deployer becomes `owner` on all proxies via initialize().
///
/// NOTE: forge script does NOT work on Bittensor EVM (prevrandao bug).
///       Use scripts/deploy_proxies.py (web3.py) for real deployments.
///       This script is for local Anvil testing only.
contract Deploy is Script {

    function _deployProxy(address impl, bytes memory initData) internal returns (address) {
        ERC1967Proxy proxy = new ERC1967Proxy(impl, initData);
        return address(proxy);
    }

    function run() external {
        uint16 netuid = uint16(vm.envUint("NETUID"));
        address metaAddr = vm.envOr("META_ADDR", address(0));
        address uidLookupAddr = vm.envOr("UID_LOOKUP_ADDR", address(0));
        address stakingAddr = vm.envOr("STAKING_ADDR", address(0));
        address ownerTreasury = vm.envOr("OWNER_TREASURY", msg.sender);
        uint16 ownerCutBps = uint16(vm.envOr("OWNER_CUT_BPS", uint256(0)));

        vm.prevrandao(bytes32(uint256(1)));
        vm.startBroadcast();

        address modelProxy = _deployProxy(
            address(new ModelRegistry()),
            abi.encodeCall(ModelRegistry.initialize, ())
        );
        address minerProxy = _deployProxy(
            address(new MinerRegistry()),
            abi.encodeCall(MinerRegistry.initialize, (netuid, metaAddr, uidLookupAddr))
        );
        address paymentProxy = _deployProxy(
            address(new PaymentGateway()),
            abi.encodeCall(PaymentGateway.initialize, (netuid, ownerTreasury, ownerCutBps, stakingAddr))
        );
        address validatorProxy = _deployProxy(
            address(new ValidatorRegistry()),
            abi.encodeCall(ValidatorRegistry.initialize, (netuid, metaAddr, uidLookupAddr, stakingAddr))
        );
        address checkpointProxy = _deployProxy(
            address(new UsageCheckpointRegistry()),
            abi.encodeCall(UsageCheckpointRegistry.initialize, (netuid, metaAddr, uidLookupAddr))
        );

        vm.stopBroadcast();

        console.log("=== Proxies (use these in chain config) ===");
        console.log("ModelRegistry:", modelProxy);
        console.log("MinerRegistry:", minerProxy);
        console.log("PaymentGateway:", paymentProxy);
        console.log("ValidatorRegistry:", validatorProxy);
        console.log("UsageCheckpointRegistry:", checkpointProxy);
        console.log("  netuid:", netuid);
        console.log("  owner:", msg.sender);
        console.log("  treasury:", ownerTreasury);
        console.log("  ownerCutBps:", ownerCutBps);
    }
}
