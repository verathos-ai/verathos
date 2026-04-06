// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IMetagraph — Bittensor MetagraphPrecompile interface
/// @notice Live, read-only access to Substrate metagraph state from EVM.
///         Deployed at 0x0000000000000000000000000000000000000802 on Bittensor.
interface IMetagraph {
    function getUidCount(uint16 netuid) external view returns (uint16);

    function getHotkey(uint16 netuid, uint16 uid) external view returns (bytes32);
    function getColdkey(uint16 netuid, uint16 uid) external view returns (bytes32);

    function getIsActive(uint16 netuid, uint16 uid) external view returns (bool);
    function getValidatorStatus(uint16 netuid, uint16 uid) external view returns (bool);
    function getLastUpdate(uint16 netuid, uint16 uid) external view returns (uint64);

    function getStake(uint16 netuid, uint16 uid) external view returns (uint64);
    function getEmission(uint16 netuid, uint16 uid) external view returns (uint64);
    function getRank(uint16 netuid, uint16 uid) external view returns (uint16);
    function getTrust(uint16 netuid, uint16 uid) external view returns (uint16);
    function getConsensus(uint16 netuid, uint16 uid) external view returns (uint16);
    function getIncentive(uint16 netuid, uint16 uid) external view returns (uint16);
    function getDividends(uint16 netuid, uint16 uid) external view returns (uint16);
    function getVtrust(uint16 netuid, uint16 uid) external view returns (uint16);

    function getAxon(uint16 netuid, uint16 uid) external view returns (
        uint64 blockNum, uint32 version, uint128 ip, uint16 port,
        uint8 ipType, uint8 protocol
    );
}
