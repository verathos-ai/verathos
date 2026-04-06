// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {OwnableUpgradeable} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title SubnetConfig — on-chain subnet-level policy for Verathos
/// @notice Stores cross-cutting configuration that applies globally to the subnet,
///         not to any individual miner or validator. Managed by the subnet owner.
///
///         Includes: TEE measurement allowlist, miner blacklist, scoring parameters,
///         feature flags, and a generic key-value store for future extensibility.
contract SubnetConfig is Initializable, UUPSUpgradeable, OwnableUpgradeable {

    // ── State variables (storage layout) ────────────────────────

    uint16 public netuid;

    // ── TEE measurement allowlist ───────────────────────────────

    /// @dev Maps keccak256(raw_measurement) to whether it is accepted.
    ///      Raw measurement = mr_td (TDX) or launch_digest (SEV-SNP).
    mapping(bytes32 => bool) public acceptedMeasurements;

    // ── Miner blacklist ─────────────────────────────────────────

    /// @dev Maps miner EVM address to blacklisted status.
    mapping(address => bool) public blacklistedMiners;

    // ── Scoring parameters (all in basis points where applicable) ──

    /// @dev TEE bonus in basis points (1000 = 10% bonus = 1.10x multiplier).
    uint16 public teeBonusBps;

    /// @dev EMA smoothing alpha in basis points (2000 = 0.20).
    uint16 public emaAlphaBps;

    /// @dev Throughput sybil-defense exponent in basis points (20000 = 2.0).
    uint16 public throughputPowerBps;

    /// @dev ZK proof sample rate in basis points (3000 = 30% of canaries request proof).
    uint16 public proofSampleRateBps;

    /// @dev Consecutive clean epochs required to exit probation.
    uint16 public probationRequiredPasses;

    /// @dev Maximum demand bonus in basis points (2000 = 20% max uplift).
    uint16 public demandBonusMaxBps;

    /// @dev Fraction of miner emissions redirected to subnet-owner UID (burn).
    ///      5000 = 50% burned, 0 = no burn, 10000 = 100% burned.
    uint16 public emissionBurnBps;

    // ── Feature flags ───────────────────────────────────────────

    /// @dev Generic boolean flags keyed by bytes32 identifier.
    mapping(bytes32 => bool) public featureFlags;

    // ── Generic key-value store ─────────────────────────────────

    /// @dev Generic bytes32 → bytes32 config values for future extensibility
    ///      without requiring contract upgrades.
    mapping(bytes32 => bytes32) public configValues;

    // ── Storage gap for future upgrades ─────────────────────────

    uint256[50] private __gap;

    // ── Events ──────────────────────────────────────────────────

    event MeasurementUpdated(bytes32 indexed hash, bool accepted);
    event MinerBlacklisted(address indexed miner, bool blacklisted);
    event ScoringParamUpdated(string indexed param, uint16 oldValue, uint16 newValue);
    event FeatureFlagUpdated(bytes32 indexed key, bool value);
    event ConfigValueUpdated(bytes32 indexed key, bytes32 value);

    // ── Scoring params struct (for single-call batch read) ──────

    struct ScoringParams {
        uint16 teeBonusBps;
        uint16 emaAlphaBps;
        uint16 throughputPowerBps;
        uint16 proofSampleRateBps;
        uint16 probationRequiredPasses;
        uint16 demandBonusMaxBps;
        uint16 emissionBurnBps;
    }

    // ── Constructor / Initializer ───────────────────────────────

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(uint16 _netuid) external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
        netuid = _netuid;
        teeBonusBps = 1000;              // 10% TEE bonus
        emaAlphaBps = 2000;              // 0.20 EMA alpha
        throughputPowerBps = 20000;      // 2.0 throughput exponent
        proofSampleRateBps = 3000;       // 30% proof sample rate
        probationRequiredPasses = 3;     // 3 consecutive passes
        demandBonusMaxBps = 2000;        // 20% max demand bonus
        emissionBurnBps = 5000;          // 50% emission burn
    }

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── Batch read (single RPC call) ────────────────────────────

    /// @notice Get all scoring parameters in a single call.
    function getScoringParams() external view returns (ScoringParams memory) {
        return ScoringParams({
            teeBonusBps: teeBonusBps,
            emaAlphaBps: emaAlphaBps,
            throughputPowerBps: throughputPowerBps,
            proofSampleRateBps: proofSampleRateBps,
            probationRequiredPasses: probationRequiredPasses,
            demandBonusMaxBps: demandBonusMaxBps,
            emissionBurnBps: emissionBurnBps
        });
    }

    // ── TEE measurement allowlist ───────────────────────────────

    /// @notice Add or remove a single measurement hash from the allowlist.
    function setAcceptedMeasurement(bytes32 hash, bool accepted) external onlyOwner {
        acceptedMeasurements[hash] = accepted;
        emit MeasurementUpdated(hash, accepted);
    }

    /// @notice Batch add/remove measurement hashes.
    function setAcceptedMeasurements(
        bytes32[] calldata hashes,
        bool[] calldata values
    ) external onlyOwner {
        require(hashes.length == values.length, "Array length mismatch");
        for (uint256 i = 0; i < hashes.length; i++) {
            acceptedMeasurements[hashes[i]] = values[i];
            emit MeasurementUpdated(hashes[i], values[i]);
        }
    }

    /// @notice Check if a measurement hash is accepted.
    function isAcceptedMeasurement(bytes32 hash) external view returns (bool) {
        return acceptedMeasurements[hash];
    }

    // ── Miner blacklist ─────────────────────────────────────────

    /// @notice Blacklist or un-blacklist a miner by EVM address.
    function setMinerBlacklisted(address miner, bool blacklisted) external onlyOwner {
        blacklistedMiners[miner] = blacklisted;
        emit MinerBlacklisted(miner, blacklisted);
    }

    /// @notice Check if a miner is blacklisted.
    function isMinerBlacklisted(address miner) external view returns (bool) {
        return blacklistedMiners[miner];
    }

    // ── Scoring parameter setters (onlyOwner) ───────────────────

    /// @notice Update the TEE bonus in basis points (max 10000 = 100%).
    function setTeeBonusBps(uint16 bps) external onlyOwner {
        require(bps <= 10000, "Exceeds 100%");
        emit ScoringParamUpdated("teeBonusBps", teeBonusBps, bps);
        teeBonusBps = bps;
    }

    /// @notice Update the EMA alpha in basis points (max 10000 = 1.0).
    function setEmaAlphaBps(uint16 bps) external onlyOwner {
        require(bps <= 10000, "Exceeds 1.0");
        emit ScoringParamUpdated("emaAlphaBps", emaAlphaBps, bps);
        emaAlphaBps = bps;
    }

    /// @notice Update the throughput power in basis points (e.g. 20000 = 2.0).
    function setThroughputPowerBps(uint16 bps) external onlyOwner {
        require(bps <= 50000, "Exceeds 5.0");
        emit ScoringParamUpdated("throughputPowerBps", throughputPowerBps, bps);
        throughputPowerBps = bps;
    }

    /// @notice Update the proof sample rate in basis points (max 10000 = 100%).
    function setProofSampleRateBps(uint16 bps) external onlyOwner {
        require(bps <= 10000, "Exceeds 100%");
        emit ScoringParamUpdated("proofSampleRateBps", proofSampleRateBps, bps);
        proofSampleRateBps = bps;
    }

    /// @notice Update the probation required passes (1-100).
    function setProbationRequiredPasses(uint16 passes) external onlyOwner {
        require(passes >= 1 && passes <= 100, "Out of range 1-100");
        emit ScoringParamUpdated("probationRequiredPasses", probationRequiredPasses, passes);
        probationRequiredPasses = passes;
    }

    /// @notice Update the max demand bonus in basis points (max 10000 = 100%).
    function setDemandBonusMaxBps(uint16 bps) external onlyOwner {
        require(bps <= 10000, "Exceeds 100%");
        emit ScoringParamUpdated("demandBonusMaxBps", demandBonusMaxBps, bps);
        demandBonusMaxBps = bps;
    }

    /// @notice Update the emission burn rate in basis points (max 10000 = 100%).
    function setEmissionBurnBps(uint16 bps) external onlyOwner {
        require(bps <= 10000, "Exceeds 100%");
        emit ScoringParamUpdated("emissionBurnBps", emissionBurnBps, bps);
        emissionBurnBps = bps;
    }

    // ── Feature flags ───────────────────────────────────────────

    /// @notice Set a feature flag.
    function setFeatureFlag(bytes32 key, bool value) external onlyOwner {
        featureFlags[key] = value;
        emit FeatureFlagUpdated(key, value);
    }

    // ── Generic key-value store ─────────────────────────────────

    /// @notice Set a generic config value.
    function setConfigValue(bytes32 key, bytes32 value) external onlyOwner {
        configValues[key] = value;
        emit ConfigValueUpdated(key, value);
    }
}
