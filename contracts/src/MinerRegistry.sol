// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IMetagraph.sol";
import "./interfaces/IUidLookup.sol";
import {ISr25519Verify} from "./interfaces/ISr25519Verify.sol";
import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {OwnableUpgradeable} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title MinerRegistry — decentralized service directory for Verathos miners
/// @notice Miners register which models they serve and at which endpoints.
///         Access control uses Bittensor native precompiles: only neurons that
///         have called the `associate_evm_key` Substrate extrinsic can interact.
///         The contract queries live Substrate state via the UidLookup (0x806)
///         and Metagraph (0x802) precompiles — no custom hotkey storage needed.
///
///         Precompile addresses are configurable for testing (Anvil doesn't
///         have real precompiles). On mainnet/testnet, pass address(0) to use
///         the baked-in Bittensor addresses.
contract MinerRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable {

    // ── Precompile addresses ──────────────────────────────────────

    // Default Bittensor precompile addresses
    address constant DEFAULT_META = 0x0000000000000000000000000000000000000802;
    address constant DEFAULT_UID_LOOKUP = 0x0000000000000000000000000000000000000806;
    address constant SR25519_VERIFY = 0x0000000000000000000000000000000000000403;

    // ── State variables (storage layout) ────────────────────────

    IMetagraph public META;
    IUidLookup public UID_LOOKUP;
    uint16 public netuid;

    // ── Miner model data ──────────────────────────────────────────

    struct MinerModel {
        string   modelId;           // must match a ModelRegistry entry
        string   endpoint;          // URL (may be behind reverse proxy)
        bytes32  modelSpecRef;      // keccak256 of ModelSpec for cross-reference
        string   quant;             // quantization mode ("bf16", "fp8", "int4", ...)
        uint32   maxContextLen;     // claimed max context length (tokens)
        uint64   expiresAt;         // must renew before this (lease model)
        bool     active;            // can be deactivated by reports
    }

    mapping(address => MinerModel[]) public minerModels;
    mapping(string => address[]) internal _modelProviders;

    /// @dev failureCount[miner][modelIndex] — number of unique validator reports
    mapping(address => mapping(uint256 => uint16)) public failureCount;

    /// @dev Tracks which validators have reported a specific miner-model as offline
    mapping(address => mapping(uint256 => mapping(address => bool))) public reported;

    uint64 public constant LEASE_DURATION = 24 hours;
    uint16 public constant OFFLINE_THRESHOLD = 3;
    uint64 public constant CLEANUP_GRACE = 30 days;

    // ── Endpoint ownership (anti-hijacking) ─────────────────────

    /// @dev Maps keccak256(endpoint) to the owning address.
    ///      Prevents two different miners from registering the same endpoint URL.
    mapping(bytes32 => address) private _endpointOwner;

    // ── TEE (optional, per-miner) ─────────────────────────────

    struct TEECapability {
        bool     enabled;
        string   platform;          // "tdx", "sev-snp", "mock", "gpu"
        bytes32  enclavePubKey;     // X25519 public key (generated inside enclave)
        bytes32  attestationHash;   // keccak256(full attestation report)
        uint64   attestedAt;        // block.number when attested
        bytes32  modelWeightHash;   // SHA256(safetensors) — model identity for on-chain cross-check
        bytes32  codeMeasurement;   // keccak256(mr_td) for TDX, keccak256(launch_digest) for SEV-SNP
    }

    mapping(address => TEECapability) public teeCapability;

    // ── Model demand scores (validator-posted, per-model) ───────

    mapping(string => uint16) public modelDemandScore;  // 0-10000 bps
    bytes32 public demandReceiptHash;
    uint64  public demandEpoch;
    address public demandUpdater;

    // ── EVM ↔ UID registration ──────────────────────────────────

    mapping(address => uint16) public evmToUid;
    mapping(address => bool)   public evmRegistered;
    /// @dev Reverse mapping: UID → address that claimed it (prevents squatting).
    mapping(uint16 => address) public uidToEvm;

    // ── Storage gap for future upgrades ─────────────────────────

    uint256[50] private __gap;

    // ── Events ────────────────────────────────────────────────────

    event EndpointClaimed(address indexed miner, bytes32 indexed endpointHash, string endpoint);
    event EndpointReleased(address indexed miner, bytes32 indexed endpointHash);
    event TEEAttestationRegistered(address indexed miner, string platform, bytes32 enclavePubKey);
    event TEEAttestationRevoked(address indexed miner);
    event ModelRegistered(address indexed miner, string modelId, string endpoint);
    event ModelReactivated(address indexed miner, uint256 index, string modelId);
    event ModelRenewed(address indexed miner, uint256 index);
    event ModelDeactivated(address indexed miner, uint256 index);
    event EndpointUpdated(address indexed miner, uint256 index, string newEndpoint);
    event OfflineReported(address indexed reporter, address indexed miner, uint256 index);
    event ModelCleanedUp(address indexed miner, uint256 index);
    event DemandUpdated(address indexed updater, uint64 epoch, bytes32 receiptHash);
    event EvmRegistered(address indexed evmAddr, uint16 uid);
    event EvmRegistrationReset(uint16 indexed uid, address oldAddr);

    // ── Constructor (disable initializers for UUPS) ─────────────

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ─────────────────────────────────────────────

    /// @param _netuid       The Bittensor subnet this registry serves.
    /// @param metaAddr      MetagraphPrecompile address (address(0) = default 0x802).
    /// @param uidLookupAddr UidLookupPrecompile address (address(0) = default 0x806).
    function initialize(uint16 _netuid, address metaAddr, address uidLookupAddr) external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
        netuid = _netuid;
        META = IMetagraph(metaAddr == address(0) ? DEFAULT_META : metaAddr);
        UID_LOOKUP = IUidLookup(uidLookupAddr == address(0) ? DEFAULT_UID_LOOKUP : uidLookupAddr);
    }

    // ── UUPS upgrade authorization ──────────────────────────────

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── EVM ↔ UID registration with hotkey ownership proof ─────────
    //
    // The UidLookup precompile (0x806) has a pagination bug (OTF subtensor
    // PR #1774) — .take(limit) runs before .filter_map(), so it only checks
    // the first N storage entries.  Until that's fixed upstream, neurons
    // prove hotkey ownership via SR25519 signature verification:
    //   1. META.getHotkey(netuid, uid) → SR25519 public key
    //   2. ISr25519Verify precompile (0x403) verifies the signature
    //
    // When UidLookup is fixed, _resolveUid() already tries it first.

    /// @notice Register EVM address → UID mapping with cryptographic proof.
    ///         The caller must provide an SR25519 signature from the hotkey
    ///         that owns the claimed UID on this subnet.
    /// @param uid   The UID to claim on this subnet.
    /// @param sigR  First 32 bytes of the SR25519 signature.
    /// @param sigS  Last 32 bytes of the SR25519 signature.
    function registerEvm(uint16 uid, bytes32 sigR, bytes32 sigS) external {
        // Verify UID exists on the subnet
        require(uid < META.getUidCount(netuid), "UID does not exist on subnet");
        // Prevent one address claiming multiple UIDs
        require(!evmRegistered[msg.sender] || evmToUid[msg.sender] == uid,
                "Already registered with different UID");
        // Prevent UID squatting — one UID per address
        address existing = uidToEvm[uid];
        require(existing == address(0) || existing == msg.sender,
                "UID already claimed by another address");

        // Verify hotkey ownership via SR25519 signature
        bytes32 hotkey = META.getHotkey(netuid, uid);
        require(hotkey != bytes32(0), "UID has no hotkey");
        bytes32 message = keccak256(abi.encodePacked(msg.sender, uid, netuid, address(this)));
        require(ISr25519Verify(SR25519_VERIFY).verify(message, hotkey, sigR, sigS),
                "Invalid SR25519 signature - caller does not own this hotkey");

        evmToUid[msg.sender] = uid;
        evmRegistered[msg.sender] = true;
        uidToEvm[uid] = msg.sender;
        emit EvmRegistered(msg.sender, uid);
    }

    /// @notice Reset a UID registration (e.g. when a neuron rotates keys).
    ///         Only callable by the SN owner or the address that owns the UID.
    function resetEvmRegistration(uint16 uid) external {
        address old = uidToEvm[uid];
        require(old != address(0), "UID not registered");
        require(msg.sender == old || msg.sender == owner(), "Not authorized");
        delete evmRegistered[old];
        delete evmToUid[old];
        delete uidToEvm[uid];
        emit EvmRegistrationReset(uid, old);
    }

    // ── Internal helpers ─────────────────────────────────────────

    /// @dev Resolve msg.sender to a UID.
    ///      Tries the UidLookup precompile first; falls back to contract-level mapping.
    function _resolveUid(address caller) internal view returns (uint16) {
        // Try native precompile first (may revert or return empty on testnet)
        try UID_LOOKUP.uidLookup(netuid, caller, 1) returns (IUidLookup.LookupItem[] memory items) {
            if (items.length > 0) {
                return items[0].uid;
            }
        } catch {
            // Precompile not available or reverted — fall through to self-registration
        }
        // Fallback: contract-level self-registration
        require(evmRegistered[caller], "Not registered - call registerEvm(uid) first");
        return evmToUid[caller];
    }

    /// @dev Revert if caller already has an active AND non-expired entry with the
    ///      same (modelId, endpoint, quant). Expired entries are candidates for
    ///      reactivation, not duplicates.
    function _requireNoDuplicate(
        string calldata modelId,
        string calldata endpoint,
        string calldata quant
    ) internal view {
        MinerModel[] storage models = minerModels[msg.sender];
        bytes32 hModel = keccak256(bytes(modelId));
        bytes32 hEndpoint = keccak256(bytes(endpoint));
        bytes32 hQuant = keccak256(bytes(quant));
        for (uint256 i = 0; i < models.length; i++) {
            if (models[i].active &&
                models[i].expiresAt > block.timestamp &&
                keccak256(bytes(models[i].modelId)) == hModel &&
                keccak256(bytes(models[i].endpoint)) == hEndpoint &&
                keccak256(bytes(models[i].quant)) == hQuant) {
                revert("Duplicate active entry - use renewModel() or deactivateModel() first");
            }
        }
    }

    /// @dev Find an expired or inactive entry with the same (modelId, endpoint, quant).
    ///      Returns (index, true) if found, (0, false) if not.
    ///      Used by registerModel() to reactivate an old slot instead of creating
    ///      a new array entry — preserves the model_index for score continuity.
    function _findReactivatable(
        string calldata modelId,
        string calldata endpoint,
        string calldata quant
    ) internal view returns (uint256, bool) {
        MinerModel[] storage models = minerModels[msg.sender];
        bytes32 hModel = keccak256(bytes(modelId));
        bytes32 hEndpoint = keccak256(bytes(endpoint));
        bytes32 hQuant = keccak256(bytes(quant));
        for (uint256 i = 0; i < models.length; i++) {
            // Skip active+non-expired entries (caught by _requireNoDuplicate)
            if (models[i].active && models[i].expiresAt > block.timestamp) {
                continue;
            }
            if (keccak256(bytes(models[i].modelId)) == hModel &&
                keccak256(bytes(models[i].endpoint)) == hEndpoint &&
                keccak256(bytes(models[i].quant)) == hQuant) {
                return (i, true);
            }
        }
        return (0, false);
    }

    /// @dev Claim an endpoint for msg.sender. Reverts if already owned by another address.
    function _claimEndpoint(string calldata endpoint) internal {
        bytes32 h = keccak256(bytes(endpoint));
        address current = _endpointOwner[h];
        require(current == address(0) || current == msg.sender, "Endpoint already claimed by another miner");
        if (current == address(0)) {
            _endpointOwner[h] = msg.sender;
            emit EndpointClaimed(msg.sender, h, endpoint);
        }
    }

    /// @dev Release an endpoint. Only the owner can release.
    function _releaseEndpoint(string memory endpoint) internal {
        bytes32 h = keccak256(bytes(endpoint));
        if (_endpointOwner[h] == msg.sender) {
            _endpointOwner[h] = address(0);
            emit EndpointReleased(msg.sender, h);
        }
    }

    /// @dev Release an endpoint owned by a specific address (for cleanup, called by anyone).
    function _releaseEndpointFor(string memory endpoint, address miner) internal {
        bytes32 h = keccak256(bytes(endpoint));
        if (_endpointOwner[h] == miner) {
            _endpointOwner[h] = address(0);
            emit EndpointReleased(miner, h);
        }
    }

    // ── Modifiers ─────────────────────────────────────────────────

    /// @dev Verify caller is a registered neuron (active on subnet).
    modifier onlyRegisteredNeuron() {
        _resolveUid(msg.sender);
        _;
    }

    /// @dev Verify caller is a registered validator on the subnet.
    modifier onlyValidator() {
        uint16 uid = _resolveUid(msg.sender);
        require(META.getValidatorStatus(netuid, uid), "Not a validator");
        _;
    }

    // ── Miner actions (only registered neurons) ───────────────────

    /// @notice Register a new model endpoint (24h lease).
    ///         Endpoint must not be claimed by another miner (anti-hijacking).
    ///         Reverts if the caller already has an active entry with the same
    ///         (modelId, endpoint, quant) — use renewModel() to extend a lease.
    function registerModel(
        string calldata modelId,
        string calldata endpoint,
        bytes32 modelSpecRef,
        string calldata quant,
        uint32 maxContextLen
    ) external onlyRegisteredNeuron {
        // Prevent duplicate active entries for the same model+endpoint+quant
        _requireNoDuplicate(modelId, endpoint, quant);
        _claimEndpoint(endpoint);

        // Try to reactivate an expired/inactive entry with the same tuple.
        // This preserves the array index (model_index) so the validator's
        // per-(address, model_index) score history stays intact.
        (uint256 reactivateIdx, bool found) = _findReactivatable(modelId, endpoint, quant);
        if (found) {
            MinerModel storage m = minerModels[msg.sender][reactivateIdx];
            m.modelSpecRef = modelSpecRef;
            m.maxContextLen = maxContextLen;
            m.expiresAt = uint64(block.timestamp) + LEASE_DURATION;
            m.active = true;
            // Reset failure tracking for the reactivated slot
            failureCount[msg.sender][reactivateIdx] = 0;

            emit ModelReactivated(msg.sender, reactivateIdx, modelId);
            return;
        }

        // No reactivatable slot — create new entry
        minerModels[msg.sender].push(MinerModel({
            modelId: modelId,
            endpoint: endpoint,
            modelSpecRef: modelSpecRef,
            quant: quant,
            maxContextLen: maxContextLen,
            expiresAt: uint64(block.timestamp) + LEASE_DURATION,
            active: true
        }));
        _modelProviders[modelId].push(msg.sender);

        emit ModelRegistered(msg.sender, modelId, endpoint);
    }

    /// @notice Renew lease — miner calls this periodically (cheap tx).
    function renewModel(uint256 index) external onlyRegisteredNeuron {
        require(index < minerModels[msg.sender].length, "Invalid index");
        MinerModel storage m = minerModels[msg.sender][index];
        require(m.expiresAt > block.timestamp, "Already expired, re-register");
        m.expiresAt = uint64(block.timestamp) + LEASE_DURATION;
        m.active = true;
        failureCount[msg.sender][index] = 0;
        // Clear reporter tracking for this entry
        // (cannot delete nested mapping, but failureCount reset prevents re-deactivation)

        emit ModelRenewed(msg.sender, index);
    }

    /// @notice Miner voluntarily deactivates a model.
    ///         Releases the endpoint so it can be reclaimed.
    function deactivateModel(uint256 index) external onlyRegisteredNeuron {
        require(index < minerModels[msg.sender].length, "Invalid index");
        _releaseEndpoint(minerModels[msg.sender][index].endpoint);
        minerModels[msg.sender][index].active = false;

        emit ModelDeactivated(msg.sender, index);
    }

    /// @notice Update endpoint (e.g. IP change).
    ///         Releases the old endpoint and claims the new one.
    function updateEndpoint(uint256 index, string calldata newEndpoint)
        external onlyRegisteredNeuron
    {
        require(index < minerModels[msg.sender].length, "Invalid index");
        _releaseEndpoint(minerModels[msg.sender][index].endpoint);
        _claimEndpoint(newEndpoint);
        minerModels[msg.sender][index].endpoint = newEndpoint;

        emit EndpointUpdated(msg.sender, index, newEndpoint);
    }

    // ── Validator actions (only registered validators) ────────────

    /// @notice Report a miner-model as unreachable.
    ///         Only validators (as determined by live metagraph) can report.
    ///         Each validator can only report once per miner-model entry.
    function reportOffline(address miner, uint256 modelIndex)
        external onlyValidator
    {
        require(modelIndex < minerModels[miner].length, "Invalid index");
        require(!reported[miner][modelIndex][msg.sender], "Already reported");

        reported[miner][modelIndex][msg.sender] = true;
        failureCount[miner][modelIndex]++;

        if (failureCount[miner][modelIndex] >= OFFLINE_THRESHOLD) {
            minerModels[miner][modelIndex].active = false;
        }

        emit OfflineReported(msg.sender, miner, modelIndex);
    }

    /// @notice Post per-model demand scores derived from epoch receipt data.
    ///         Any validator can write; any validator can verify/overwrite.
    ///         Epoch must be strictly increasing to prevent stale writes.
    function updateDemandScores(
        string[] calldata modelIds,
        uint16[] calldata scores,
        bytes32 receiptSetHash,
        uint64 epochNumber
    ) external onlyValidator {
        require(modelIds.length == scores.length, "Array length mismatch");
        require(epochNumber > demandEpoch, "Stale epoch");

        for (uint256 i = 0; i < modelIds.length; i++) {
            require(scores[i] <= 10000, "Score exceeds 10000 bps");
            modelDemandScore[modelIds[i]] = scores[i];
        }

        demandReceiptHash = receiptSetHash;
        demandEpoch = epochNumber;
        demandUpdater = msg.sender;

        emit DemandUpdated(msg.sender, epochNumber, receiptSetHash);
    }

    // ── TEE attestation (only registered neurons) ────────────────

    /// @notice Register or update TEE attestation for this miner.
    function registerTEEAttestation(
        string calldata platform,
        bytes32 enclavePubKey,
        bytes32 attestationHash,
        bytes32 modelWeightHash,
        bytes32 codeMeasurement
    ) external onlyRegisteredNeuron {
        require(enclavePubKey != bytes32(0), "Empty enclave public key");
        require(attestationHash != bytes32(0), "Empty attestation hash");

        teeCapability[msg.sender] = TEECapability({
            enabled: true,
            platform: platform,
            enclavePubKey: enclavePubKey,
            attestationHash: attestationHash,
            attestedAt: uint64(block.number),
            modelWeightHash: modelWeightHash,
            codeMeasurement: codeMeasurement
        });

        emit TEEAttestationRegistered(msg.sender, platform, enclavePubKey);
    }

    /// @notice Revoke TEE attestation (e.g. key rotation, enclave restart).
    function revokeTEEAttestation() external onlyRegisteredNeuron {
        delete teeCapability[msg.sender];
        emit TEEAttestationRevoked(msg.sender);
    }

    /// @notice Check if a miner has active TEE attestation.
    function hasTEE(address miner) external view returns (bool) {
        return teeCapability[miner].enabled;
    }

    /// @notice Get full TEE capability for a miner.
    function getTEECapability(address miner)
        external view returns (TEECapability memory)
    {
        return teeCapability[miner];
    }

    // ── Read functions (free view calls) ──────────────────────────

    /// @notice Get the UID associated with an EVM address on this subnet.
    ///         Returns (uid, true) if associated, (0, false) if not.
    ///         Tries native UidLookup precompile first, falls back to
    ///         contract-level self-registration (registerEvm).
    function getAssociatedUid(address evmAddr)
        external view returns (uint16 uid, bool exists)
    {
        // Try native precompile first (may revert or return empty on testnet).
        try UID_LOOKUP.uidLookup(netuid, evmAddr, 1) returns (IUidLookup.LookupItem[] memory items) {
            if (items.length > 0) return (items[0].uid, true);
        } catch {}

        // Fallback: contract-level self-registration.
        if (evmRegistered[evmAddr]) return (evmToUid[evmAddr], true);
        return (0, false);
    }

    /// @notice Get all models for a miner.
    function getMinerModels(address miner)
        external view returns (MinerModel[] memory)
    {
        return minerModels[miner];
    }

    /// @notice Get number of model entries for a miner.
    function getMinerModelCount(address miner) external view returns (uint256) {
        return minerModels[miner].length;
    }

    /// @notice Get all miners serving a specific model.
    function getProvidersForModel(string calldata modelId)
        external view returns (address[] memory)
    {
        return _modelProviders[modelId];
    }

    /// @notice Check if a specific entry is currently active and not expired.
    function isModelActive(address miner, uint256 index)
        external view returns (bool)
    {
        if (index >= minerModels[miner].length) return false;
        MinerModel storage m = minerModels[miner][index];
        return m.active && m.expiresAt > block.timestamp;
    }

    /// @notice Get the demand score for a single model.
    function getModelDemandScore(string calldata modelId)
        external view returns (uint16)
    {
        return modelDemandScore[modelId];
    }

    /// @notice Batch-read demand scores for multiple models.
    function getModelDemandScores(string[] calldata modelIds)
        external view returns (uint16[] memory)
    {
        uint16[] memory result = new uint16[](modelIds.length);
        for (uint256 i = 0; i < modelIds.length; i++) {
            result[i] = modelDemandScore[modelIds[i]];
        }
        return result;
    }

    /// @notice Check who owns a given endpoint (address(0) = free).
    function getEndpointOwner(string calldata endpoint) external view returns (address) {
        return _endpointOwner[keccak256(bytes(endpoint))];
    }

    // ── Cleanup (anyone can call) ─────────────────────────────────

    /// @notice Remove entries expired for > 30 days.
    ///         Anyone can call — the caller gets a natural gas refund from storage
    ///         slot clearing (SSTORE to zero = 4800 gas refund per slot).
    function cleanup(address miner, uint256 index) external {
        require(index < minerModels[miner].length, "Invalid index");
        require(
            minerModels[miner][index].expiresAt + CLEANUP_GRACE < block.timestamp,
            "Not yet eligible for cleanup"
        );

        // Release endpoint ownership before removal
        _releaseEndpointFor(minerModels[miner][index].endpoint, miner);

        // Swap-and-pop removal
        uint256 last = minerModels[miner].length - 1;
        if (index != last) {
            minerModels[miner][index] = minerModels[miner][last];
        }
        minerModels[miner].pop();

        emit ModelCleanedUp(miner, index);
    }
}
