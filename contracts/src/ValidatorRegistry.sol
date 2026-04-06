// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IMetagraph.sol";
import "./interfaces/IUidLookup.sol";
import "./interfaces/IStakingV2.sol";
import {ISr25519Verify} from "./interfaces/ISr25519Verify.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title ValidatorRegistry v3 — stake-gated proxy registration
/// @notice Validators register their proxy endpoints so users can discover
///         which proxies are available on the subnet. Only neurons with
///         validator status (as verified by the Metagraph precompile) can
///         register. Proxy registration additionally requires a minimum alpha
///         stake on the subnet (configurable by the owner).
///
///         The subnet owner sets a `defaultValidator` — the recommended proxy
///         for new users. Users can choose any registered validator instead.
///
///         Access control mirrors MinerRegistry: callers must have called the
///         `associate_evm_key` Substrate extrinsic and be resolved via UidLookup
///         precompile (0x806) or contract-level self-registration.
contract ValidatorRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable {

    // ── Precompile addresses ──────────────────────────────────────

    IMetagraph public META;
    IUidLookup public UID_LOOKUP;
    IStakingV2 public STAKING;

    address constant DEFAULT_META = 0x0000000000000000000000000000000000000802;
    address constant DEFAULT_UID_LOOKUP = 0x0000000000000000000000000000000000000806;
    address constant DEFAULT_STAKING_V2 = 0x0000000000000000000000000000000000000805;
    address constant SR25519_VERIFY = 0x0000000000000000000000000000000000000403;

    // ── Subnet ────────────────────────────────────────────────────

    uint16 public netuid;

    // ── Stake thresholds ──────────────────────────────────────────

    /// @notice Minimum alpha stake (RAO) to register as a validator (no proxy).
    ///         0 = no minimum (default at launch).
    uint256 public minValidatorStake;

    /// @notice Minimum alpha stake (RAO) to register with a proxy endpoint.
    ///         Must be >= minValidatorStake. 0 = no minimum.
    uint256 public minProxyStake;

    /// @notice Whitelisted addresses bypass all stake checks.
    mapping(address => bool) public whitelisted;

    // ── Validator data ────────────────────────────────────────────

    struct ValidatorInfo {
        string   proxyEndpoint;  // HTTPS URL of the proxy (e.g. "https://proxy.example.com:8080")
        uint16   uid;            // Bittensor UID on this subnet
        uint64   registeredAt;   // block.timestamp of registration
        uint64   updatedAt;      // block.timestamp of last update
        bool     active;         // validator can deactivate voluntarily
    }

    /// @dev EVM address → validator info. One entry per validator.
    mapping(address => ValidatorInfo) public validators;

    /// @dev List of all validator addresses that have ever registered.
    ///      Used for enumeration. Inactive/removed entries remain but
    ///      are filtered out by getActiveValidators().
    address[] public validatorList;
    mapping(address => bool) internal _inList;

    /// @dev The default validator recommended to new users.
    ///      Set by the SN owner. Must be a registered, active validator.
    address public defaultValidator;

    // ── EVM ↔ UID registration with hotkey ownership proof ─────────

    mapping(address => uint16) public evmToUid;
    mapping(address => bool)   public evmRegistered;
    mapping(uint16 => address) public uidToEvm;

    // ── Events ────────────────────────────────────────────────────

    event ValidatorRegistered(address indexed validator, uint16 uid, string proxyEndpoint);
    event ValidatorUpdated(address indexed validator, string proxyEndpoint);
    event ValidatorDeactivated(address indexed validator);
    event ValidatorReactivated(address indexed validator);
    event DefaultValidatorSet(address indexed validator);
    event EvmRegistered(address indexed evmAddr, uint16 uid);
    event EvmRegistrationReset(uint16 indexed uid, address oldAddr);
    event MinValidatorStakeSet(uint256 oldValue, uint256 newValue);
    event MinProxyStakeSet(uint256 oldValue, uint256 newValue);
    event WhitelistUpdated(address indexed addr, bool status);

    // ── Constructor (disabled for UUPS) ─────────────────────────────

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ─────────────────────────────────────────────────

    /// @param _netuid       The Bittensor subnet this registry serves.
    /// @param metaAddr      MetagraphPrecompile address (address(0) = default 0x802).
    /// @param uidLookupAddr UidLookupPrecompile address (address(0) = default 0x806).
    /// @param stakingAddr   StakingV2 precompile address (address(0) = default 0x805).
    function initialize(uint16 _netuid, address metaAddr, address uidLookupAddr, address stakingAddr) external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();

        META = IMetagraph(metaAddr == address(0) ? DEFAULT_META : metaAddr);
        UID_LOOKUP = IUidLookup(uidLookupAddr == address(0) ? DEFAULT_UID_LOOKUP : uidLookupAddr);
        STAKING = IStakingV2(stakingAddr == address(0) ? DEFAULT_STAKING_V2 : stakingAddr);
        netuid = _netuid;
    }

    // ── UUPS authorization ──────────────────────────────────────────

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── Internal: UID resolution ──────────────────────────────────

    /// @dev Resolve msg.sender to a UID. Same pattern as MinerRegistry.
    function _resolveUid(address caller) internal view returns (uint16) {
        try UID_LOOKUP.uidLookup(netuid, caller, 1) returns (IUidLookup.LookupItem[] memory items) {
            if (items.length > 0) {
                return items[0].uid;
            }
        } catch {}
        require(evmRegistered[caller], "Not registered - call registerEvm(uid) first");
        return evmToUid[caller];
    }

    /// @dev Verify caller is a validator on the subnet via Metagraph precompile.
    function _requireValidator(address caller) internal view returns (uint16) {
        uint16 uid = _resolveUid(caller);
        require(META.getValidatorStatus(netuid, uid), "Not a validator");
        return uid;
    }

    /// @dev Check that a validator meets the minimum alpha stake requirement.
    ///      Owner and whitelisted addresses are always exempt.
    function _requireStake(address caller, uint16 uid, uint256 minStake) internal view {
        if (minStake == 0) return;
        if (caller == owner()) return;
        if (whitelisted[caller]) return;
        bytes32 hotkey = META.getHotkey(netuid, uid);
        uint256 alphaStake = STAKING.getTotalAlphaStaked(hotkey, uint256(netuid));
        require(alphaStake >= minStake, "Insufficient alpha stake");
    }

    // ── EVM registration with hotkey ownership proof ───────────────

    /// @notice Register EVM address → UID mapping with SR25519 hotkey proof.
    /// @param uid   The UID to claim on this subnet.
    /// @param sigR  First 32 bytes of the SR25519 signature.
    /// @param sigS  Last 32 bytes of the SR25519 signature.
    function registerEvm(uint16 uid, bytes32 sigR, bytes32 sigS) external {
        require(uid < META.getUidCount(netuid), "UID does not exist on subnet");
        require(!evmRegistered[msg.sender] || evmToUid[msg.sender] == uid,
                "Already registered with different UID");
        address existing = uidToEvm[uid];
        require(existing == address(0) || existing == msg.sender,
                "UID already claimed by another address");

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

    /// @notice Reset a UID registration (key rotation).
    function resetEvmRegistration(uint16 uid) external {
        address old = uidToEvm[uid];
        require(old != address(0), "UID not registered");
        require(msg.sender == old || msg.sender == owner(), "Not authorized");
        delete evmRegistered[old];
        delete evmToUid[old];
        delete uidToEvm[uid];
        emit EvmRegistrationReset(uid, old);
    }

    // ── Validator registration ────────────────────────────────────

    /// @notice Register as a validator. Pass an empty string if you do not
    ///         run a proxy yet — you can call updateEndpoint() later to add one.
    ///         Caller must be a validator on the subnet (verified via Metagraph).
    ///         If registering with a proxy endpoint, must meet minProxyStake.
    ///         Otherwise must meet minValidatorStake.
    function register(string calldata proxyEndpoint) external {
        uint16 uid = _requireValidator(msg.sender);
        bool hasProxy = bytes(proxyEndpoint).length > 0;
        _requireStake(msg.sender, uid, hasProxy ? minProxyStake : minValidatorStake);

        ValidatorInfo storage v = validators[msg.sender];
        if (v.registeredAt == 0) {
            // New registration
            v.registeredAt = uint64(block.timestamp);
        }
        v.proxyEndpoint = proxyEndpoint;
        v.uid = uid;
        v.updatedAt = uint64(block.timestamp);
        v.active = true;

        if (!_inList[msg.sender]) {
            validatorList.push(msg.sender);
            _inList[msg.sender] = true;
        }

        emit ValidatorRegistered(msg.sender, uid, proxyEndpoint);
    }

    /// @notice Update proxy endpoint URL. Pass empty string to clear it.
    ///         Setting a non-empty endpoint requires minProxyStake.
    function updateEndpoint(string calldata newEndpoint) external {
        uint16 uid = _requireValidator(msg.sender);

        ValidatorInfo storage v = validators[msg.sender];
        require(v.registeredAt > 0, "Not registered");

        if (bytes(newEndpoint).length > 0) {
            _requireStake(msg.sender, uid, minProxyStake);
        }

        v.proxyEndpoint = newEndpoint;
        v.updatedAt = uint64(block.timestamp);

        emit ValidatorUpdated(msg.sender, newEndpoint);
    }

    /// @notice Voluntarily deactivate (e.g. going offline for maintenance).
    function deactivate() external {
        ValidatorInfo storage v = validators[msg.sender];
        require(v.registeredAt > 0, "Not registered");
        v.active = false;
        emit ValidatorDeactivated(msg.sender);
    }

    /// @notice Reactivate after deactivation. Re-checks validator status and stake.
    function reactivate() external {
        uint16 uid = _requireValidator(msg.sender);
        ValidatorInfo storage v = validators[msg.sender];
        require(v.registeredAt > 0, "Not registered");

        bool hasProxy = bytes(v.proxyEndpoint).length > 0;
        _requireStake(msg.sender, uid, hasProxy ? minProxyStake : minValidatorStake);

        v.active = true;
        v.updatedAt = uint64(block.timestamp);
        emit ValidatorReactivated(msg.sender);
    }

    // ── Owner admin ───────────────────────────────────────────────

    /// @notice Set the default validator recommended to new users.
    ///         Must be a registered, active validator.
    function setDefaultValidator(address validator) external onlyOwner {
        require(validators[validator].registeredAt > 0, "Validator not registered");
        require(validators[validator].active, "Validator not active");
        defaultValidator = validator;
        emit DefaultValidatorSet(validator);
    }

    /// @notice Set minimum alpha stake for validator registration (no proxy).
    function setMinValidatorStake(uint256 amount) external onlyOwner {
        uint256 old = minValidatorStake;
        minValidatorStake = amount;
        emit MinValidatorStakeSet(old, amount);
    }

    /// @notice Set minimum alpha stake for proxy registration.
    function setMinProxyStake(uint256 amount) external onlyOwner {
        uint256 old = minProxyStake;
        minProxyStake = amount;
        emit MinProxyStakeSet(old, amount);
    }

    /// @notice Add or remove an address from the stake-check whitelist.
    function setWhitelisted(address addr, bool status) external onlyOwner {
        whitelisted[addr] = status;
        emit WhitelistUpdated(addr, status);
    }

    // ── Read functions (free view calls) ──────────────────────────

    /// @notice Get the default validator's info and proxy endpoint.
    function getDefault() external view returns (
        address validatorAddr,
        ValidatorInfo memory info
    ) {
        require(defaultValidator != address(0), "No default validator set");
        return (defaultValidator, validators[defaultValidator]);
    }

    /// @notice Get a specific validator's info.
    function getValidator(address addr) external view returns (ValidatorInfo memory) {
        return validators[addr];
    }

    /// @notice Get all active validators with their endpoints.
    ///         Returns parallel arrays of addresses and info structs.
    function getActiveValidators() external view returns (
        address[] memory addrs,
        ValidatorInfo[] memory infos
    ) {
        // Count active first
        uint256 count = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            if (validators[validatorList[i]].active) count++;
        }

        addrs = new address[](count);
        infos = new ValidatorInfo[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            address addr = validatorList[i];
            if (validators[addr].active) {
                addrs[idx] = addr;
                infos[idx] = validators[addr];
                idx++;
            }
        }
    }

    /// @notice Get active validators that have a non-empty proxy endpoint.
    ///         Use this for proxy discovery. getActiveValidators() returns all
    ///         participants (including those without a proxy).
    function getProxyValidators() external view returns (
        address[] memory addrs,
        ValidatorInfo[] memory infos
    ) {
        uint256 count = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            address addr = validatorList[i];
            if (validators[addr].active && bytes(validators[addr].proxyEndpoint).length > 0) count++;
        }

        addrs = new address[](count);
        infos = new ValidatorInfo[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            address addr = validatorList[i];
            if (validators[addr].active && bytes(validators[addr].proxyEndpoint).length > 0) {
                addrs[idx] = addr;
                infos[idx] = validators[addr];
                idx++;
            }
        }
    }

    /// @notice Get total number of registered validators (including inactive).
    function getValidatorCount() external view returns (uint256) {
        return validatorList.length;
    }

    /// @notice Check if an address is a registered, active validator.
    function isActiveValidator(address addr) external view returns (bool) {
        return validators[addr].active && validators[addr].registeredAt > 0;
    }

    // ── Storage gap ─────────────────────────────────────────────────

    uint256[50] private __gap;
}
