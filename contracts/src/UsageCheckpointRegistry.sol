// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IMetagraph.sol";
import "./interfaces/IUidLookup.sol";
import {ISr25519Verify} from "./interfaces/ISr25519Verify.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title UsageCheckpointRegistry — on-chain usage checkpoints for disaster recovery
/// @notice Validators periodically record a Merkle root of all user balances.
///         One event per checkpoint — constant gas cost regardless of user count.
///
///         Recovery: off-chain Merkle tree (stored alongside DB backups) is
///         verified against the on-chain root. Deposits are always fully
///         recoverable from PaymentGateway/Base chain events.
///
///         Access control: only validators (verified via MetagraphPrecompile)
///         can record checkpoints. EVM registration requires SR25519 hotkey proof.
contract UsageCheckpointRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable {

    // ── Precompile addresses ──────────────────────────────────────

    IMetagraph public META;
    IUidLookup public UID_LOOKUP;

    address constant DEFAULT_META = 0x0000000000000000000000000000000000000802;
    address constant DEFAULT_UID_LOOKUP = 0x0000000000000000000000000000000000000806;
    address constant SR25519_VERIFY = 0x0000000000000000000000000000000000000403;

    // ── Subnet ────────────────────────────────────────────────────

    uint16 public netuid;

    // ── EVM ↔ UID registration with hotkey ownership proof ─────────

    mapping(address => uint16) public evmToUid;
    mapping(address => bool)   public evmRegistered;
    mapping(uint16 => address) public uidToEvm;

    // ── Events ──────────────────────────────────────────────────────

    /// @notice Emitted once per checkpoint. The Merkle root commits to all
    ///         user balances at this point in time. The full tree is stored
    ///         off-chain alongside DB backups.
    event UsageCheckpoint(
        address indexed validator,
        bytes32 merkleRoot,
        uint32  userCount,
        uint256 totalConsumedWei,
        uint256 totalConsumedUsdMicros,
        uint256 timestamp
    );

    event EvmRegistered(address indexed evmAddr, uint16 uid);
    event EvmRegistrationReset(uint16 indexed uid, address oldAddr);

    // ── Constructor (disabled for UUPS) ─────────────────────────────

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ─────────────────────────────────────────────────

    /// @param _netuid       The Bittensor subnet this registry serves.
    /// @param metaAddr      MetagraphPrecompile address (address(0) = default 0x802).
    /// @param uidLookupAddr UidLookupPrecompile address (address(0) = default 0x806).
    function initialize(uint16 _netuid, address metaAddr, address uidLookupAddr) external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();

        META = IMetagraph(metaAddr == address(0) ? DEFAULT_META : metaAddr);
        UID_LOOKUP = IUidLookup(uidLookupAddr == address(0) ? DEFAULT_UID_LOOKUP : uidLookupAddr);
        netuid = _netuid;
    }

    // ── UUPS authorization ──────────────────────────────────────────

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── Internal: UID resolution (same pattern as MinerRegistry) ────

    function _resolveUid(address caller) internal view returns (uint16) {
        try UID_LOOKUP.uidLookup(netuid, caller, 1) returns (IUidLookup.LookupItem[] memory items) {
            if (items.length > 0) {
                return items[0].uid;
            }
        } catch {}
        require(evmRegistered[caller], "Not registered - call registerEvm(uid) first");
        return evmToUid[caller];
    }

    function _requireValidator(address caller) internal view {
        uint16 uid = _resolveUid(caller);
        require(META.getValidatorStatus(netuid, uid), "Not a validator");
    }

    // ── EVM registration with hotkey ownership proof ───────────────

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

    function resetEvmRegistration(uint16 uid) external {
        address old = uidToEvm[uid];
        require(old != address(0), "UID not registered");
        require(msg.sender == old || msg.sender == owner(), "Not authorized");
        delete evmRegistered[old];
        delete evmToUid[old];
        delete uidToEvm[uid];
        emit EvmRegistrationReset(uid, old);
    }

    // ── Checkpoint recording ────────────────────────────────────────

    /// @notice Record a Merkle root of all user consumed balances.
    /// @dev    Called by validator proxy periodically (default: every ~24h).
    ///         Emits a single event — constant gas cost regardless of user count.
    ///         The full Merkle tree (leaves = user balances) is stored off-chain.
    /// @param merkleRoot           Root of the user balance Merkle tree.
    ///                             Leaf = keccak256(abi.encodePacked(user, taoWei, usdMicros)).
    /// @param userCount            Number of users in the tree.
    /// @param totalConsumedWei     Sum of all consumed TAO (wei) across all users.
    /// @param totalConsumedUsdMicros Sum of all consumed USD (microdollars).
    function recordCheckpoint(
        bytes32 merkleRoot,
        uint32  userCount,
        uint256 totalConsumedWei,
        uint256 totalConsumedUsdMicros
    ) external {
        _requireValidator(msg.sender);
        emit UsageCheckpoint(
            msg.sender,
            merkleRoot,
            userCount,
            totalConsumedWei,
            totalConsumedUsdMicros,
            block.timestamp
        );
    }

    // ── Storage gap ─────────────────────────────────────────────────

    uint256[50] private __gap;
}
