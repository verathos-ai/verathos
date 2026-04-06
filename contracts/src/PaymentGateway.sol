// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IStakingV2.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";

/// @title PaymentGateway — prepaid credit system for Verathos inference
/// @notice Users deposit TAO to purchase inference credits. Deposits are split:
///         - ownerCutBps (default 0%, max 20%) → owner treasury
///         - remainder → staked as alpha on the subnet via StakingV2 precompile
///
///         Credit accounting is off-chain (validator-side DB). The on-chain
///         contract only tracks deposits and emits events that validators watch.
///
///         The staking precompile address is configurable for testing (Anvil
///         doesn't have real precompiles). On mainnet/testnet, pass address(0)
///         to use the baked-in Bittensor address.
contract PaymentGateway is Initializable, UUPSUpgradeable, OwnableUpgradeable, ReentrancyGuardUpgradeable {

    // ── Precompile ──────────────────────────────────────────────────

    IStakingV2 public STAKING;
    address constant DEFAULT_STAKING = 0x0000000000000000000000000000000000000805;

    // ── Subnet ──────────────────────────────────────────────────────

    uint16 public netuid;

    // ── Owner config ────────────────────────────────────────────────

    address public ownerTreasury;
    uint16  public ownerCutBps;  // basis points, default 0 = 0%

    uint16 constant MAX_OWNER_CUT_BPS = 2000;  // 20% cap

    // ── Deposit tracking ────────────────────────────────────────────

    mapping(address => uint256) public totalDeposited;

    // ── Emergency withdraw timelock ─────────────────────────────────

    uint256 public emergencyAnnouncedAt;
    uint256 constant EMERGENCY_TIMELOCK = 7 days;

    // ── Events ──────────────────────────────────────────────────────

    event Deposit(
        address indexed user,
        bytes32 indexed validatorHotkey,
        uint256 taoAmount
    );

    event OwnerCutUpdated(uint16 oldBps, uint16 newBps);
    event TreasuryUpdated(address oldTreasury, address newTreasury);
    event EmergencyAnnounced(uint256 timestamp);
    event EmergencyWithdraw(bytes32 hotkey, uint256 amount);

    // ── Constructor (disabled for UUPS) ─────────────────────────────

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ─────────────────────────────────────────────────

    function initialize(
        uint16 _netuid,
        address _ownerTreasury,
        uint16 _ownerCutBps,
        address stakingAddr
    ) external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();

        require(_ownerTreasury != address(0), "Zero treasury");
        require(_ownerCutBps <= MAX_OWNER_CUT_BPS, "Cut too high");

        STAKING = IStakingV2(
            stakingAddr == address(0) ? DEFAULT_STAKING : stakingAddr
        );
        netuid = _netuid;
        ownerTreasury = _ownerTreasury;
        ownerCutBps = _ownerCutBps;
    }

    // ── UUPS authorization ──────────────────────────────────────────

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── Deposit ─────────────────────────────────────────────────────

    /// @notice Deposit TAO to purchase inference credits.
    ///         The owner cut goes to the treasury. The remainder is staked
    ///         as alpha on the subnet to the specified validator hotkey.
    /// @param validatorHotkey The validator that will serve this user.
    function deposit(bytes32 validatorHotkey) external payable nonReentrant {
        require(msg.value > 0, "Zero deposit");

        uint256 ownerCut = (msg.value * ownerCutBps) / 10000;
        uint256 stakeCut = msg.value - ownerCut;

        // Effects first (checks-effects-interactions pattern)
        totalDeposited[msg.sender] += msg.value;
        emit Deposit(msg.sender, validatorHotkey, msg.value);

        // Interactions: transfer owner cut
        if (ownerCut > 0) {
            (bool sent,) = payable(ownerTreasury).call{value: ownerCut}("");
            require(sent, "Treasury transfer failed");
        }

        // Interactions: stake remainder as alpha via StakingV2 precompile.
        // V2 deducts from the caller's substrate balance directly (Bittensor
        // unifies EVM and substrate balances). Do NOT forward msg.value —
        // that would reduce balance twice. Amount must be in RAO (1e9/TAO).
        if (stakeCut > 0) {
            uint256 stakeRao = stakeCut / 1e9;
            require(stakeRao > 0, "Stake amount too small");
            STAKING.addStake(validatorHotkey, stakeRao, uint256(netuid));
        }
    }

    // ── Read functions ──────────────────────────────────────────────

    /// @notice Get total TAO deposited by a user (across all validators).
    function getDeposited(address user) external view returns (uint256) {
        return totalDeposited[user];
    }

    // ── Owner admin ─────────────────────────────────────────────────

    /// @notice Update the owner cut percentage (max 20%).
    function setOwnerCut(uint16 newBps) external onlyOwner {
        require(newBps <= MAX_OWNER_CUT_BPS, "Cut too high");
        uint16 oldBps = ownerCutBps;
        ownerCutBps = newBps;
        emit OwnerCutUpdated(oldBps, newBps);
    }

    /// @notice Update the treasury address.
    function setTreasury(address newTreasury) external onlyOwner {
        require(newTreasury != address(0), "Zero treasury");
        address oldTreasury = ownerTreasury;
        ownerTreasury = newTreasury;
        emit TreasuryUpdated(oldTreasury, newTreasury);
    }

    // ── Emergency withdraw (7-day timelock) ─────────────────────────

    /// @notice Announce intent to emergency withdraw. Starts 7-day timelock.
    function announceEmergencyWithdraw() external onlyOwner {
        emergencyAnnouncedAt = block.timestamp;
        emit EmergencyAnnounced(block.timestamp);
    }

    /// @notice Execute emergency withdraw after timelock expires.
    ///         Unstakes alpha from the subnet for a specific hotkey.
    /// @param amount Amount to unstake in RAO (1e9 per TAO).
    function executeEmergencyWithdraw(bytes32 hotkey, uint256 amount) external onlyOwner {
        require(emergencyAnnouncedAt > 0, "Not announced");
        require(
            block.timestamp >= emergencyAnnouncedAt + EMERGENCY_TIMELOCK,
            "Timelock not expired"
        );

        STAKING.removeStake(hotkey, amount, uint256(netuid));
        emergencyAnnouncedAt = 0;  // reset after execution

        emit EmergencyWithdraw(hotkey, amount);
    }

    /// @notice Cancel a pending emergency withdraw announcement.
    function cancelEmergencyWithdraw() external onlyOwner {
        emergencyAnnouncedAt = 0;
    }

    // Accept TAO (for receiving unstaked alpha converted back to TAO)
    receive() external payable {}

    // ── Storage gap ─────────────────────────────────────────────────

    uint256[50] private __gap;
}
