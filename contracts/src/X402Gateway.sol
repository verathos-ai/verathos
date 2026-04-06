// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/ITaoFiBridge.sol";

/// @title X402Gateway — USDC collection + TaoFi bridge for x402 payments on Base
/// @notice Collects USDC from x402 facilitator settlements on Base L2.
///         Periodically bridges accumulated USDC to Bittensor EVM via TaoFi,
///         where it is swapped to TAO and staked as alpha — a PERMANENT sink.
///
///         The more the network is used, the more alpha is locked. It never
///         comes out. This creates a monotonically increasing stake floor that
///         amplifies subnet emissions via the alpha price feedback loop.
///
///         Revenue split: configurable owner cut (default 10%) to treasury as
///         USDC on Base, remainder (90%) batched and bridged for alpha staking.
///
///         The bridge() function is PUBLIC — anyone can call it (keeper bot,
///         validator, user). The caller provides slippage parameters and pays
///         Hyperlane messaging fees in ETH.
///
///         There is NO unstake function. Once bridged and staked, alpha is
///         permanently locked. Emergency withdraw only covers unbridged USDC
///         still sitting on Base (7-day timelock).
contract X402Gateway {

    // ── Immutables ─────────────────────────────────────────────────

    address public immutable USDC;
    ITaoFiBridge public immutable TAOFI;
    uint16 public immutable netuid;

    // ── Owner config ───────────────────────────────────────────────

    address public owner;
    address public ownerTreasury;
    uint16  public ownerCutBps;           // basis points, default 1000 = 10%

    uint16 constant MAX_OWNER_CUT_BPS = 2000;  // 20% cap

    // ── Staking target ─────────────────────────────────────────────

    bytes32 public validatorHotkey;       // Hotkey for alpha staking on Bittensor

    // ── Remote addresses (Bittensor EVM) ───────────────────────────

    address public remoteSwapAndStake;    // TaoFi SwapAndStake on Bittensor EVM
    address public remoteWTAO;            // WTAO on Bittensor EVM
    address public remoteUSDC;            // Bridged USDC on Bittensor EVM

    // ── Tracking ───────────────────────────────────────────────────

    uint256 public totalRevenue;          // Total USDC processed (6 decimals)
    uint256 public totalTreasury;         // Total USDC sent to treasury
    uint256 public totalBridged;          // Total USDC sent via TaoFi bridge
    uint256 public bridgeCount;           // Number of bridge() calls

    // ── Thresholds ─────────────────────────────────────────────────

    uint256 public minBridgeAmount;       // Min USDC to bridge (avoids dust txs)

    // ── Reentrancy guard ───────────────────────────────────────────

    uint256 private _locked;

    // ── Emergency timelock ─────────────────────────────────────────

    uint256 public emergencyAnnouncedAt;
    uint256 constant EMERGENCY_TIMELOCK = 7 days;

    // ── Events ─────────────────────────────────────────────────────

    event Bridged(
        address indexed caller,
        uint256 usdcAmount,
        uint256 treasuryCut,
        uint256 bridgedAmount
    );

    event OwnerCutUpdated(uint16 oldBps, uint16 newBps);
    event TreasuryUpdated(address oldTreasury, address newTreasury);
    event ValidatorHotkeyUpdated(bytes32 oldHotkey, bytes32 newHotkey);
    event MinBridgeAmountUpdated(uint256 oldAmount, uint256 newAmount);
    event RemoteConfigUpdated(
        address swapAndStake,
        address wtao,
        address usdc
    );
    event OwnershipTransferred(address oldOwner, address newOwner);
    event EmergencyAnnounced(uint256 timestamp);
    event EmergencyWithdrawUSDC(address indexed to, uint256 amount);
    event EmergencyCancelled();

    // ── Modifiers ──────────────────────────────────────────────────

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier nonReentrant() {
        require(_locked == 0, "Reentrant");
        _locked = 1;
        _;
        _locked = 0;
    }

    // ── Constructor ────────────────────────────────────────────────

    constructor(
        address _usdc,
        address _taofi,
        uint16  _netuid,
        address _ownerTreasury,
        uint16  _ownerCutBps,
        bytes32 _validatorHotkey,
        uint256 _minBridgeAmount
    ) {
        require(_usdc != address(0), "Zero USDC");
        require(_taofi != address(0), "Zero TaoFi");
        require(_ownerTreasury != address(0), "Zero treasury");
        require(_ownerCutBps <= MAX_OWNER_CUT_BPS, "Cut too high");
        require(_validatorHotkey != bytes32(0), "Zero hotkey");

        USDC = _usdc;
        TAOFI = ITaoFiBridge(_taofi);
        netuid = _netuid;
        owner = msg.sender;
        ownerTreasury = _ownerTreasury;
        ownerCutBps = _ownerCutBps;
        validatorHotkey = _validatorHotkey;
        minBridgeAmount = _minBridgeAmount;
    }

    // ── Bridge (public, anyone can call) ───────────────────────────

    /// @notice Bridge accumulated USDC to Bittensor EVM via TaoFi.
    ///         Splits the owner treasury cut, then bridges the remainder.
    ///         The USDC is swapped to TAO and staked as alpha — PERMANENTLY.
    ///
    ///         The caller pays Hyperlane messaging fees in ETH (msg.value).
    ///         Anyone can call this — keeper bot, validator, or user.
    ///
    /// @param bridgeCallData  Encoded TaoFi bridge parameters (SwapParams +
    ///                        RemoteCallsParams). Built off-chain by the keeper
    ///                        with current slippage params and TaoFi SDK.
    function bridge(bytes calldata bridgeCallData) external payable nonReentrant {
        uint256 balance = _usdcBalance();
        require(balance >= minBridgeAmount, "Below min bridge amount");

        // Split: owner cut to treasury
        uint256 treasuryCut = (balance * ownerCutBps) / 10000;
        uint256 bridgeAmount = balance - treasuryCut;

        // Send treasury cut
        if (treasuryCut > 0) {
            _usdcTransfer(ownerTreasury, treasuryCut);
        }

        // Approve TaoFi to spend our USDC for bridging
        _usdcApprove(address(TAOFI), bridgeAmount);

        // Decode and execute TaoFi bridge call
        // The bridgeCallData is ABI-encoded (SwapParams, RemoteCallsParams, uint256)
        // built off-chain by the keeper with current market slippage params.
        (
            ITaoFiBridge.SwapParams memory swapParams,
            ITaoFiBridge.RemoteCallsParams memory remoteParams,
            uint256 bridgeCost
        ) = abi.decode(
            bridgeCallData,
            (ITaoFiBridge.SwapParams, ITaoFiBridge.RemoteCallsParams, uint256)
        );

        // Override the input amount to match the actual bridge amount
        // (keeper may have estimated differently)
        swapParams.fromToken = USDC;
        swapParams.fromAmount = bridgeAmount;

        TAOFI.lifiSwapBridgeAndCall{value: msg.value}(
            swapParams,
            remoteParams,
            bridgeCost
        );

        // Track
        totalRevenue += balance;
        totalTreasury += treasuryCut;
        totalBridged += bridgeAmount;
        bridgeCount++;

        emit Bridged(msg.sender, balance, treasuryCut, bridgeAmount);
    }

    // ── Read functions ─────────────────────────────────────────────

    /// @notice Get the USDC balance pending bridging.
    function pendingBridgeAmount() external view returns (uint256) {
        return _usdcBalance();
    }

    // ── Owner admin ────────────────────────────────────────────────

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

    /// @notice Update the validator hotkey for alpha staking.
    function setValidatorHotkey(bytes32 newHotkey) external onlyOwner {
        require(newHotkey != bytes32(0), "Zero hotkey");
        bytes32 oldHotkey = validatorHotkey;
        validatorHotkey = newHotkey;
        emit ValidatorHotkeyUpdated(oldHotkey, newHotkey);
    }

    /// @notice Update the minimum USDC amount for bridging.
    function setMinBridgeAmount(uint256 newAmount) external onlyOwner {
        uint256 oldAmount = minBridgeAmount;
        minBridgeAmount = newAmount;
        emit MinBridgeAmountUpdated(oldAmount, newAmount);
    }

    /// @notice Update remote Bittensor EVM addresses for bridge params.
    function setRemoteConfig(
        address _swapAndStake,
        address _wtao,
        address _usdc
    ) external onlyOwner {
        remoteSwapAndStake = _swapAndStake;
        remoteWTAO = _wtao;
        remoteUSDC = _usdc;
        emit RemoteConfigUpdated(_swapAndStake, _wtao, _usdc);
    }

    /// @notice Transfer ownership.
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Zero owner");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }

    // ── Emergency withdraw (unbridged USDC only, 7-day timelock) ──
    //
    // NOTE: There is NO unstake function for bridged alpha.
    // Once bridged, alpha is permanently locked. This is by design —
    // the permanent sink amplifies subnet emissions.

    /// @notice Announce intent to emergency withdraw. Starts 7-day timelock.
    function announceEmergencyWithdraw() external onlyOwner {
        emergencyAnnouncedAt = block.timestamp;
        emit EmergencyAnnounced(block.timestamp);
    }

    /// @notice Execute emergency withdraw after timelock expires.
    ///         Can only withdraw unbridged USDC still on Base.
    function executeEmergencyWithdraw(address to, uint256 amount) external onlyOwner {
        require(emergencyAnnouncedAt > 0, "Not announced");
        require(
            block.timestamp >= emergencyAnnouncedAt + EMERGENCY_TIMELOCK,
            "Timelock not expired"
        );
        require(to != address(0), "Zero recipient");
        require(amount <= _usdcBalance(), "Exceeds balance");

        _usdcTransfer(to, amount);
        emergencyAnnouncedAt = 0;  // reset after execution

        emit EmergencyWithdrawUSDC(to, amount);
    }

    /// @notice Cancel a pending emergency withdraw announcement.
    function cancelEmergencyWithdraw() external onlyOwner {
        emergencyAnnouncedAt = 0;
        emit EmergencyCancelled();
    }

    // ── Internal ERC-20 helpers (low-level, no OZ) ─────────────────

    function _usdcBalance() internal view returns (uint256) {
        (bool ok, bytes memory ret) = USDC.staticcall(
            abi.encodeWithSignature("balanceOf(address)", address(this))
        );
        require(ok && ret.length >= 32, "balanceOf failed");
        return abi.decode(ret, (uint256));
    }

    function _usdcTransfer(address to, uint256 amount) internal {
        (bool ok, bytes memory ret) = USDC.call(
            abi.encodeWithSignature("transfer(address,uint256)", to, amount)
        );
        require(
            ok && (ret.length == 0 || abi.decode(ret, (bool))),
            "USDC transfer failed"
        );
    }

    function _usdcApprove(address spender, uint256 amount) internal {
        (bool ok, bytes memory ret) = USDC.call(
            abi.encodeWithSignature("approve(address,uint256)", spender, amount)
        );
        require(
            ok && (ret.length == 0 || abi.decode(ret, (bool))),
            "USDC approve failed"
        );
    }

    // Accept ETH (for Hyperlane fee refunds)
    receive() external payable {}
}
