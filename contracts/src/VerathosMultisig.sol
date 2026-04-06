// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title VerathosMultisig — lightweight N-of-M multisig for contract upgrades
/// @notice Collects off-chain EIP-191 signatures and executes arbitrary calls
///         when the threshold is met. Designed for UUPS proxy upgrade authorization
///         and contract admin operations.
///
///         Signatures must be sorted by signer address (ascending) to prevent
///         duplicates without gas-expensive on-chain mapping lookups.
///
///         Self-management (add/remove signers, change threshold) can only be
///         performed through execute() itself — requiring threshold approval.
contract VerathosMultisig {

    // ── State ──────────────────────────────────────────────────────

    address[] public signers;
    mapping(address => bool) public isSigner;
    uint256 public threshold;
    uint256 public nonce;

    // ── Events ─────────────────────────────────────────────────────

    event Executed(address indexed target, uint256 value, bytes data, uint256 nonce);
    event SignerAdded(address indexed signer);
    event SignerRemoved(address indexed signer);
    event ThresholdChanged(uint256 oldThreshold, uint256 newThreshold);

    // ── Constructor ────────────────────────────────────────────────

    constructor(address[] memory _signers, uint256 _threshold) {
        require(_signers.length >= _threshold, "Threshold exceeds signer count");
        require(_threshold > 0, "Zero threshold");

        for (uint256 i = 0; i < _signers.length; i++) {
            require(_signers[i] != address(0), "Zero signer");
            require(!isSigner[_signers[i]], "Duplicate signer");
            isSigner[_signers[i]] = true;
            signers.push(_signers[i]);
        }
        threshold = _threshold;
    }

    // ── Execute with off-chain signatures ──────────────────────────

    /// @notice Execute an arbitrary call after verifying threshold signatures.
    /// @dev    Digest = EIP-191 personal sign of:
    ///         keccak256(chainId, address(this), nonce, target, value, data)
    ///         Signatures MUST be sorted by signer address (ascending).
    /// @param target     Contract to call (e.g. a UUPS proxy).
    /// @param value      Native currency to forward (0 for non-payable calls).
    /// @param data       Encoded function call (e.g. upgradeToAndCall calldata).
    /// @param signatures Array of 65-byte ECDSA signatures, sorted by signer address.
    function execute(
        address target,
        uint256 value,
        bytes calldata data,
        bytes[] calldata signatures
    ) external payable returns (bytes memory) {
        require(signatures.length >= threshold, "Below threshold");

        bytes32 digest = _digest(target, value, data);
        address lastSigner = address(0);

        for (uint256 i = 0; i < signatures.length; i++) {
            address recovered = _recover(digest, signatures[i]);
            require(isSigner[recovered], "Not a signer");
            require(recovered > lastSigner, "Signatures not sorted or duplicate");
            lastSigner = recovered;
        }

        uint256 currentNonce = nonce;
        nonce = currentNonce + 1;

        (bool success, bytes memory result) = target.call{value: value}(data);
        require(success, "Execution failed");

        emit Executed(target, value, data, currentNonce);
        return result;
    }

    // ── Self-management (only callable through execute()) ──────────

    /// @notice Add a new signer. Must be called through execute().
    function addSigner(address signer) external {
        require(msg.sender == address(this), "Only via execute()");
        require(signer != address(0), "Zero signer");
        require(!isSigner[signer], "Already a signer");

        isSigner[signer] = true;
        signers.push(signer);

        emit SignerAdded(signer);
    }

    /// @notice Remove a signer. Must be called through execute().
    ///         Cannot remove if it would break the threshold.
    function removeSigner(address signer) external {
        require(msg.sender == address(this), "Only via execute()");
        require(isSigner[signer], "Not a signer");
        require(signers.length - 1 >= threshold, "Would break threshold");

        isSigner[signer] = false;

        // Swap-and-pop removal
        for (uint256 i = 0; i < signers.length; i++) {
            if (signers[i] == signer) {
                signers[i] = signers[signers.length - 1];
                signers.pop();
                break;
            }
        }

        emit SignerRemoved(signer);
    }

    /// @notice Change the signature threshold. Must be called through execute().
    function changeThreshold(uint256 newThreshold) external {
        require(msg.sender == address(this), "Only via execute()");
        require(newThreshold > 0, "Zero threshold");
        require(newThreshold <= signers.length, "Exceeds signer count");

        uint256 old = threshold;
        threshold = newThreshold;

        emit ThresholdChanged(old, newThreshold);
    }

    // ── View functions ─────────────────────────────────────────────

    /// @notice Get the full list of current signers.
    function getSigners() external view returns (address[] memory) {
        return signers;
    }

    /// @notice Get the number of signers.
    function getSignerCount() external view returns (uint256) {
        return signers.length;
    }

    /// @notice Compute the digest that signers must sign for a given call.
    ///         Use this off-chain to build the message each signer signs.
    function getDigest(
        address target,
        uint256 value,
        bytes calldata data
    ) external view returns (bytes32) {
        return _digest(target, value, data);
    }

    // ── Internal ───────────────────────────────────────────────────

    function _digest(
        address target,
        uint256 value,
        bytes calldata data
    ) internal view returns (bytes32) {
        bytes32 structHash = keccak256(
            abi.encodePacked(block.chainid, address(this), nonce, target, value, data)
        );
        // EIP-191 personal sign prefix
        return keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", structHash)
        );
    }

    function _recover(
        bytes32 digest,
        bytes calldata sig
    ) internal pure returns (address) {
        require(sig.length == 65, "Invalid signature length");

        bytes32 r;
        bytes32 s;
        uint8 v;

        assembly {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }

        if (v < 27) v += 27;
        require(v == 27 || v == 28, "Invalid v value");
        // EIP-2: reject signatures with s in the upper half of the curve
        require(
            uint256(s) <= 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0,
            "Invalid s value (malleability)"
        );

        address recovered = ecrecover(digest, v, r, s);
        require(recovered != address(0), "Invalid signature");
        return recovered;
    }

    // Accept native currency (for forwarding value in execute calls)
    receive() external payable {}
}
