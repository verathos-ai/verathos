// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title ModelRegistry — trusted store for model weight Merkle roots
/// @notice Stores ModelSpec data (per-layer weight Merkle roots) for each
///         approved model. Written once by a trusted registrant, read for
///         free by all miners and validators.
///
///         On-chain data is ~1-4 KB per model (32 bytes per layer root).
///         The computation is deterministic: same model + same quantization
///         = identical Merkle roots. Anyone can verify by running
///         verallm.registry.roots.compute_model_roots().
contract ModelRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable {

    // ── Types ─────────────────────────────────────────────────────

    struct ModelSpec {
        string    modelId;              // e.g. "Qwen/Qwen3-8B"
        bytes32   weightMerkleRoot;     // overall model commitment
        bytes32[] layerRoots;           // per-layer Merkle roots (32 bytes each)
        uint32    numLayers;
        uint32    hiddenDim;
        uint32    intermediateDim;
        uint32    numHeads;
        uint32    headDim;
        uint32    vocabSize;
        string    quantMode;            // "fp16" | "int8" | "int4"
        uint32    merkleChunkSize;      // elements per Merkle leaf (default 128)
        string    activation;           // "silu" | "gelu" | "relu"
        string    normType;             // "rmsnorm" | "layernorm"
        string    attentionType;        // "mha" | "gqa" | "mqa"
        uint32    numExperts;           // MoE expert count (0 = dense model)
        uint32    expertWNumCols;       // columns per expert W matrix (0 = use intermediateDim)
        bytes32   lmHeadRoot;           // Merkle root of lm_head (output projection) weights
        bytes32   embeddingRoot;        // Merkle root of embedding table weights (input binding)
        bytes32   weightFileHash;       // SHA256(safetensors bytes) — flat model identity for TEE verification
        bytes32   tokenizerHash;        // keccak/SHA256 of tokenizer.json + chat_template — anchor for validator-side drift detection
    }

    // ── State ─────────────────────────────────────────────────────

    /// netuid -> model_id -> ModelSpec
    mapping(uint16 => mapping(string => ModelSpec)) internal _specs;

    /// netuid -> model_id -> registered flag (for clean removal)
    mapping(uint16 => mapping(string => bool)) public isRegistered;

    /// netuid -> list of registered model IDs
    mapping(uint16 => string[]) internal _modelList;

    /// netuid -> address -> authorized to register/remove
    mapping(uint16 => mapping(address => bool)) public authorized;

    /// @dev Reserved storage gap for future upgrades.
    uint256[50] private __gap;

    // ── Events ────────────────────────────────────────────────────

    event ModelRegistered(uint16 indexed netuid, string modelId);
    event ModelRemoved(uint16 indexed netuid, string modelId);
    event AuthorizationChanged(uint16 indexed netuid, address registrant, bool status);

    // ── Modifiers ─────────────────────────────────────────────────

    modifier onlyAuthorized(uint16 netuid) {
        require(
            authorized[netuid][msg.sender] || msg.sender == owner(),
            "Not authorized"
        );
        _;
    }

    // ── Constructor & Initializer ────────────────────────────────

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() { _disableInitializers(); }

    function initialize() external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
    }

    // ── UUPS upgrade authorization ──────────────────────────────

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── Write functions (gas, signed transactions) ────────────────

    /// @notice Register or update a model's weight Merkle roots.
    function registerModel(uint16 netuid, ModelSpec calldata spec)
        external onlyAuthorized(netuid)
    {
        _specs[netuid][spec.modelId] = spec;

        if (!isRegistered[netuid][spec.modelId]) {
            isRegistered[netuid][spec.modelId] = true;
            _modelList[netuid].push(spec.modelId);
        }

        emit ModelRegistered(netuid, spec.modelId);
    }

    /// @notice Remove a model from the registry.
    function removeModel(uint16 netuid, string calldata modelId)
        external onlyAuthorized(netuid)
    {
        require(isRegistered[netuid][modelId], "Model not registered");

        delete _specs[netuid][modelId];
        isRegistered[netuid][modelId] = false;

        // Swap-and-pop to remove from the model list
        string[] storage list = _modelList[netuid];
        for (uint256 i = 0; i < list.length; i++) {
            if (keccak256(bytes(list[i])) == keccak256(bytes(modelId))) {
                list[i] = list[list.length - 1];
                list.pop();
                break;
            }
        }

        emit ModelRemoved(netuid, modelId);
    }

    /// @notice Authorize an address to register/remove models for a subnet.
    function authorize(uint16 netuid, address registrant) external onlyOwner {
        authorized[netuid][registrant] = true;
        emit AuthorizationChanged(netuid, registrant, true);
    }

    /// @notice Revoke authorization.
    function deauthorize(uint16 netuid, address registrant) external onlyOwner {
        authorized[netuid][registrant] = false;
        emit AuthorizationChanged(netuid, registrant, false);
    }

    // ── Read functions (free view calls) ──────────────────────────

    /// @notice Get a specific model's spec.
    function getModelSpec(uint16 netuid, string calldata modelId)
        external view returns (ModelSpec memory)
    {
        return _specs[netuid][modelId];
    }

    /// @notice List all registered model IDs for a subnet.
    function getModelList(uint16 netuid)
        external view returns (string[] memory)
    {
        return _modelList[netuid];
    }

    /// @notice Get the number of registered models for a subnet.
    function getModelCount(uint16 netuid) external view returns (uint256) {
        return _modelList[netuid].length;
    }
}
