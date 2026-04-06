// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Re-export so Foundry compiles the proxy and produces a deployable artifact.
// Used by scripts/deploy_proxies.py to deploy UUPS proxy instances via web3.py.
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
