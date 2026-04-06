#!/usr/bin/env python3
"""Show the EVM address and SS58 mirror for a Bittensor wallet.

Prints the derived EVM address and its SS58 mirror so you know where
to send TAO for gas funding before running a miner or validator.

Usage:
    python scripts/show_evm_info.py --wallet miner --hotkey default
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Show EVM address for a Bittensor wallet")
    parser.add_argument("--wallet", default="default", help="Bittensor wallet name")
    parser.add_argument("--hotkey", default="default", help="Bittensor hotkey name")
    parser.add_argument("--check-balance", action="store_true",
                        help="Also check EVM balance (requires RPC access)")
    parser.add_argument("--subtensor-network", default="finney",
                        choices=["finney", "test"],
                        help="Network for RPC URL resolution (default: finney)")
    args = parser.parse_args()

    from verallm.chain.wallet import derive_evm_address, h160_to_ss58_mirror

    # Read hotkey seed from keyfile
    hk_path = Path.home() / f".bittensor/wallets/{args.wallet}/hotkeys/{args.hotkey}"
    if not hk_path.exists():
        print(f"ERROR: Hotkey file not found: {hk_path}")
        print(f"  Create it with: btcli wallet new-hotkey --wallet {args.wallet} --hotkey {args.hotkey}")
        sys.exit(1)

    hk_data = json.loads(hk_path.read_text())
    hotkey_seed = bytes.fromhex(hk_data["secretSeed"].replace("0x", ""))

    evm_addr = derive_evm_address(hotkey_seed)
    mirror_ss58 = h160_to_ss58_mirror(evm_addr)

    print(f"Wallet:      {args.wallet}/{args.hotkey}")
    print(f"EVM address: {evm_addr}")
    print(f"SS58 mirror: {mirror_ss58}")

    if args.check_balance:
        try:
            from verallm.chain.config import ChainConfig
            from web3 import Web3

            rpc_url = ChainConfig.resolve_rpc_url(None, args.subtensor_network)
            if not rpc_url:
                rpc_url = "https://lite.chain.opentensor.ai"
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            balance_wei = w3.eth.get_balance(evm_addr)
            balance_tao = balance_wei / 1e18
            status = "sufficient" if balance_tao >= 0.05 else "INSUFFICIENT (need >= 0.05 for gas)"
            print(f"EVM balance: {balance_tao:.4f} TAO ({status})")
        except Exception as e:
            print(f"EVM balance: unable to check ({e})")

    print()
    print(f"To fund this EVM wallet for gas, transfer TAO to the SS58 mirror:")
    print(f"  btcli wallet transfer --dest {mirror_ss58} --amount 0.1")


if __name__ == "__main__":
    main()
