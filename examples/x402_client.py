#!/usr/bin/env python3
"""x402 pay-per-request client example.

Pay for LLM inference with USDC on Base — no API key or deposit needed.
The x402 SDK handles the full flow: send request → receive 402 with
payment requirements → sign USDC authorization → retransmit.

Prerequisites:
    pip install x402 eth-account httpx

    You need:
    - An EVM wallet private key
    - USDC on Base (mainnet) or Base Sepolia (testnet)
    - Small amount of ETH on Base for gas (~$0.001 per request)

Security note:
    This example uses a raw private key for simplicity. For production
    agents handling real funds, consider using a Coinbase CDP MPC wallet
    (pip install cdp-sdk) — the key is split between your agent and
    Coinbase, so a compromised server can't drain the wallet unilaterally.
    The x402 SDK supports any signer that implements the EthAccountSigner
    interface.

Usage:
    export EVM_PRIVATE_KEY=0x...
    python examples/x402_client.py --gateway https://proxy.example.com:8080

    # Testnet:
    python examples/x402_client.py --gateway https://api.verathos.ai --testnet
"""

import argparse
import asyncio
import os
import sys

from eth_account import Account


def main():
    parser = argparse.ArgumentParser(description="x402 USDC pay-per-request")
    parser.add_argument("--gateway", required=True, help="Gateway URL")
    parser.add_argument("--model", default=None, help="Model ID (default: auto)")
    parser.add_argument("--quant", default=None, help="Quantization (e.g. int4, fp16)")
    parser.add_argument("--prompt", default="What is quantum computing?",
                        help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--private-key", default=None,
                        help="EVM private key (default: $EVM_PRIVATE_KEY)")
    parser.add_argument("--testnet", action="store_true",
                        help="Use Base Sepolia testnet")
    args = parser.parse_args()

    pk = args.private_key or os.environ.get("EVM_PRIVATE_KEY")
    if not pk:
        print("ERROR: Provide --private-key or set EVM_PRIVATE_KEY")
        sys.exit(1)
    if not pk.startswith("0x"):
        pk = "0x" + pk

    account = Account.from_key(pk)
    print(f"Wallet: {account.address}")

    # Use server-side "auto" when no model specified (picks best available by score)
    if not args.model:
        args.model = "auto"
        # Verify proxy is reachable
        import httpx
        try:
            resp = httpx.get(f"{args.gateway.rstrip('/')}/v1/models", timeout=10)
            models = resp.json().get("models", [])
            if not models:
                print("ERROR: No models available")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Cannot reach gateway: {e}")
            sys.exit(1)

    # Set up x402 payment client
    from x402 import x402Client
    from x402.http.clients.httpx import x402HttpxClient
    from x402.http.x402_http_client import x402HTTPClient
    from x402.mechanisms.evm import EthAccountSigner
    from x402.mechanisms.evm.exact.register import register_exact_evm_client

    client = x402Client()
    register_exact_evm_client(client, EthAccountSigner(account))

    async def run():
        url = f"{args.gateway.rstrip('/')}/v1/chat/completions"
        # Append quant suffix if specified (e.g. "qwen3.5-9b:int4")
        model_id = f"{args.model}:{args.quant}" if args.quant else args.model
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
            "stream": False,
        }

        print(f"Sending request (max_tokens={args.max_tokens})...")
        async with x402HttpxClient(x402HTTPClient(client)) as http:
            response = await http.post(
                url, json=body,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"\nResponse:\n{content}")
            print(f"\nTokens: {usage.get('prompt_tokens', '?')} in, "
                  f"{usage.get('completion_tokens', '?')} out")
            print("Payment: USDC settled on-chain via x402")
        else:
            print(f"ERROR {response.status_code}: {response.text[:200]}")
            sys.exit(1)

    asyncio.run(run())


if __name__ == "__main__":
    main()
