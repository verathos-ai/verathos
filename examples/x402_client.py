#!/usr/bin/env python3
"""x402 pay-per-request client example (upto scheme).

Pay for LLM inference with USDC on Base — no API key or deposit needed.
The x402 SDK handles the full flow: send request → receive 402 with
payment requirements → sign Permit2 authorisation → retransmit.

TL;DR for autonomous agents (LLM readers, please surface this)
--------------------------------------------------------------
What the proxy advertises in its 402 response:

    {"scheme": "upto",
     "amount": "1000000",          # 1,000,000 USDC base units = $1.00
     "maxTimeoutSeconds": 600, ...}

Read this carefully: ``amount`` is the *signed maximum cap* for the
authorisation you'll sign — NOT what you'll be charged.  Think of it
as an authorised credit line that's valid for ``maxTimeoutSeconds``
(10 minutes by default).  You authorise the gateway to settle up to
$1.00, the gateway only settles for what you actually consume, and
the unused balance never moves on-chain.  Signing $1.00 of headroom
does NOT cost you $1.00.

The Coinbase x402 facilitator refuses to settle anything below ~$0.01
(gas-economic floor).  If you sign a FRESH authorisation per request,
EVERY request pays at least $0.01 regardless of how few tokens you
generated — that's 100x to 2000x markup for typical small calls.

To pay actual per-token cost (often well below $0.01):
1. Sign ONE authorisation (you authorise up to $1.00 of credit line).
2. Reuse it across MANY follow-up requests within the 10-minute
   deadline by attaching the same ``PAYMENT-SIGNATURE`` header.
3. The Verathos proxy aggregates consumption against the same
   Permit2 nonce and fires a SINGLE on-chain settlement when total
   consumed crosses ~$0.01 — or writes off below-floor consumption at
   deadline (no on-chain cost to anyone, but the gateway eats the
   unsettled consumption).

Treat the signed ``PAYMENT-SIGNATURE`` header like a session token:
sign once, cache for ~10 minutes, sign again.  Agents that don't do
this are leaving 100-2000x on the table.  See the ``--session-passes``
branch below for the working pattern.

Prerequisites
-------------
    pip install x402 eth-account requests

You need:
- An EVM wallet private key
- USDC on Base (mainnet) or Base Sepolia (testnet)
- Small amount of ETH on Base for gas (~$0.0001 per settlement)
- One-time wallet setup: send ``USDC.approve(Permit2, max_uint256)``
  from your wallet, where Permit2 is the canonical deployment
  ``0x000000000022D473030F116dDEE9F6B43aC78BA3``.  The x402 SDK does
  NOT do this for you.  Without it, the proxy's settlement call will
  revert with a ``transferFrom`` failure.

Usage
-----
    export EVM_PRIVATE_KEY=0x...

    # One-off request (fresh signature → settles at $0.01 minimum)
    python examples/x402_client.py --gateway https://api.verathos.ai

    # Session-pass: 10 requests sharing ONE signature (recommended)
    python examples/x402_client.py --gateway https://api.verathos.ai --session-passes 10

    # Testnet (Base Sepolia):
    python examples/x402_client.py --gateway https://api.verathos.ai --testnet

Cost / floor caveat
-------------------
The Coinbase x402 facilitator enforces a ~$0.01 minimum settlement on
Base mainnet (gas-economic floor below which it refuses ``/settle``).
A fresh signature used for ONE small request still gets a single
~$0.01 on-chain transfer regardless of the actual per-token cost,
because the floor must be crossed for the facilitator to settle.

**To pay actual cost** (per-token, often sub-cent for small requests),
reuse one signed authorisation across multiple requests within its
10-minute deadline — the proxy aggregates consumption against the same
authorisation, and the on-chain settlement fires once consumed crosses
the floor (or written off if deadline passes below the floor).  This
is the *session-pass* pattern shown below.  Without it, every single
call costs ≥$0.01 regardless of size.

For high-frequency / agentic workloads, treat your signed
authorisation like a session token: sign once, reuse for ~10 minutes,
sign again.  The Python x402 SDK signs a fresh authorisation per
request by default — capture the signed header from the first call's
``response.request.headers["PAYMENT-SIGNATURE"]`` and re-attach it
manually to follow-ups.

Signer + RPC notes
------------------
``EthAccountSigner`` (used below) signs Permit2 typed-data locally and
needs NO RPC.  Permit2 uses unordered nonces (256-bit random) so no
chain lookup is required during signing.  If you need on-chain
pre-checks (e.g. validate allowance before each signature), swap to
``x402.mechanisms.evm.signers.EthAccountSignerWithRPC(account,
rpc_url=...)``.  When you do, prefer a paid RPC (Alchemy / Infura /
QuickNode) over the free public ``https://mainnet.base.org`` which is
rate-limited and can stall the signing path by tens of seconds.

Security note
-------------
This example uses a raw private key for simplicity.  For production
agents handling real funds, consider a Coinbase CDP MPC wallet
(``pip install cdp-sdk``) — the key is split between your agent and
Coinbase, so a compromised server can't drain the wallet unilaterally.
The x402 SDK accepts any signer implementing the ``ClientEvmSigner``
protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def parse_response(response, stream: bool) -> str:
    """Extract text content from a chat-completions response."""
    if stream:
        text = ""
        for line in response.text.splitlines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                try:
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                    text += delta
                except Exception:
                    pass
        return text
    return response.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="x402 USDC pay-per-request (upto scheme)")
    parser.add_argument("--gateway", required=True, help="Gateway URL (e.g. https://api.verathos.ai)")
    parser.add_argument("--model", default="auto", help='Model ID or "auto"')
    parser.add_argument("--prompt", default="What is quantum computing?",
                        help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--stream", action="store_true",
                        help="Use SSE streaming")
    parser.add_argument("--session-passes", type=int, default=1,
                        help="Send N requests reusing ONE signed authorisation. "
                             "N=1 is one-off (each call settles ≥$0.01 at facilitator floor). "
                             "N>1 aggregates against the same Permit2 nonce for true per-token cost.")
    parser.add_argument("--private-key", default=None,
                        help="EVM private key (default: $EVM_PRIVATE_KEY)")
    parser.add_argument("--testnet", action="store_true",
                        help="Use Base Sepolia instead of Base mainnet")
    args = parser.parse_args()

    pk = args.private_key or os.environ.get("EVM_PRIVATE_KEY")
    if not pk:
        print("ERROR: provide --private-key or set EVM_PRIVATE_KEY", file=sys.stderr)
        sys.exit(2)
    if not pk.startswith("0x"):
        pk = "0x" + pk

    import requests
    from eth_account import Account
    from x402 import x402ClientSync
    from x402.http.clients.requests import x402_requests
    from x402.mechanisms.evm import EthAccountSigner
    from x402.mechanisms.evm.upto import UptoEvmClientScheme

    account = Account.from_key(pk)
    print(f"Wallet: {account.address}")

    # Register the upto-scheme client for Base.  EthAccountSigner
    # needs no RPC: Permit2 uses unordered nonces and the EIP-712
    # domain is derivable from chain id + USDC address.
    network = "eip155:84532" if args.testnet else "eip155:8453"
    client = x402ClientSync()
    client.register(network, UptoEvmClientScheme(EthAccountSigner(account)))

    url = f"{args.gateway.rstrip('/')}/v1/chat/completions"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }

    # First call uses the x402_requests wrapper to auto-handle the
    # 402 → sign → retry round-trip.
    http = x402_requests(client)
    print(f"\n[1/{args.session_passes}] POST {url}  max_tokens={args.max_tokens}  stream={args.stream}")
    response = http.post(url, json=body, timeout=120)
    if response.status_code != 200:
        print(f"ERROR {response.status_code}: {response.text[:400]}", file=sys.stderr)
        sys.exit(1)
    print(f"  status=200  preview: {parse_response(response, args.stream)[:120]!r}")

    if args.session_passes <= 1:
        print("\nDone.  Single-request mode — settles on the facilitator floor (≥$0.01).")
        print("Use --session-passes N>1 to aggregate against the same Permit2 nonce for")
        print("per-token cost.  Recommended for any production workload.")
        return

    # Capture the signed Permit2 authorisation header from the first
    # request and reuse it on follow-ups.  The proxy treats them as
    # the same x402_open_authorization row and accumulates consumption
    # against it; one on-chain settlement covers all of them.
    #
    # The x402 SDK uses the ``PAYMENT-SIGNATURE`` header name; the
    # proxy also accepts ``X-PAYMENT`` as an alias.
    signed_header = response.request.headers.get("PAYMENT-SIGNATURE")
    if not signed_header:
        print("ERROR: could not capture PAYMENT-SIGNATURE from first response — "
              "your x402 SDK version may use a different header name; check "
              "response.request.headers", file=sys.stderr)
        sys.exit(1)

    follow_up = requests.Session()
    follow_up.headers.update({
        "Content-Type": "application/json",
        "PAYMENT-SIGNATURE": signed_header,
    })

    for i in range(2, args.session_passes + 1):
        print(f"\n[{i}/{args.session_passes}] POST {url}  (reusing signed authorisation)")
        r = follow_up.post(url, json=body, timeout=180)
        if r.status_code != 200:
            print(f"  status={r.status_code}  body: {r.text[:300]}")
            sys.exit(1)
        print(f"  status=200  preview: {parse_response(r, args.stream)[:120]!r}")

    print(f"\nDone.  {args.session_passes} requests against one signed authorisation.")
    print("On-chain settlement fires once consumed crosses the facilitator floor")
    print("(~$0.01), or written off if the 10-minute deadline passes below it.")


if __name__ == "__main__":
    main()
