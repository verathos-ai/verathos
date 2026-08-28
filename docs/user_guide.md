# Verathos User Guide

Use a wallet account for prepaid inference and API keys, or use x402 for
accountless USDC pay-per-request.

## Wallet account

Open [verathos.ai/account](https://verathos.ai/account) and sign in with a
Bittensor SS58 or Base EVM wallet. Signing proves control of the public address;
it does not submit a transaction or expose a private key.

Wallet challenges are site-bound, expire after five minutes, and are consumed
once. The browser receives a Secure, HttpOnly session cookie. Sessions expire
after 30 days and rotate when you sign in again. Anonymous chat sessions remain
separate from account identity.

An account can link one SS58 conviction wallet and one EVM payment wallet.
Addresses are unique across accounts. Link a missing wallet under **Account →
Security** with another fresh signature.

## Balances and usage

All account balances use USD microcredits, where `1 credit = $1` of inference.
The Account overview separates:

- paid credit available;
- today's conviction entitlement, spent amount, and remaining amount; and
- total available inference credit.

Paid credits never expire, but are non-refundable and non-transferable. Usage
reserves the request's maximum cost before dispatch, settles verified token
usage afterward, and releases unused headroom. Conviction allowance is spent
first; a single request may split between conviction and paid credit. The Usage
page shows both charged amounts.

## Add credits

Open **Account → Billing**, choose the asset and amount, and select the Add
button. The connected wallet opens immediately. Cancelling before the wallet
sends creates no payment record.

### TAO

- Network: Bittensor
- Minimum: `0.01 TAO`
- Sender: the linked SS58 wallet
- Receiver: the service's fixed TAO receiver

Credit is issued only after the transfer is finalized and a fresh live TAO/USD
price is available. The finalized price is applied once. If the price source is
unavailable, no credit is issued until the same transaction can be verified
with a fresh price. No configured fallback price is used to issue credit.
If the browser closes after a valid transfer, the receiver reconciliation job
recognizes the linked sender and credits it automatically. Billing history
contains only successful, credited payments.

### USDC

- Network: Base
- Minimum: `$0.10 USDC`
- Sender: the linked EVM wallet
- Settlement: x402 `upto` authorization through the configured facilitator

USDC is credited 1:1. The first payment may require a USDC approval for Permit2
before the wallet asks for the payment authorization. The backend binds that
authorization to the selected amount, linked wallet, network, asset, receiver,
and account billing resource. It verifies and settles through the facilitator,
then atomically records the payment and credit. An authorization can be
credited only once, and unsuccessful attempts do not appear in Billing history.

## Conviction allowance

The linked SS58 wallet qualifies only through a perpetual SN96 lock targeting
the current subnet-owner hotkey. At a finalized snapshot, let `A` be eligible
locked alpha:

```text
if A < 10:
    daily allowance = $0
else if A < 100:
    daily allowance = $0.01 × A
else:
    daily allowance = min($25, $1 × sqrt(A / 100))
```

| Locked alpha | Daily allowance |
|---:|---:|
| Below 10 | $0 |
| 10 | $0.10 |
| 50 | $0.50 |
| 100 | $1 |
| 400 | $2 |
| 2,500 | $5 |
| 10,000 | $10 |
| 62,500+ | $25 cap |

Allowance resets at `00:00 UTC`, never rolls over, and cannot be transferred or
withdrawn. Lock state refreshes periodically and when the Account page opens.
During a chain RPC outage, a recent finalized snapshot may be used briefly;
after it becomes stale, conviction spending pauses while paid credit and x402
continue.

## API keys

Create keys under **Account → API keys**. Key creation requires a fresh linked
wallet signature. Name each key so it can be identified later.

The full `vrt_sk_...` value is displayed once. Store it in a secret manager or
environment variable; do not put it in source control or browser storage. Only
a secure hash and non-secret prefix are retained by the service. An account can
have at most ten active keys, and revocation takes effect immediately.

Use the key with the OpenAI-compatible API:

```bash
curl https://api.verathos.ai/v1/chat/completions \
  -H "Authorization: Bearer vrt_sk_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="vrt_sk_YOUR_KEY",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain zero-knowledge proofs."}],
)
print(response.choices[0].message.content)
```

## Model selection and tools

Use `auto` to select the highest-scored healthy route, or request a qualified
model ID returned by `GET /v1/models`. Tool-capable model entries advertise
`tools`, `tool_choice`, and `parallel_tool_calls` in `supported_parameters`.
Raw API clients execute their own tools and send tool results back in the next
chat-completion request.

## x402 pay-per-request

x402 pays with USDC on Base without an account, prepaid balance, or API key.
Send an unauthenticated inference request, read the HTTP 402 payment
requirements, sign the requested Permit2 authorization with an x402-compatible
client, and retry with its payment header.

x402 is independent of account credit and settles to its configured receiver.
It cannot fund an account balance. Each successful request costs at least
`$0.01`; requests whose token price exceeds that minimum cost their calculated
usage price. The `upto` scheme authorizes a cap, and unused headroom is never
settled. A compatible authorization may be reused within its deadline for
multiple requests, subject to its signed cap.

The payer wallet must hold USDC on the requested Base network and complete any
one-time Permit2 approval required by its x402 client.

## TEE inference

TEE-qualified routes appear in `GET /v1/models` when available. Append the
advertised `:tee` qualifier for OpenAI-compatible TEE routing. See the
[API Reference](api.md#tee-encrypted-inference) for the encrypted TEE endpoints
and current availability.

## Account security

- Verify the domain shown by the wallet before signing.
- Keep API keys in a secret manager and revoke any exposed key immediately.
- Sign out from **Account → Security** to revoke the current browser session.
- Treat every paid-credit purchase as final.
- Use only the receiver and network displayed before approving a payment.
