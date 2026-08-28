# Quickstart

Go from a wallet sign-in to your first account-funded API call.

## 1. Sign in with a wallet

Open [verathos.ai/account](https://verathos.ai/account) and sign a one-time
challenge with either a Bittensor SS58 wallet or a Base EVM wallet. The
challenge is bound to the site, expires after five minutes, and can be used
only once. No email or password is required.

Link the other wallet under **Account → Security** if you want both TAO and
USDC top-ups. One SS58 wallet and one EVM wallet can be linked to an account.

## 2. Add inference credits

Open **Account → Billing**, choose TAO or USDC, enter an amount, and select
**Add TAO** or **Add USDC**. The connected wallet opens immediately. Cancelling
the wallet request creates no payment record.

- Minimum TAO top-up: `0.01 TAO`
- Minimum USDC top-up: `$0.10 USDC` on Base
- `1 credit = $1` of inference
- Paid credits do not expire, but are non-refundable and non-transferable

TAO is credited after finalized-chain verification with a fresh market price.
USDC uses an x402 Permit2 authorization and is credited only after facilitator
settlement. Billing history contains only successful, credited payments. A
background TAO reconciliation pass credits a valid transfer from a linked
wallet if the browser closes before confirmation finishes.

Eligible perpetual SN96 locks directed to the current subnet-owner hotkey also
provide a daily inference allowance. The current formula and status are shown
on the Account page. Daily allowance is consumed before paid credit and does
not roll over.

## 3. Create an API key

Open **Account → API keys**, enter a name, and sign the fresh wallet challenge.
Copy the key immediately: the full value is shown once and only its secure hash
is stored. Each account can have up to ten active keys.

## 4. Make your first request

```bash
curl https://api.verathos.ai/v1/chat/completions \
  -H "Authorization: Bearer vrt_sk_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 256
  }'
```

The API is OpenAI-compatible. Use `auto` for score-weighted routing or select a
qualified model returned by `GET /v1/models`.

## 5. Use an OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="vrt_sk_YOUR_KEY",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain ZK proofs in one sentence."}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## Accountless alternative

[x402](https://docs.cdp.coinbase.com/x402) supports USDC pay-per-request without
an account or API key. See the [User Guide](user_guide.md#x402-pay-per-request)
for the payment flow.

## What's next?

- [User Guide](user_guide.md): accounts, billing, conviction, API keys, and x402
- [API Reference](api.md): HTTP endpoint documentation
- [What is Verathos?](intro.md): verification architecture
- [Verified Chat](https://verathos.ai/chat): browser inference
