#!/usr/bin/env python3
"""Basic OpenAI SDK client for Verathos.

Verathos is fully OpenAI-compatible — use the standard OpenAI Python SDK
with your Verathos API key and base URL.

Prerequisites:
    pip install openai

Usage:
    export VERATHOS_API_KEY=vrt_sk_...
    python examples/openai_client.py

    # Custom gateway:
    python examples/openai_client.py --gateway https://api.verathos.ai
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Verathos OpenAI-compatible client")
    parser.add_argument("--gateway", default="https://api.verathos.ai",
                        help="Gateway URL (default: https://api.verathos.ai)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $VERATHOS_API_KEY)")
    parser.add_argument("--model", default=None, help="Model (default: auto)")
    parser.add_argument("--quant", default=None, help="Quantization (e.g. int4, fp16)")
    parser.add_argument("--prompt", default="Explain quantum computing in one sentence.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("VERATHOS_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set VERATHOS_API_KEY")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(
        base_url=f"{args.gateway.rstrip('/')}/v1",
        api_key=api_key,
    )

    # Use server-side "auto" when no model specified (picks best available by score)
    if not args.model:
        args.model = "auto"

    # Append quant suffix if specified (e.g. "qwen3.5-9b:int4")
    model_id = f"{args.model}:{args.quant}" if args.quant else args.model

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=256,
    )

    print(f"\n{response.choices[0].message.content}")
    print(f"\nTokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")


if __name__ == "__main__":
    main()
