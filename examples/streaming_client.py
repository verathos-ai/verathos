#!/usr/bin/env python3
"""Streaming client for Verathos — token-by-token output via SSE.

Uses the OpenAI SDK's streaming mode for real-time token delivery.

Prerequisites:
    pip install openai

Usage:
    export VERATHOS_API_KEY=vrt_sk_...
    python examples/streaming_client.py

    # Custom gateway + model:
    python examples/streaming_client.py --gateway https://api.verathos.ai --model qwen3.5-9b
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Verathos streaming client")
    parser.add_argument("--gateway", default="https://api.verathos.ai",
                        help="Gateway URL (default: https://api.verathos.ai)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $VERATHOS_API_KEY)")
    parser.add_argument("--model", default=None, help="Model (default: auto)")
    parser.add_argument("--quant", default=None, help="Quantization (e.g. int4, fp16)")
    parser.add_argument("--prompt", default="Write a short poem about cryptographic proofs.")
    parser.add_argument("--max-tokens", type=int, default=256)
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

    print(f"Prompt: {args.prompt}\n")

    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()
