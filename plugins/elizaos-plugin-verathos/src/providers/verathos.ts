import { createOpenAI, type OpenAIProvider } from "@ai-sdk/openai";
import type { IAgentRuntime } from "@elizaos/core";
import {
  getApiKey,
  getBaseURL,
  isX402Mode,
  isX402Testnet,
} from "../utils/config.js";
import { createSigner, createX402Fetch } from "../x402.js";

/**
 * Per-runtime cache for x402 fetch wrappers.
 * Avoids re-creating signer + fetch wrapper on every model call,
 * while staying safe for multi-agent scenarios.
 */
const _x402Cache = new WeakMap<
  IAgentRuntime,
  Promise<typeof globalThis.fetch>
>();

function getOrCreateX402Fetch(
  runtime: IAgentRuntime,
): Promise<typeof globalThis.fetch> {
  let cached = _x402Cache.get(runtime);
  if (!cached) {
    cached = (async () => {
      const { account } = await createSigner(runtime);
      const testnet = isX402Testnet(runtime);
      const baseFetch = runtime.fetch ?? globalThis.fetch;
      return createX402Fetch(account, testnet, baseFetch);
    })();
    _x402Cache.set(runtime, cached);
  }
  return cached;
}

/**
 * Create an OpenAI-compatible client pointed at the Verathos API.
 *
 * Three auth modes:
 * - **API key**: `VERATHOS_API_KEY` sent as Bearer token
 * - **x402 raw key**: `VERATHOS_X402_PRIVATE_KEY` — hot wallet signs USDC payments
 * - **x402 CDP**: `CDP_API_KEY_ID` + secrets — Coinbase MPC wallet (production)
 */
export async function createVerathosClient(
  runtime: IAgentRuntime,
): Promise<OpenAIProvider> {
  const baseURL = getBaseURL(runtime);

  if (isX402Mode(runtime)) {
    const x402Fetch = await getOrCreateX402Fetch(runtime);

    return createOpenAI({
      apiKey: "x402", // placeholder — x402 fetch wrapper handles auth
      baseURL,
      fetch: x402Fetch,
    });
  }

  // API key mode
  const apiKey = getApiKey(runtime);
  return createOpenAI({
    apiKey: apiKey!,
    baseURL,
    fetch: runtime.fetch,
  });
}
