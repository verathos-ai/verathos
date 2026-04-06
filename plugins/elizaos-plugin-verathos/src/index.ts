import { type Plugin, ModelType } from "@elizaos/core";
import {
  getApiKey,
  getAuthMode,
  getBaseURL,
  isX402Mode,
  setDiscoveredModel,
} from "./utils/config.js";
import { createSigner } from "./x402.js";
import {
  handleTextSmall,
  handleTextLarge,
  handleObjectSmall,
  handleObjectLarge,
  handleTokenizerEncode,
  handleTokenizerDecode,
} from "./models/index.js";

export const verathosPlugin: Plugin = {
  name: "verathos",
  description:
    "Verified LLM inference via Verathos on Bittensor — cryptographic proof " +
    "verification with optional x402 USDC pay-per-request",

  config: {
    VERATHOS_API_KEY: process.env.VERATHOS_API_KEY ?? null,
    VERATHOS_API_URL: process.env.VERATHOS_API_URL ?? null,
    VERATHOS_X402_PRIVATE_KEY: process.env.VERATHOS_X402_PRIVATE_KEY ?? null,
    VERATHOS_X402_TESTNET: process.env.VERATHOS_X402_TESTNET ?? null,
    VERATHOS_X402_CDP_ACCOUNT: process.env.VERATHOS_X402_CDP_ACCOUNT ?? null,
    CDP_API_KEY_ID: process.env.CDP_API_KEY_ID ?? null,
    CDP_API_KEY_SECRET: process.env.CDP_API_KEY_SECRET ?? null,
    CDP_WALLET_SECRET: process.env.CDP_WALLET_SECRET ?? null,
    VERATHOS_SMALL_MODEL: process.env.VERATHOS_SMALL_MODEL ?? null,
    VERATHOS_LARGE_MODEL: process.env.VERATHOS_LARGE_MODEL ?? null,
    VERATHOS_EMBEDDING_MODEL: process.env.VERATHOS_EMBEDDING_MODEL ?? null,
  },

  async init(_config, runtime) {
    const baseURL = getBaseURL(runtime);
    const authMode = getAuthMode(runtime);

    // Validate auth and log mode
    if (authMode === "x402_cdp") {
      const { description } = await createSigner(runtime);
      console.log(`[verathos] x402 CDP mode: ${description}`);
    } else if (authMode === "x402_raw") {
      const { description } = await createSigner(runtime);
      console.log(`[verathos] x402 raw key mode: ${description}`);
    } else {
      getApiKey(runtime); // validate key exists
      console.log("[verathos] API key mode");
    }

    // Connectivity check
    try {
      const fetchFn = runtime.fetch ?? globalThis.fetch;
      const headers: Record<string, string> = {};
      if (!isX402Mode(runtime)) {
        const apiKey = getApiKey(runtime);
        if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
      }

      const res = await fetchFn(`${baseURL}/models`, { headers });
      if (res.ok) {
        const data = (await res.json()) as { data?: { id: string }[] };
        const models = data.data?.map((m) => m.id) ?? [];
        // Cache first discovered model as default for "auto" resolution
        if (models.length > 0) {
          setDiscoveredModel(models[0]);
        }
        console.log(
          `[verathos] Connected to ${baseURL} — ${models.length} model(s) available` +
            (models.length > 0 ? `: ${models.join(", ")}` : ""),
        );
      } else {
        console.warn(
          `[verathos] API returned ${res.status} from ${baseURL}/models`,
        );
      }
    } catch (err) {
      console.warn(
        `[verathos] Could not reach ${baseURL} — ${err instanceof Error ? err.message : err}`,
      );
    }
  },

  models: {
    [ModelType.TEXT_SMALL]: handleTextSmall,
    [ModelType.TEXT_LARGE]: handleTextLarge,
    [ModelType.OBJECT_SMALL]: handleObjectSmall,
    [ModelType.OBJECT_LARGE]: handleObjectLarge,
    [ModelType.TEXT_TOKENIZER_ENCODE]: handleTokenizerEncode,
    [ModelType.TEXT_TOKENIZER_DECODE]: handleTokenizerDecode,
  },

  tests: [
    {
      name: "verathos-text-generation",
      tests: [
        {
          name: "should generate text with TEXT_SMALL",
          fn: async (runtime) => {
            const result = await runtime.useModel(ModelType.TEXT_SMALL, {
              prompt: "Say hello in one sentence.",
              maxTokens: 64,
            });
            if (typeof result !== "string" || result.length === 0) {
              throw new Error("TEXT_SMALL returned empty or non-string result");
            }
          },
        },
        {
          name: "should generate text with TEXT_LARGE",
          fn: async (runtime) => {
            const result = await runtime.useModel(ModelType.TEXT_LARGE, {
              prompt: "Explain Bittensor in one sentence.",
              maxTokens: 128,
            });
            if (typeof result !== "string" || result.length === 0) {
              throw new Error("TEXT_LARGE returned empty or non-string result");
            }
          },
        },
      ],
    },
  ],
};

export default verathosPlugin;
