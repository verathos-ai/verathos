import type { IAgentRuntime } from "@elizaos/core";

/** Resolve a setting from runtime > env > default. */
export function getSetting(
  runtime: IAgentRuntime,
  key: string,
  defaultValue?: string,
): string | undefined {
  const value = runtime.getSetting(key);
  if (value !== undefined && value !== null) return String(value);
  return process.env[key] ?? defaultValue;
}

// ── Auth mode detection ─────────────────────────────────────────

export type AuthMode = "api_key" | "x402_raw" | "x402_cdp";

/**
 * Detect which auth mode is configured.
 *
 * Priority: CDP wallet > raw private key > API key.
 * CDP is preferred over raw key when both are set (more secure).
 */
export function getAuthMode(runtime: IAgentRuntime): AuthMode {
  if (getSetting(runtime, "CDP_API_KEY_ID")) return "x402_cdp";
  if (getSetting(runtime, "VERATHOS_X402_PRIVATE_KEY")) return "x402_raw";
  return "api_key";
}

export function isX402Mode(runtime: IAgentRuntime): boolean {
  const mode = getAuthMode(runtime);
  return mode === "x402_raw" || mode === "x402_cdp";
}

/** Get API key — required unless x402 mode is active. */
export function getApiKey(runtime: IAgentRuntime): string | undefined {
  const key = getSetting(runtime, "VERATHOS_API_KEY");
  if (!key && !isX402Mode(runtime)) {
    throw new Error(
      "Set VERATHOS_API_KEY, VERATHOS_X402_PRIVATE_KEY, or CDP_API_KEY_ID",
    );
  }
  return key;
}

export function getBaseURL(runtime: IAgentRuntime): string {
  return getSetting(runtime, "VERATHOS_API_URL") ?? "https://api.verathos.ai/v1";
}

export function getX402PrivateKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, "VERATHOS_X402_PRIVATE_KEY");
}

export function isX402Testnet(runtime: IAgentRuntime): boolean {
  const val = getSetting(runtime, "VERATHOS_X402_TESTNET");
  return val === "true" || val === "1";
}

// ── CDP config ──────────────────────────────────────────────────

export function getCdpApiKeyId(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, "CDP_API_KEY_ID");
}

export function getCdpApiKeySecret(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, "CDP_API_KEY_SECRET");
}

export function getCdpWalletSecret(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, "CDP_WALLET_SECRET");
}

export function getCdpAccountName(runtime: IAgentRuntime): string {
  return getSetting(runtime, "VERATHOS_X402_CDP_ACCOUNT") ?? "verathos-agent";
}

// ── Model discovery ─────────────────────────────────────────────

/** Model discovered from /v1/models during init(). Used when no explicit model is set. */
let _discoveredModel: string | null = null;

export function setDiscoveredModel(modelId: string): void {
  _discoveredModel = modelId;
}

function resolveModel(explicit: string | undefined): string {
  if (explicit && explicit !== "auto") return explicit;
  if (_discoveredModel) return _discoveredModel;
  return "auto"; // last resort — proxy may reject this
}

// ── Model config ────────────────────────────────────────────────

export function getSmallModel(runtime: IAgentRuntime): string {
  return resolveModel(
    getSetting(runtime, "VERATHOS_SMALL_MODEL") ??
    getSetting(runtime, "SMALL_MODEL"),
  );
}

export function getLargeModel(runtime: IAgentRuntime): string {
  return resolveModel(
    getSetting(runtime, "VERATHOS_LARGE_MODEL") ??
    getSetting(runtime, "LARGE_MODEL"),
  );
}

export function getEmbeddingModel(runtime: IAgentRuntime): string {
  return resolveModel(
    getSetting(runtime, "VERATHOS_EMBEDDING_MODEL") ??
    getSetting(runtime, "EMBEDDING_MODEL"),
  );
}
