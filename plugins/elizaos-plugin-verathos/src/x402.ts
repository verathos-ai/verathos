/**
 * x402 USDC pay-per-request — signer abstraction + fetch wrapper.
 *
 * Wraps `fetch` to automatically handle HTTP 402 responses using the
 * official x402 SDK (`@x402/client`), which generates the correct
 * EIP-712 typed data signatures that the Coinbase facilitator expects.
 *
 * Supports two signer backends:
 *   - **Raw private key** (dev/testnet): `VERATHOS_X402_PRIVATE_KEY`
 *   - **Coinbase CDP MPC wallet** (production): `CDP_API_KEY_ID` + secrets
 *     Keys live in AWS Nitro Enclave TEE — never exposed to the application.
 *
 * Both signers produce a viem-compatible Account, so the x402 SDK
 * payment logic is identical regardless of backend.
 */

import type { IAgentRuntime } from "@elizaos/core";
import {
  getAuthMode,
  getX402PrivateKey,
  getCdpApiKeyId,
  getCdpApiKeySecret,
  getCdpWalletSecret,
  getCdpAccountName,
  isX402Testnet,
} from "./utils/config.js";

/** Minimal viem-compatible account interface. */
interface ViemAccount {
  address: string;
  signMessage(args: { message: string }): Promise<string>;
  signTypedData?(args: {
    domain: Record<string, unknown>;
    types: Record<string, unknown[]>;
    primaryType: string;
    message: Record<string, unknown>;
  }): Promise<string>;
}

// ── Signer creation ─────────────────────────────────────────────

/**
 * Create a viem account from a raw private key (dev/testnet).
 *
 * The private key is held in process memory. Use a dedicated hot wallet
 * funded with only what you're willing to spend.
 */
async function createRawKeyAccount(privateKey: string): Promise<ViemAccount> {
  const { privateKeyToAccount } = await import("viem/accounts");
  return privateKeyToAccount(privateKey as `0x${string}`);
}

/**
 * Create a viem-compatible account from Coinbase CDP MPC wallet (production).
 *
 * Private keys never leave the AWS Nitro Enclave TEE. The application
 * only holds API credentials — even a compromised server cannot extract
 * the signing key. Requires `@coinbase/cdp-sdk` (optional dependency).
 *
 * @see https://docs.cdp.coinbase.com/server-wallets/v2/introduction/welcome
 */
async function createCdpAccount(
  apiKeyId: string,
  apiKeySecret: string,
  walletSecret: string,
  accountName: string,
): Promise<ViemAccount> {
  let CdpClient: any;
  try {
    const mod = await import("@coinbase/cdp-sdk");
    CdpClient = mod.CdpClient;
  } catch {
    throw new Error(
      "[verathos] @coinbase/cdp-sdk is required for CDP wallet mode. " +
        "Install it with: bun add @coinbase/cdp-sdk",
    );
  }

  const cdp = new CdpClient({
    apiKeyId,
    apiKeySecret,
    walletSecret,
  });

  // getOrCreateAccount is idempotent — same name always returns same address
  const account = await cdp.evm.getOrCreateAccount({ name: accountName });

  // CDP accounts implement the viem Account interface natively
  return account;
}

/**
 * Create the appropriate signer account based on config.
 *
 * Returns the account and a human-readable description for logging.
 */
export async function createSigner(
  runtime: IAgentRuntime,
): Promise<{ account: ViemAccount; description: string }> {
  const mode = getAuthMode(runtime);
  const testnet = isX402Testnet(runtime);
  const network = testnet ? "Base Sepolia" : "Base";

  if (mode === "x402_cdp") {
    const apiKeyId = getCdpApiKeyId(runtime)!;
    const apiKeySecret = getCdpApiKeySecret(runtime);
    const walletSecret = getCdpWalletSecret(runtime);

    if (!apiKeySecret || !walletSecret) {
      throw new Error(
        "CDP wallet requires CDP_API_KEY_ID, CDP_API_KEY_SECRET, and CDP_WALLET_SECRET",
      );
    }

    const accountName = getCdpAccountName(runtime);
    const account = await createCdpAccount(
      apiKeyId,
      apiKeySecret,
      walletSecret,
      accountName,
    );

    return {
      account,
      description: `CDP MPC wallet "${accountName}" (${account.address}) on ${network}`,
    };
  }

  if (mode === "x402_raw") {
    const privateKey = getX402PrivateKey(runtime)!;
    const account = await createRawKeyAccount(privateKey);

    return {
      account,
      description: `hot wallet ${account.address} on ${network}`,
    };
  }

  throw new Error("createSigner called but no x402 credentials configured");
}

// ── Fetch wrapper ───────────────────────────────────────────────

/**
 * Create a fetch wrapper that automatically handles x402 payments.
 *
 * Uses the official x402 SDK to generate correctly formatted EIP-712
 * payment signatures that the Coinbase facilitator accepts.
 *
 * Falls back to a manual implementation if `x402` is not installed,
 * with a warning that it may not work with all facilitators.
 */
export async function createX402Fetch(
  account: ViemAccount,
  testnet: boolean,
  baseFetch: typeof globalThis.fetch = globalThis.fetch,
): Promise<typeof globalThis.fetch> {
  // Try official x402 SDK first
  try {
    const x402Mod = await import("x402");
    const evmMod = await import("x402/evm");

    const client = new x402Mod.x402Client();
    evmMod.registerExactEvmScheme(client, { signer: account as any });

    // The x402 SDK provides wrapFetch which handles 402 → sign → retry
    const wrappedFetch = x402Mod.wrapFetch(baseFetch, client);

    console.log("[verathos] x402 SDK loaded — using official payment signing");
    return wrappedFetch;
  } catch {
    // x402 SDK not installed — fall back to manual implementation
    console.warn(
      "[verathos] x402 package not installed — using manual payment signing. " +
        "For production, install: bun add x402",
    );
  }

  // ── Manual fallback (works with Verathos proxy's custom verification) ──

  const BASE_MAINNET_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";
  const BASE_SEPOLIA_USDC = "0x036CbD53842c5426634e7929541eC2318f3dCF7e";
  const usdcAddress = testnet ? BASE_SEPOLIA_USDC : BASE_MAINNET_USDC;

  return async function x402FetchFallback(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    const response = await baseFetch(input, init);

    if (response.status !== 402) {
      return response;
    }

    // Parse payment requirements from 402 response
    let requirements: {
      accepts?: Array<{
        scheme: string;
        network: string;
        maxAmountRequired: string;
        resource: string;
        payTo: string;
        maxTimeoutSeconds: number;
        asset: string;
      }>;
    };
    try {
      requirements = await response.json();
    } catch {
      throw new Error(
        "[verathos] x402: received 402 but could not parse payment requirements",
      );
    }

    if (!requirements.accepts?.length) {
      throw new Error(
        "[verathos] x402: 402 response has no accepted payment schemes",
      );
    }

    const accept = requirements.accepts[0];

    // Sign the payment authorization (EIP-191)
    const payload = {
      scheme: "exact",
      network: accept.network,
      asset: accept.asset || usdcAddress,
      payTo: accept.payTo,
      amount: accept.maxAmountRequired,
      resource: accept.resource,
      nonce: Date.now().toString(),
      expiry: Math.floor(Date.now() / 1000) + (accept.maxTimeoutSeconds || 60),
    };

    const message = JSON.stringify(payload);
    const signature = await account.signMessage({ message });

    const paymentPayload = {
      payload: {
        authorization: {
          from: account.address,
          to: accept.payTo,
          value: accept.maxAmountRequired,
          asset: accept.asset || usdcAddress,
          chain: accept.network,
        },
        nonce: payload.nonce,
        expiry: payload.expiry,
      },
      signature,
    };

    const paymentHeader = btoa(JSON.stringify(paymentPayload));

    const retryInit: RequestInit = {
      ...init,
      headers: {
        ...(init?.headers instanceof Headers
          ? Object.fromEntries(init.headers.entries())
          : (init?.headers ?? {})),
        "X-PAYMENT": paymentHeader,
      },
    };

    const retryResponse = await baseFetch(input, retryInit);

    if (retryResponse.status === 402) {
      throw new Error(
        `[verathos] x402: payment rejected (${accept.maxAmountRequired} USDC to ${accept.payTo})`,
      );
    }

    return retryResponse;
  };
}
