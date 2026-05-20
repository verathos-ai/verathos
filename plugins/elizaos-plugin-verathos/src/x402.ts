/**
 * x402 USDC pay-per-request — signer abstraction + fetch wrapper.
 *
 * STATUS: temporarily disabled at runtime.
 *
 * The Verathos gateway migrated to the x402 ``upto`` scheme (Permit2-
 * based, post-inference settlement) in v0.1.10.  The upstream x402
 * TypeScript SDK (npm ``x402``) only ships ``exact``-scheme support
 * as of v1.2.0 (April 2026), so the existing TS plugin can't sign
 * authorisations the gateway will accept.  Hand-rolling Permit2
 * EIP-712 typed-data signing in TypeScript is risky (silent payment
 * failures are the most common bug class here), so this plugin's
 * x402 path raises a clear error instead.
 *
 * Track upstream TS upto support:
 *   https://github.com/x402-foundation/x402
 *
 * Working alternatives until then:
 *   • This plugin's ``api_key`` mode (sponsored key + USDC deposit)
 *   • The reference Python client at
 *     ``examples/x402_client.py`` in the public Verathos repo
 *     (uses the Python x402 SDK which DOES support upto)
 *
 * The signer-creation code below (CDP MPC, raw key) is still useful
 * for the future re-enable; it's kept intact.
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
// Currently disabled — see the file-level docstring above for context
// and re-enable plan.  When the TS x402 SDK ships upto-scheme support,
// restore the wrapFetch flow with the upto registration call instead of
// ``registerExactEvmScheme``.
export async function createX402Fetch(
  _account: ViemAccount,
  _testnet: boolean,
  _baseFetch: typeof globalThis.fetch = globalThis.fetch,
): Promise<typeof globalThis.fetch> {
  throw new Error(
    "[verathos] x402 mode is currently unavailable in this plugin: the " +
      "Verathos gateway requires the x402 'upto' scheme (Permit2-based, " +
      "post-inference settlement, v0.1.10+), but the TypeScript x402 SDK " +
      "only supports the 'exact' scheme as of v1.2.0. " +
      "Use auth_mode='api_key' for now, or call api.verathos.ai directly " +
      "with the Python x402 SDK (see examples/x402_client.py in the " +
      "Verathos public repo).",
  );
}
