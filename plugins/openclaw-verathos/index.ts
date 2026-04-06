/**
 * OpenClaw provider plugin for Verathos — verified LLM inference on Bittensor.
 *
 * Registers a "verathos" provider that routes to the Verathos API
 * (OpenAI-compatible). No SDK imports needed — OpenClaw loads plugins
 * via jiti and duck-types the exported object.
 */

const PROVIDER_ID = "verathos";
const BASE_URL = "https://api.verathos.ai/v1";
const API_KEY_ENV = "VERATHOS_API_KEY";

export default {
  id: PROVIDER_ID,
  name: "Verathos Provider",
  description:
    "Verified LLM inference on Bittensor — every response is cryptographically proven",
  version: "0.1.0",

  register(api: any) {
    api.registerProvider({
      id: PROVIDER_ID,
      label: "Verathos",
      docsPath: "/providers/verathos",
      envVars: [API_KEY_ENV],

      auth: [
        {
          id: "api-key",
          label: "Verathos API key",
          hint: "Get a key at verathos.ai",
          kind: "api_key",
          async run(ctx: any) {
            const key = await ctx.prompter.text({
              message: "Enter your Verathos API key:",
              validate: (v: string) => (v.length > 10 ? true : "Key too short"),
            });
            return {
              profiles: [
                {
                  profileId: `${PROVIDER_ID}-default`,
                  credential: { type: "api_key", apiKey: key },
                },
              ],
              configPatch: {
                models: {
                  providers: {
                    [PROVIDER_ID]: {
                      baseUrl: BASE_URL,
                      api: "openai-completions",
                      apiKey: key,
                    },
                  },
                },
              },
              defaultModel: `${PROVIDER_ID}/auto`,
            };
          },
        },
      ],

      discovery: {
        order: "simple",
        async run(ctx: any) {
          const resolved = ctx.resolveProviderApiKey(PROVIDER_ID);
          const apiKey = resolved.apiKey || ctx.env[API_KEY_ENV];
          if (!apiKey) return null;

          // Fetch live model list
          let models: Array<{ id: string }> = [];
          try {
            const res = await fetch(`${BASE_URL}/models`, {
              headers: { Authorization: `Bearer ${apiKey}` },
            });
            if (res.ok) {
              const data = await res.json();
              models = (data as any).data ?? [];
            }
          } catch {}

          const allModels = [
            {
              id: "auto",
              name: "Auto (best available)",
              api: "openai-completions",
              reasoning: false,
              input: ["text"],
              contextWindow: 131072,
              maxTokens: 8192,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            },
            ...models.map((m) => ({
              id: m.id,
              name: m.id,
              api: "openai-completions",
              reasoning: false,
              input: ["text"],
              contextWindow: 131072,
              maxTokens: 8192,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            })),
          ];

          return {
            provider: {
              baseUrl: BASE_URL,
              api: "openai-completions",
              apiKey,
              models: allModels,
            },
          };
        },
      },

      wizard: {
        onboarding: {
          choiceId: "verathos",
          choiceLabel: "Verathos",
          choiceHint: "Verified LLM inference on Bittensor",
          methodId: "api-key",
        },
        modelPicker: {
          label: "Verathos",
          hint: 'Use "auto" for best available model',
          methodId: "api-key",
        },
      },
    });

    api.logger.info("Verathos provider registered");
  },
};
