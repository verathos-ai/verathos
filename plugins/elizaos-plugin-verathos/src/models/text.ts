import { generateText, streamText } from "ai";
import type { IAgentRuntime, GenerateTextParams, TextStreamResult, TokenUsage } from "@elizaos/core";
import { createVerathosClient } from "../providers/verathos.js";
import { getSmallModel, getLargeModel } from "../utils/config.js";

type ModelNameGetter = (runtime: IAgentRuntime) => string;

async function handleTextGeneration(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
  getModelName: ModelNameGetter,
): Promise<string | TextStreamResult> {
  const client = await createVerathosClient(runtime);
  const modelName = getModelName(runtime);
  const model = client.chat(modelName);

  const systemPrompt = runtime.character?.system ?? undefined;

  const generateParams = {
    model,
    prompt: params.prompt,
    system: systemPrompt,
    maxTokens: params.maxTokens ?? 8192,
    temperature: params.temperature ?? 0.7,
    topP: params.topP,
    frequencyPenalty: params.frequencyPenalty,
    presencePenalty: params.presencePenalty,
    stopSequences: params.stopSequences,
  };

  // Streaming path
  if (params.stream) {
    const result = streamText(generateParams);

    return {
      textStream: result.textStream,
      text: result.text,
      usage: result.usage.then((u): TokenUsage | undefined =>
        u
          ? {
              promptTokens: u.inputTokens ?? 0,
              completionTokens: u.outputTokens ?? 0,
              totalTokens: u.totalTokens ?? (u.inputTokens ?? 0) + (u.outputTokens ?? 0),
            }
          : undefined,
      ),
      finishReason: result.finishReason.then((r) => r as string | undefined),
    };
  }

  // Non-streaming path
  const { text } = await generateText(generateParams);
  return text;
}

export async function handleTextSmall(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
): Promise<string | TextStreamResult> {
  return handleTextGeneration(runtime, params, getSmallModel);
}

export async function handleTextLarge(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
): Promise<string | TextStreamResult> {
  return handleTextGeneration(runtime, params, getLargeModel);
}
