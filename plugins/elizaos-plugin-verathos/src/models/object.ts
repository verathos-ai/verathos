import { generateObject, jsonSchema } from "ai";
import type { IAgentRuntime, ObjectGenerationParams } from "@elizaos/core";
import { createVerathosClient } from "../providers/verathos.js";
import { getSmallModel, getLargeModel } from "../utils/config.js";

type ModelNameGetter = (runtime: IAgentRuntime) => string;

async function handleObjectGeneration(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
  getModelName: ModelNameGetter,
): Promise<Record<string, unknown>> {
  const client = await createVerathosClient(runtime);
  const modelName = getModelName(runtime);
  const model = client.chat(modelName);

  // AI SDK accepts Zod schemas or jsonSchema() wrapper for plain JSON schemas
  const schema = params.schema ? jsonSchema(params.schema as any) : undefined;

  const { object } = await generateObject({
    model,
    prompt: params.prompt,
    schema: schema as any,
    system: runtime.character?.system ?? undefined,
    maxTokens: params.maxTokens ?? 8192,
    temperature: params.temperature ?? 0.7,
  });

  return object as Record<string, unknown>;
}

export async function handleObjectSmall(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
): Promise<Record<string, unknown>> {
  return handleObjectGeneration(runtime, params, getSmallModel);
}

export async function handleObjectLarge(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
): Promise<Record<string, unknown>> {
  return handleObjectGeneration(runtime, params, getLargeModel);
}
