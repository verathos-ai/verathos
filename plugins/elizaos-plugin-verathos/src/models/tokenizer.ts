import { getEncoding, type TiktokenEncoding } from "js-tiktoken";

// cl100k_base covers GPT-4, GPT-3.5, and most modern models
const DEFAULT_ENCODING: TiktokenEncoding = "cl100k_base";

let cachedEncoder: ReturnType<typeof getEncoding> | null = null;

function getEncoder() {
  if (!cachedEncoder) {
    cachedEncoder = getEncoding(DEFAULT_ENCODING);
  }
  return cachedEncoder;
}

export async function handleTokenizerEncode(
  _runtime: unknown,
  params: { prompt: string },
): Promise<number[]> {
  const encoder = getEncoder();
  return encoder.encode(params.prompt);
}

export async function handleTokenizerDecode(
  _runtime: unknown,
  params: { tokens: number[] },
): Promise<string> {
  const encoder = getEncoder();
  return encoder.decode(params.tokens);
}
