const browserFetch = self.fetch.bind(self);
self.fetch = (input, init = {}) => browserFetch(input, { ...init, referrerPolicy: "no-referrer" });

const webllm = await import("https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm");
const MODELS = {
  "Qwen3-0.6B-q4f16_1-MLC": "Qwen3 0.6B",
  "Qwen3.5-0.8B-q4f16_1-MLC": "Qwen3.5 0.8B",
};
const MIN_OPTIONS = 2;
const MAX_OPTIONS = 20;
const labelsFor = (count) => Array.from({ length: count }, (_, index) => String.fromCharCode(65 + index));
const labelTokenId = (label) => label.charCodeAt(0) - 33;
let engine;
let modelId;

function send(type, data = {}) { self.postMessage({ type, ...data }); }
function optionBlock(options, labels = labelsFor(options.length)) {
  return options.map((option, index) => `${labels[index]}. ${option}`).join("\n");
}
function messagesFor({ state, question, options }, mode) {
  const labels = labelsFor(options.length);
  const outputInstruction = mode === "direct"
    ? `Reply with exactly one option letter from: ${labels.join(", ")}.`
    : `Estimate the probability that each allowed option is the correct decision.
Return only one JSON object mapping each option to its probability. Form every key as "<label>: <full option text>" using the allowed options above.
For example, if the unrelated options were "A. Route north" and "B. Route south", valid output would be:
{"A: Route north": 0.65, "B: Route south": 0.35}
For the actual decision, include every supplied option exactly once and in order. Each value must be a JSON number from 0 to 1, and the probabilities must sum to 1. Output JSON only, with no markdown or explanation.`;
  return [
    { role: "system", content: "Make the requested decision from the supplied state. Follow the output format exactly." },
    { role: "user", content: `State:\n${state}\n\nQuestion:\n${question}\n\nAllowed options:\n${optionBlock(options, labels)}\n\n${outputInstruction}` },
  ];
}
function softmax(values) {
  const maximum = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

function validateOptionLogprobs(values, labels) {
  if (!values || values.length !== labels.length || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`The model did not return log-probabilities for ${labels.join(", ")}.`);
  }
}

function optionLogprobs(response, labels) {
  const entries = response.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs ?? [];
  return labels.map((label) => {
    const ascii = label.charCodeAt(0);
    const entry = entries.find((item) => item.token === label || item.bytes?.[0] === ascii);
    return Number(entry?.logprob);
  });
}
function validateGeneration(text, data) {
  try {
    // Qwen3 may emit an empty or populated reasoning wrapper even when thinking
    // is disabled. It is model output, so keep displaying it, but validate the
    // decision payload that follows it.
    const payload = text.trim().replace(/^<think>[\s\S]*?<\/think>\s*/i, "");
    const parsed = JSON.parse(payload);
    const exactKeys = (value, expected) => {
      if (!value || typeof value !== "object" || Array.isArray(value)) return false;
      const actual = Object.keys(value);
      return actual.length === expected.length && expected.every((key) => actual.includes(key));
    };
    const labels = labelsFor(data.options.length);
    const expectedKeys = data.options.map((option, index) => `${labels[index]}: ${option}`);
    if (!exactKeys(parsed, expectedKeys)) throw new Error("expected one probability for every exact option key");
    const probabilities = expectedKeys.map((key) => {
      const probability = parsed[key];
      if (typeof probability !== "number" || !Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new Error("probabilities must be numbers from 0 to 1");
      }
      return probability;
    });
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    if (Math.abs(total - 1) > 0.02) throw new Error("probabilities must sum to 1");
    const index = probabilities.indexOf(Math.max(...probabilities));
    return { valid: true, choice: labels[index], choiceDescription: data.options[index], validationError: "" };
  } catch (error) {
    return { valid: false, choice: null, choiceDescription: null, validationError: error?.message ?? "invalid JSON" };
  }
}

async function load(requestedModelId) {
  if (engine) return;
  if (!Object.hasOwn(MODELS, requestedModelId)) throw new Error("Choose one of the listed models.");
  modelId = requestedModelId;
  send("loading", { message: "Fetching the model or reading it from your browser cache…" });
  const loadStart = performance.now();
  engine = await webllm.CreateMLCEngine(
    modelId,
    {
      logLevel: "WARN",
      initProgressCallback: (report) => {
        send("progress", { event: {
          status: "progress", file: modelId,
          loaded: Math.max(0, Math.min(1, report.progress ?? 0)), total: 1, text: report.text,
        } });
        if (report.text) send("loading", { message: report.text });
      },
    },
    { context_window_size: 2048 },
  );
  const loadMs = performance.now() - loadStart;
  send("loaded", { loadMs });
  send("loading", { message: "Model loaded. Compiling a real Qwen pass…" });
  const warmupStart = performance.now();
  const warmup = await engine.chat.completions.create({
    messages: [{ role: "user", content: "Reply with the single word ready." }],
    max_tokens: 1, temperature: 0, extra_body: { enable_thinking: false },
  });
  if (!warmup?.choices?.length) throw new Error("Qwen warmup returned no completion.");
  send("ready", { warmupMs: performance.now() - warmupStart, modelId, modelName: MODELS[modelId] });
}

async function directScore(data) {
  const started = performance.now();
  const labels = labelsFor(data.options.length);
  const groups = labels.length <= 5
    ? [labels]
    : Array.from({ length: Math.ceil((labels.length - 1) / 4) }, (_, index) => [labels[0], ...labels.slice(1 + index * 4, 1 + (index + 1) * 4)]);
  const relativeLogits = new Map([[labels[0], 0]]);
  let inputTokens = 0;
  for (const group of groups) {
    const logitBias = Object.fromEntries(group.map((label) => [String(labelTokenId(label)), 100]));
    const response = await engine.chat.completions.create({
      messages: messagesFor(data, "direct"), max_tokens: 1, temperature: 1,
      logprobs: true, top_logprobs: group.length, logit_bias: logitBias,
      extra_body: { enable_thinking: false },
    });
    inputTokens += response.usage?.prompt_tokens ?? 0;
    const values = optionLogprobs(response, group);
    validateOptionLogprobs(values, group);
    const anchor = values[0];
    group.forEach((label, index) => relativeLogits.set(label, values[index] - anchor));
  }
  const logits = labels.map((label) => relativeLogits.get(label));
  validateOptionLogprobs(logits, labels);
  const probabilities = softmax(logits);
  return {
    totalMs: performance.now() - started, inputTokens, readouts: groups.length,
    options: data.options.map((description, index) => ({
      label: labels[index], description, probability: probabilities[index], logit: logits[index],
    })),
  };
}
async function generateAnswer(data) {
  const started = performance.now();
  let firstTokenAt = null;
  let generatedText = "";
  let usage = null;
  send("generation-start");
  const stream = await engine.chat.completions.create({
    messages: messagesFor(data, "generation"), stream: true,
    stream_options: { include_usage: true }, max_tokens: 512, temperature: 0,
    extra_body: { enable_thinking: false },
  });
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content ?? "";
    if (text && firstTokenAt == null) firstTokenAt = performance.now();
    generatedText += text;
    if (chunk.usage) usage = chunk.usage;
    send("generation-update", {
      text: generatedText, tokens: usage?.completion_tokens ?? 0,
      ttftMs: firstTokenAt == null ? null : firstTokenAt - started,
    });
  }
  generatedText = generatedText.trim();
  return {
    generationMs: performance.now() - started,
    inputTokens: usage?.prompt_tokens ?? 0,
    ttftMs: firstTokenAt == null ? null : firstTokenAt - started,
    generatedTokens: usage?.completion_tokens ?? 0,
    generatedText, ...validateGeneration(generatedText, data),
  };
}

async function compare(data) {
  if (!engine) throw new Error("Load the model before running a comparison.");
  if (!Array.isArray(data.options) || data.options.length < MIN_OPTIONS || data.options.length > MAX_OPTIONS) {
    throw new Error(`This demo requires ${MIN_OPTIONS} to ${MAX_OPTIONS} options.`);
  }
  const direct = await directScore(data);
  send("direct", direct);
  const generation = await generateAnswer(data);
  send("complete", { ...generation, directMs: direct.totalMs });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await load(data.modelId);
    if (data.type === "compare") await compare(data.data);
  } catch (error) {
    console.error(error);
    send("error", { message: error?.message ?? String(error) });
  }
});
