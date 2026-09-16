import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

const MODEL_ID = "onnx-community/Qwen3-0.6B-ONNX";
const MODEL_REVISION = "da1453100cf3ff33ef56d17983fc7a8648706db6";
const LABELS = ["A", "B", "C"];

let tokenizer;
let model;

async function disposeTensors(value, seen = new Set()) {
  if (value == null || typeof value !== "object" || seen.has(value)) return;
  seen.add(value);
  if (typeof value.dispose === "function") {
    await value.dispose();
    return;
  }
  await Promise.all(Object.values(value).map((item) => disposeTensors(item, seen)));
}

function send(type, data = {}) {
  self.postMessage({ type, ...data });
}

function optionBlock(options) {
  return options.map((option, index) => `${LABELS[index]}. ${option}`).join("\n");
}

function messagesFor({ state, question, options }, mode) {
  const outputInstruction = mode === "direct"
    ? "Reply with exactly one option letter: A, B, or C."
    : 'Return only compact JSON in this exact shape: {"choice":"A"}. Replace A with one allowed option letter. Do not explain.';
  return [
    { role: "system", content: "Make the requested decision from the supplied state. Follow the output format exactly." },
    { role: "user", content: `State:\n${state}\n\nQuestion:\n${question}\n\nAllowed options:\n${optionBlock(options)}\n\n${outputInstruction}` },
  ];
}

function chatInputs(messages) {
  return tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: false,
  });
}

function tokenIdsForLabels() {
  return LABELS.map((label) => {
    const ids = tokenizer.encode(label, { add_special_tokens: false });
    if (ids.length !== 1) throw new Error(`Option label ${label} is not one token for this tokenizer.`);
    return Number(ids[0]);
  });
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

function validateGeneration(text, data) {
  try {
    const parsed = JSON.parse(text.trim());
    const keys = parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? Object.keys(parsed)
      : [];
    if (keys.length !== 1 || keys[0] !== "choice") {
      throw new Error('expected exactly {"choice":"A"}');
    }
    if (typeof parsed.choice !== "string" || !LABELS.includes(parsed.choice)) {
      throw new Error("choice must be A, B, or C");
    }
    const index = LABELS.indexOf(parsed.choice);
    return {
      valid: true,
      choice: parsed.choice,
      choiceDescription: data.options[index],
      validationError: "",
    };
  } catch (error) {
    return {
      valid: false,
      choice: null,
      choiceDescription: null,
      validationError: error?.message ?? "invalid JSON",
    };
  }
}

async function load() {
  if (model && tokenizer) return;
  send("loading", { message: "Fetching model files or reading them from the browser cache…" });
  const loadStart = performance.now();
  const progress_callback = (event) => send("progress", { event });
  [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL_ID, { revision: MODEL_REVISION, progress_callback }),
    AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      revision: MODEL_REVISION,
      dtype: "q4f16",
      device: "webgpu",
      progress_callback,
    }),
  ]);
  const loadMs = performance.now() - loadStart;
  send("loaded", { loadMs });

  send("loading", { message: "Model loaded. Compiling both decision paths…" });
  const warmupStart = performance.now();
  tokenIdsForLabels();
  const sample = {
    state: "The account is locked.",
    question: "Which queue should handle this request?",
    options: ["Account access", "Billing", "Close"],
  };
  const directInputs = chatInputs(messagesFor(sample, "direct"));
  const directOutput = await model(directInputs);
  await disposeTensors(directOutput);
  await disposeTensors(directInputs);
  const generationInputs = chatInputs(messagesFor(sample, "generation"));
  const generationOutput = await model.generate({
    ...generationInputs,
    min_new_tokens: 2,
    max_new_tokens: 2,
    do_sample: false,
  });
  await disposeTensors(generationOutput);
  await disposeTensors(generationInputs);
  const warmupMs = performance.now() - warmupStart;
  send("ready", { warmupMs, modelId: MODEL_ID, revision: MODEL_REVISION });
}

async function directScore(data) {
  const started = performance.now();
  const inputs = chatInputs(messagesFor(data, "direct"));
  const output = await model(inputs);
  const logits = output.logits;
  const dimensions = logits.dims;
  const vocabularySize = dimensions.at(-1);
  const sequenceLength = dimensions.at(-2);
  const offset = (sequenceLength - 1) * vocabularySize;
  const values = tokenIdsForLabels().map((tokenId) => Number(logits.data[offset + tokenId]));
  const probabilities = softmax(values);
  const inputTokens = Number(inputs.input_ids.dims.at(-1));
  const totalMs = performance.now() - started;
  await disposeTensors(output);
  await disposeTensors(inputs);
  return {
    totalMs,
    inputTokens,
    options: data.options.map((description, index) => ({
      label: LABELS[index],
      description,
      probability: probabilities[index],
      logit: values[index],
    })),
  };
}

async function generateAnswer(data) {
  const started = performance.now();
  const inputs = chatInputs(messagesFor(data, "generation"));
  const promptTokens = Number(inputs.input_ids.dims.at(-1));
  let firstTokenAt = null;
  let generatedTokens = 0;
  let streamedText = "";
  send("generation-start");
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    token_callback_function: (tokens) => {
      firstTokenAt ??= performance.now();
      generatedTokens += tokens.length;
    },
    callback_function: (text) => {
      streamedText += text;
      send("generation-update", {
        text: streamedText,
        tokens: generatedTokens,
        ttftMs: firstTokenAt == null ? null : firstTokenAt - started,
      });
    },
  });
  const result = await model.generate({
    ...inputs,
    max_new_tokens: 48,
    do_sample: false,
    streamer,
  });
  const totalTokens = Number(result.dims.at(-1));
  const decoded = tokenizer.decode(result.data.slice(promptTokens), { skip_special_tokens: true });
  const generatedText = decoded.trim() || streamedText.trim();
  const validation = validateGeneration(generatedText, data);
  const generationMs = performance.now() - started;
  await disposeTensors(result);
  await disposeTensors(inputs);
  return {
    generationMs,
    inputTokens: promptTokens,
    ttftMs: firstTokenAt == null ? null : firstTokenAt - started,
    generatedTokens: Math.max(0, totalTokens - promptTokens),
    generatedText,
    ...validation,
  };
}

async function compare(data) {
  if (!model || !tokenizer) throw new Error("Load the model before running a comparison.");
  if (!Array.isArray(data.options) || data.options.length !== LABELS.length) throw new Error("This demo requires exactly three options.");
  const direct = await directScore(data);
  send("direct", direct);
  const generation = await generateAnswer(data);
  send("complete", { ...generation, directMs: direct.totalMs });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await load();
    if (data.type === "compare") await compare(data.data);
  } catch (error) {
    console.error(error);
    send("error", { message: error?.message ?? String(error) });
  }
});
