# OpenJev browser lab

This is a browser-only comparison of two readout paths through the same quantized Qwen3-0.6B model:

1. **Direct readout** runs one causal-model forward pass and normalizes the next-token logits for the allowed single-token labels `A`, `B`, and `C`.
2. **Generation** greedily decodes a compact JSON answer, with a 48-token limit.

It is a live experiment, not a prerecorded benchmark. The page displays only timings collected in the current browser session. Model loading and shader warmup are reported separately from both decision paths. The paths run sequentially to avoid WebGPU contention.

## Run locally

Web Workers and model downloads require an HTTP origin:

```bash
cd webgpu-demo
python3 -m http.server 8080
```

Open `http://localhost:8080` in a current WebGPU-capable browser whose adapter exposes `shader-f16`. Expect roughly 570 MB of model weights on first load. Browser caching controls repeat downloads. The page reports a clear compatibility message before any download begins.

For deployment, any static HTTPS host is sufficient. No build step, API, database, telemetry, or server-side inference is used. Runtime UI and inference dependencies come from CDN: pinned Vue and Transformers.js builds, plus Material Symbols from Google Fonts.

## Pins

- Transformers.js: `4.3.0`
- Vue: `3.5.21`
- Model: [`onnx-community/Qwen3-0.6B-ONNX`](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX)
- Model revision: `da1453100cf3ff33ef56d17983fc7a8648706db6`
- ONNX dtype: `q4f16` (`onnx/model_q4f16.onnx`, approximately 570 MB)

## Measurement boundary

- **Model load** starts before tokenizer/model construction and ends after both promises resolve. It includes network/cache reads, decoding, and GPU setup exposed by the library.
- **Warmup** measures an unreported direct pass and a forced two-token generation so both the initial read and cached decode step compile before the comparison.
- **Direct total** includes prompt rendering, tokenization, one forward pass, selected-logit extraction, and softmax.
- **Generation TTFT** starts before prompt rendering/tokenization and stops in the first token callback.
- **Generation total** uses the same start and stops after the returned answer is decoded and checked against the required JSON shape.
- **Generated tokens** are computed from the returned sequence length. The live counter uses the token callback while generation is running.

The two prompts contain identical state, question, and option text. Their final format instructions differ: direct readout requests one option letter; generation requests `{"choice":"A"}`. This is necessary to exercise each interface honestly.

Direct probabilities are conditional on only the three displayed label tokens. They are not calibrated probabilities, and a high value does not establish that the underlying decision is correct.

## Primary sources

- [Official Transformers.js Qwen3 WebGPU example](https://github.com/huggingface/transformers.js-examples/tree/main/qwen3-webgpu)
- [Official Transformers.js WebGPU guide](https://huggingface.co/docs/transformers.js/guides/webgpu)
- [Qwen3-0.6B ONNX model card and browser usage](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX)
- [Transformers.js source](https://github.com/huggingface/transformers.js)

The upstream model and runtime retain their respective licenses. This repository's original code is MIT licensed.
