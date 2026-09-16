# OpenJev browser lab

This is a browser-only comparison of two readout paths through the same selected quantized Qwen model:

1. **Direct readout** obtains WebLLM's log-probabilities for the allowed single-token labels and normalizes them over the displayed options. Two to five options need one readout. Six to twenty use groups of four plus a shared `A` anchor because WebLLM exposes at most five top log-probabilities per call; log-probability differences against that anchor place every group on one scale.
2. **Generation** greedily decodes a JSON distribution, with a 512-token limit. The model writes each full option string as a key and its estimated probability as the value.

It is a live experiment, not a prerecorded benchmark. The page displays only timings collected in the current browser session. Model loading and shader warmup are reported separately from both decision paths. The paths run sequentially to avoid WebGPU contention.

The account-support and email-triage buttons only prefill the editable inputs. They do not constrain the prompt or run the model. Options remain editable, with add/remove controls for two through twenty choices.

## Run locally

Web Workers and model downloads require an HTTP origin:

```bash
cd webgpu-demo
python3 -m http.server 8080
```

Open `http://localhost:8080` in a current WebGPU-capable browser whose adapter exposes `shader-f16`. Depending on the selected model, expect roughly 352 MB to 447 MB of model assets on first load. Browser caching controls repeat downloads. The page reports a clear compatibility message before any download begins.

For deployment, any static HTTPS host is sufficient. No build step, API, database, telemetry, or server-side inference is used. Runtime UI and inference dependencies come from CDN: pinned Vue and WebLLM builds, plus Material Symbols from Google Fonts.

Keep `_headers` when deploying to Cloudflare. It applies `Referrer-Policy: no-referrer`, matching the page and worker policy, so direct cross-origin Hugging Face asset requests do not carry the hosting URL as a referrer.

## Pins

- WebLLM: `0.2.85`
- Vue: `3.5.21`
- Models: [`Qwen3-0.6B`](https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC) and [`Qwen3.5-0.8B`](https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC)
- MLC dtype: `q4f16_1`; model assets are approximately 352 MB and 447 MB respectively

The JavaScript runtime versions are pinned, but WebLLM resolves the two model IDs through its catalog to live Hugging Face repositories. The browser lab is an operational demo rather than a bit-for-bit archival benchmark; the measured Phase 1 evidence uses the immutable model revisions in `../manifests/models.json`.

## Measurement boundary

- **Model load** starts before WebLLM engine construction and ends when its model load resolves. It includes network/cache reads and GPU setup exposed by the library.
- **Warmup** measures an unreported one-token completion that compiles a real Qwen pass before the comparison.
- **Direct total** includes prompt rendering, tokenization, every required one-token readout, and softmax. An equal `+100` logit bias places each group in WebLLM's bounded top-logprobs response without changing relative logits. Groups beyond the first reuse `A` as a common anchor.
- **Generation TTFT** starts before prompt rendering/tokenization and stops in the first token callback.
- **Generation total** uses the same start and stops after the returned answer is decoded and checked against the required JSON shape.
- **Generated tokens** come from WebLLM's completion usage record.

Qwen3 may prepend a `<think>...</think>` block even when thinking is disabled. The page streams that model output unchanged, removes one leading reasoning block for format validation, before parsing it internally for diagnostics. The page shows the raw model output without a validation/error banner.

The two prompts contain identical state, question, and option text. Their final format instructions differ: direct readout requests one option letter; generation asks the model to report a distribution by writing every option and probability as JSON. These generated, self-reported probabilities are a separate readout and need not match the direct token probabilities.

Direct probabilities are conditional on only the displayed label tokens. They are not calibrated probabilities, and a high value does not establish that the underlying decision is correct.

## Primary sources

- [Official WebLLM Qwen3 example](https://github.com/mlc-ai/web-llm/tree/main/examples/qwen3)
- [Official WebLLM API reference](https://webllm.mlc.ai/docs/user/api_reference.html)
- [WebLLM 0.2.85 model catalog](https://github.com/mlc-ai/web-llm/blob/v0.2.85/src/config.ts)
- [WebLLM source and documentation](https://github.com/mlc-ai/web-llm)

The upstream model and runtime retain their respective licenses. This repository's original code is MIT licensed.
