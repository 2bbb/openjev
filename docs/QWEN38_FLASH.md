# Qwen3.8 Flash Next with an existing mlx-serve process

This branch starts from [Apple Silicon PR #2](https://github.com/TheoLeeCJ/openjev/pull/2),
commit `1cd6eb2a8395598b5936451e02fbc201ce86f514`, and adds `--backend mlx-serve`.
It reuses a resident Qwen3.8 Flash Next model without loading a second copy or
downloading Qwen3.5. Native `--backend mlx` remains limited to Qwen3.5. The additional
[`qwen38-native` backend](QWEN38_NATIVE.md) directly loads this checkpoint and
supports shared prefixes without the API top-20 limit.

## Prerequisites and usage

- An existing **mlx-serve 26.9.4** process on Apple Silicon, serving one
  `qwen4_exp` model on loopback HTTP. KV and decode-attention quantization must
  be off. This adapter does not start, stop, or reconfigure the server.
- Local tokenizer assets from the same model directory. Verified checkpoint:
  [`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/tree/7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3).
- Python environment installed with `pip install -e '.[test]'`. The adapter
  itself loads only the tokenizer and uses standard-library HTTP; the MLX extra
  is needed for PR #2's separate native MLX backend, not this bridge.

Run from this repository, replacing the model directory with your existing path:

```sh
openjev-score --backend mlx-serve --mode direct \
  --server-url http://127.0.0.1:18082 \
  --model /path/to/Qwen3.8-Flash-Next \
  --revision 7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3 \
  --input examples/qwen38flash-decisions.jsonl \
  --output results-qwen38flash.jsonl
```

The output path must be new. Inputs are not truncated. On a failed row, the
process exits unsuccessfully; the output may retain earlier successful rows.
The revision is the caller's provenance declaration, not a server weight-hash
attestation. Verify the model download separately and point both processes at
the same checkpoint.

## Readout and limits

The adapter uses OpenJev's unchanged `direct-options-v1` messages, prompt
construction, single-token answer slots, and boundary checks. It also checks
that the server tokenizes the full prompt and every answer boundary identically.
The rendered prompt is sent to `/v1/completions`, avoiding a second chat template.

mlx-serve's first-position logprobs are computed before temperature and sampling
filters. For a complete set of answer-slot logprobs `l_i = z_i - logsumexp(z)`,
`softmax(l_i)` equals the conditional option softmax from the original logits.
The adapter never parses the generated text into a decision. It returns
`option_logprobs`, not mislabeled raw `option_logits`, and preserves the raw API
response, prompt/token hashes, tokenizer-asset hashes, and transport timing.

There are material limits:

- The pinned API returns **at most 20 vocabulary alternatives**. Every supplied
  answer slot must be present. If even one is missing, scoring fails rather than
  assigning it zero probability or renormalizing an incomplete set. Even a
  two-option question can fail this check; arbitrary 2–16-option coverage is not
  guaranteed. A native selected-token-logit endpoint would remove this limit.
- The API samples **one transport token**, which is discarded. There is no
  generated reasoning or multi-token continuation, but this is not the native
  no-generation forward path. Serial/shared/reranker modes are rejected.
  The server's ordinary prefix cache can still operate; it is not OpenJev's
  explicit serial/shared cache algorithm.
- Scores inherit the model's mixed quantization, runtime floating-point precision,
  and six-decimal logprob serialization. They are not bit-identical to a direct
  FP32-logit readout or calibrated confidence estimates. No equivalence to the
  published Qwen3.5 BF16 quality results is claimed.

The adapter pins the server version because the API semantics are part of the
readout contract. See the upstream [completion handler](https://github.com/ddalcu/mlx-serve/blob/v26.9.4/src/server.zig)
and [logprob computation](https://github.com/ddalcu/mlx-serve/blob/v26.9.4/src/generate.zig).
MTP, draft decoding, PLD, and sampling penalties are disabled per request.

## Local validation, 2026-09-18

M5 Max, 128 GiB, macOS 26.4; MLX 0.32.2 in the native server. The three retained
smoke rows cover English account routing, reversed option order, and Japanese
billing routing. All selected the expected option, with finite probabilities
summing to one. Request times were approximately 1.16 s, 0.17 s, and 0.97 s with
the server already resident. These are tiny functional checks, not a quality or
throughput benchmark. Raw records are in `results/qwen38flash/2026-09-18/`.

The combined test suite passed **61 tests**; all **19 published raw-file hashes**
matched, and `benchmarks/verify_published.py` verified **69 existing claims**.
These integrity checks do not constitute a fresh CUDA hardware run.
