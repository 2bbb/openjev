# Native Qwen3.8 scoring on Apple Silicon

This branch adds `--backend qwen38-native` alongside the resident `mlx-serve`
bridge. It directly reads the **same existing packed checkpoint**; it does not
convert or duplicate weights on disk. Text decisions only, with zero generated
tokens and no API top-20 restriction.

## Installation and execution

Install `pip install -e '.[test,qwen38]'` on Apple Silicon. The adapter enforces
MLX `0.32.2` and [mlx-vlm commit
`10db092733a416439fce874cd96a7835b700f43d`](https://github.com/Blaizzy/mlx-vlm/tree/10db092733a416439fce874cd96a7835b700f43d/mlx_vlm/models/qwen4_exp).
The tested checkpoint is
[`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/tree/7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3)
at revision `7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3`.

Stop another server holding this model before loading it directly. The loader
requires the selected weight bytes plus 5 GiB of available memory. Weights are
approximately 72.9 GB on the tested pack; the 32 GB n-gram table remains a
read-only CPU mmap, with only requested rows transferred to the GPU.

```sh
openjev-score --backend qwen38-native --mode shared \
  --model /path/to/Qwen3.8-Flash-Next \
  --revision 7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3 \
  --input examples/qwen38flash-decisions.jsonl \
  --output native-scores.jsonl
```

Outputs are create-only. The CLI groups rows by their exact JSON-serialized
state, computes each group's prefix once, and preserves the original output
order. `--mode direct` instead starts a fresh cache for every decision.
`--prefill-chunk` controls the maximum tokens per prefill call (default 2048).
Long inputs exceeding `--max-tokens` are rejected without truncation. The model
is loaded once per invocation and released on exit, so batch decisions into one
JSONL file to amortize startup. Serial and reranker modes are not implemented.

## Checkpoint adaptation

The implementation uses upstream `mlx-vlm` Qwen4 computation with a narrow loader
for this `mlx-serve` pack:

- Preserve the pack's folded RMSNorm weights instead of applying `1 + weight`
  a second time. Other ordinary/gated norms keep their original semantics.
- Infer each 4- or 8-bit affine quantizer and group size from packed tensor
  shapes; check all expected tensor keys and shapes before loading shards.
- Read the merged n-gram safetensors header into a temporary mmap manifest;
  no 32 GB table copy or vocabulary-wide GPU allocation is needed.
- Exclude MTP and vision weights. Read the last hidden position and the selected
  answer logits directly. No text generation, sampling, MTP, or HTTP request.

Source/revision, runtime pins, tokenizer/prompt token fingerprints, selected
logits, and model header hashes are recorded. Header hashes and a caller-supplied
revision do **not** authenticate payload bytes; verify the downloaded model's
SHA-256 manifest independently.

## Shared prefix semantics

A branch copies the complete upstream cache, including attention KV, QSA indexer
state, recurrent GDN state, PLE convolution state, and n-gram history. Immutable
MLX array storage may be shared until operations replace it. The tokenized prefix
must match every complete prompt exactly.

Suffixes run as **isolated sequential branches**. The pinned upstream runtime
itself serializes batched multi-token prefill for Qwen4 numerical stability.
This implementation therefore claims shared prefix reuse, not simultaneous
parallel suffix execution. Changing prefill boundaries can change floating-point
reduction order and near-tied decisions; empirical comparisons must retain
probabilities and decision flips rather than claim bitwise equivalence.

## Reproducible evaluation

`benchmarks/qwen38_native.py` evaluates the repository's authored, perturbation,
and shared-state fixtures. It records load time separately from warmup and timed
scoring, keeps failed rows in the denominator, preserves raw predictions, and
reports quality, timing and native MLX memory. Supply each backend in a separate
process so two model copies are never resident together.

```sh
python benchmarks/qwen38_native.py \
  --backend qwen38-native --model /path/to/Qwen3.8-Flash-Next \
  --revision 7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3 \
  --dataset shape777 --limit 21 --mode direct --mode shared --runs 2 \
  --output results/native-shape-new
```

Local measurements and retained evidence are documented in the
[performance report](QWEN38_PERFORMANCE.md). These fixtures do not establish superiority over another
model, GGUF/llama.cpp, all workloads, or the published CUDA results. Conditional
option scores remain uncalibrated as decision confidence.
