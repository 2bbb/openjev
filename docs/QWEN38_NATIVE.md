# Native Qwen3.8 scoring on Apple Silicon

This branch adds `--backend qwen38-native` alongside the resident `mlx-serve`
bridge. It directly reads the **same existing packed checkpoint**; it does not
convert or duplicate weights on disk. Text decisions only, with zero generated
tokens and no API top-20 restriction.

## Standalone installation: start here

These steps use only this public fork; they do not require the author's parent
`qwens` project, personal paths, or an existing mlx-serve installation.

Requirements: Apple Silicon, Git, and native arm64 Python **3.12** (not Rosetta).
The tested machine has **128 GiB unified memory**. The loader needs roughly
72.9 GB of weight storage in memory plus at least 5 GiB of available headroom;
64 GB machines cannot fit this pack. The complete download is **107.3 GB**
(about 100 GiB). Plan roughly **150–160 GB free disk** for a fresh setup,
including Python dependencies, download working space and headroom. The 32 GB
n-gram table stays read-only mmap; it is not all allocated on the GPU.

### 1. Clone the fork and install the environment

```sh
git clone --branch qwen38flash --single-branch https://github.com/2bbb/openjev.git
cd openjev
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/qwen38-macos.txt -e '.[test,qwen38]'
```

The requirements file preserves the tested macOS/Python 3.12 dependency versions.
It installs the `hf` download CLI and `openjev-score` in this virtual environment.
The backend enforces MLX `0.32.2` and [mlx-vlm commit
`10db092733a416439fce874cd96a7835b700f43d`](https://github.com/Blaizzy/mlx-vlm/tree/10db092733a416439fce874cd96a7835b700f43d/mlx_vlm/models/qwen4_exp).
For later shells, return to this clone and run `. .venv/bin/activate` again.

### 2. Download the model OR point to an existing copy

The supported, tested pack is
[`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/tree/7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3).
Use this exact revision for the commands below:

```sh
MODEL_REPO="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
MODEL_REVISION="7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3"
```

**A. First download:** choose where to store the complete checkpoint. This may
be outside the repository, including an external SSD. Keep the quotes for paths
with spaces. Do not limit the download to `*.safetensors`: the tokenizer,
configuration and `ngram_table.bin` are also required.

```sh
MODEL_DIR="$PWD/models/Qwen3.8-Flash-Next"
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR" --max-workers 4
```

Rerun the same command after an interruption; already downloaded, up-to-date
files are reused. Keep the download metadata under `.cache/huggingface` inside
the model directory. See the [official Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).
The downloaded model has its own [Qwen Community License](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/blob/7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3/LICENSE),
separate from the code license.

**B. Already downloaded:** skip the download command and set the existing
**directory** instead. No copying, symlink or model registration is necessary.
An existing Hugging Face snapshot directory also works if all its files resolve.

```sh
MODEL_DIR="/Volumes/Models/Qwen3.8-Flash-Next"
```

Point to the folder containing `config.json`, tokenizer assets,
`model-00001.safetensors` through `model-00100.safetensors`, and
`ngram_table.bin`. Do not pass a single shard or a GGUF file. This custom pack's
layout matters: another model with a similar name, an ordinary MLX conversion,
or Qwen3.5 GGUF cannot be substituted into `qwen38-native`.

### 3. Verify the downloaded or reused files

Run this for either path above. It checks local files against the pinned Hub
revision, requires the complete download, needs network access for metadata,
and reads roughly 107 GB from disk. It does not load the model onto the GPU.

```sh
hf cache verify "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR" --fail-on-missing-files
```

A mismatch or missing file must be resolved before scoring. The scorer's
`--revision` argument records provenance; it does not download files, switch an
existing directory to that revision, or verify payload checksums for you.

### 4. Run the sample, then your own decisions

Stop any server holding this model first. The standalone OpenJev CLI does not
start or stop another process. Stay in the repository root with `.venv` active:

```sh
openjev-score --backend qwen38-native --mode shared \
  --model "$MODEL_DIR" --revision "$MODEL_REVISION" \
  --input examples/qwen38flash-decisions.jsonl \
  --output artifacts/qwen38-sample.jsonl
```

The sample's winning option IDs should be `access`, `access`, then `billing`.
Use a fresh output name on every run. For your own JSONL input, replace the
`--input` path; the sample file shows the `id`, `state`, `question`, and `options`
format. `--model` always accepts the local directory you selected; the backend
never downloads weights automatically.

The `bash openjev.sh` convenience wrapper mentioned in the author's local setup
belongs to the separate `qwens` parent project and is not part of this fork.
The standalone commands above are the supported entry point for other users.

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
