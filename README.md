# OpenJev

This fork's `qwen38flash` branch starts from Apple Silicon PR #2 and supports
Qwen3.8 Flash Next through [direct native loading and shared prefixes](docs/QWEN38_NATIVE.md)
or an [existing mlx-serve API](docs/QWEN38_FLASH.md). Both reuse the same packed checkpoint.
Start with [Quick start](#quick-start) for Qwen3.8. The original 4B/RTX 3090
demo and measurements below retain their upstream model and hardware scope.

<div align="center">

**Can we run something like Jev on a 3090 at home?**

**Wow! No waitlist.** [Run it in your browser today.](https://openjev.com)

[![Measured replay: typed decisions appear together while JSON streams token by token](demo/assets/openjev-phase1-replay.gif)](demo/index.html)

*Same frozen 4B model · same state · same 21 questions · measured separately, aligned at t=0 in the replay*

</div>

![Some AI company asks you to join a waitlist; OpenJev runs in your browser today](assets/openjev-no-waitlist.png)

Most agent decisions are small: *route this*, *retry that*, *does the evidence support X?* A chat model can answer them, but it spends time generating text that software immediately parses back into an `if` statement.

Jev is TypeSafe's closed service for runtime-defined semantic decisions. This project reproduces that **interface pattern** with open models; it does not reproduce Jev's undisclosed model or training.

This baseline reads typed option probabilities directly from a model. No answer sentence, JSON repair, or decoding loop.

## Quick start

This branch's Qwen3.8 backend is **`qwen38-native`**. Use an Apple Silicon Mac
and native arm64 Python **3.12**. The tested machine has **128 GiB unified memory**;
the complete model occupies **107.3 GB on disk**. See the
[setup guide](docs/QWEN38_NATIVE.md) for memory, free-disk and verification details.

```bash
git clone --branch qwen38flash --single-branch https://github.com/2bbb/openjev.git
cd openjev
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/qwen38-macos.txt -e '.[test,qwen38]'
```

Select the pinned checkpoint and its local directory:

```bash
MODEL_REPO="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
MODEL_REVISION="7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3"
MODEL_DIR="$PWD/models/Qwen3.8-Flash-Next"
```

For a first download, run:

```bash
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR" --max-workers 4
```

If that exact checkpoint already exists, **skip the download** and instead set
`MODEL_DIR="/absolute/path/to/Qwen3.8-Flash-Next"`. This can be on an external SSD;
no copy is needed. Follow the [integrity-check instructions](docs/QWEN38_NATIVE.md#3-verify-the-downloaded-or-reused-files)
for either path. The backend accepts this specific packed format, not GGUF or an
arbitrary MLX conversion.

Stop any server already holding this model, then run the unmodified upstream
examples with the Qwen3.8 backend:

```bash
openjev-score --backend qwen38-native --mode shared \
  --model "$MODEL_DIR" --revision "$MODEL_REVISION" \
  --input examples/decisions.jsonl \
  --output artifacts/qwen38-examples.jsonl
```

Expected winning option IDs are `yes`, `account_access`, and `not_required`.
Each result includes option scores, timing, the model revision, and prompt
fingerprints. Choose a new output path for every run.

`shared` groups identical states and reuses their prefix computation; the
remaining suffixes run as isolated sequential branches. These three examples
have different states, so each uses an independent forward. Use `--mode direct`
to explicitly score every row independently. Qwen3.8 supports `direct` and
`shared`; it does not implement `serial` or `reranker`.

Keep **`--backend qwen38-native`** in the command. The CLI default is still
Torch/CUDA for upstream compatibility; setting `CUDA_VISIBLE_DEVICES` does not
select MLX and is unnecessary for this native backend.

Other runtime/model paths have separate instructions:

- [Qwen3.5 on Apple Silicon](docs/MLX.md): `--backend mlx`, with its own model,
  dependency extra, and direct/serial/parallel-shared modes.
- [Original NVIDIA/CUDA reproduction](docs/REPRODUCE.md): the original
  Qwen3.5-4B and reranker baselines using Torch.
- [Qwen3.8 through a resident mlx-serve API](docs/QWEN38_FLASH.md):
  `--backend mlx-serve --mode direct`.

## How it works

```mermaid
flowchart LR
    S[Unstructured state] --> M[Loaded language model]
    C[Runtime criteria] --> M
    O[Typed options] --> M
    M -- native option logits --> P[Probabilities]
```

- **Runtime-defined:** criteria and option descriptions arrive with the request.
- **Decision-native:** one forward pass reads declared option logits; no answer token is sampled.
- **Shared-state aware:** one long state can be prefetched once, then branched across many criteria.
- **Auditable:** the owned fixture, exact runners, row-level outputs, revisions, prompts, and known failures are committed.

## Speed

The tables below retain the original **Qwen3.5-4B / RTX 3090** results. For this
branch's Qwen3.8 / Apple Silicon measurements, see [Qwen3.8 performance](docs/QWEN38_PERFORMANCE.md).

### Decisions versus a compact generated array

Same frozen Qwen3.5-4B, same owned state, same 21 binary criteria, one RTX 3090:

| Output path | Time | Output tokens | Result |
|---|---:|---:|---|
| Direct typed logits, median of 3 | **1.023 s** | **0** | 21 probability pairs |
| Autoregressive JSON array, median of 3 | 5.332 s | 111 | Valid ordered 21-value array |

The compact generative baseline emits only ordered `"yes"`/`"no"` values—no keys, confidence objects, or explanations. Its median first-token time was 0.489 s, but completing the array took **5.21×** as long as direct readout. All three arrays were valid and identical. Their choices agreed with direct argmax on 18/21 criteria, so this is a systems comparison rather than a claim that the two readouts are semantically equivalent. [Exact prompt, outputs, token timeline, and runs](results/raw/decision-vs-compact-array.json) are committed.

### Reusing a state across 21 decisions

On an owned 37-state × 21-criterion workload:

| Execution path | Decisions/s | 777 decisions |
|---|---:|---:|
| Fresh direct scoring | 2.33 | 333.1 s |
| Serial prefix reuse | 10.75 | 72.3 s |
| Parallel suffixes | **20.03** | **38.8 s** |
| Native reranker | 1.86 | 417.3 s |

The owned [37×21 fixture](benchmarks/data/shape777.jsonl), [direct/reuse runner](benchmarks/shape777.py), [reranker runner](benchmarks/shape777_reranker.py), [raw timings](results/raw/shape777-direct.json), and [row-level predictions](results/raw/shape777-direct.predictions.jsonl) are included. The fast reuse paths are experimental: BF16 execution changed 5–6 of 777 argmaxes relative to fresh scoring.

## Quality

These are the original frozen-model results. The separate
[Qwen3.8 evaluation](docs/QWEN38_PERFORMANCE.md#quality-and-numerical-agreement)
reports this branch's native backend.

| Frozen workload | Rows | Direct logits | Native reranker | Published Jev |
|---|---:|---:|---:|---:|
| Authored decisions, balanced accuracy | 144 | **0.813** | 0.625 | — |
| WANLI, balanced accuracy | 256 | **0.637** | 0.522 | — |
| TypeSafe selected subset, modal agreement | 102 across 20 cases | **0.845** | 0.560 | 0.883 |
| Every judgment grid, accuracy | 36 | **0.806** | 0.694 | — |

The reranker remained strong at retrieval ranking, but direct logits were the better general-decision baseline.

The Jev number is read from TypeSafe's published records; we did not run a live Jev endpoint. The comparison covers the 102 rows that could be aligned from public artifacts, not TypeSafe's reported 711-row aggregate.

## Input

```json
{
  "id": "route-1",
  "state": "Customer cannot access an account after a password reset.",
  "question": "Which queue should handle this request?",
  "options": [
    {"id": "access", "description": "Account access support."},
    {"id": "billing", "description": "Billing support."}
  ]
}
```

Returned probabilities are conditional on the supplied options. Calibrate and validate them on the workload where they will make decisions.
`state` may also be a nonempty JSON object or array. Direct modes preserve it as structured JSON; reranker mode renders it as document text.

## Documentation

- [Results](docs/RESULTS.md) — quality, speed, perturbations, and claim boundaries
- [Method](docs/METHOD.md) — frozen prompts, metrics, and timing scope
- [Reproduce](docs/REPRODUCE.md) — exact environment, pinned commands, perturbations, and verification
- [Interactive replay](demo/index.html)
- [Browser-only WebGPU demo](webgpu-demo/index.html) — no waitlist; use it today
- [Machine-readable summary](results/phase1-summary.json)
- [Benchmark bundle](benchmarks/README.md) — fixtures, runners, selection IDs, and reproduction commands
- [Raw results and checksums](results/raw/)
- [Third-party sources](THIRD_PARTY.md)

## Evaluation sources

- [TypeSafe public evaluations](https://evals.typesafe.ai/) — public comparison cases used for selected-subset agreement
- [Every parallel judgment lab](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/) and its [downloadable experiment data](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/experiments.json)
- [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) — external natural-language inference check
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) — frozen baseline models

This is an independent research project. Model weights and third-party records without a redistribution grant are excluded; immutable selection IDs and fetch manifests are included. Upstream models retain their licenses. Project code is released under the [MIT License](LICENSE).
