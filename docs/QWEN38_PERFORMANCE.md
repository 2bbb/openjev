# Qwen3.8 native shared-prefix measurements

Measured 2026-09-18 on an Apple M5 Max, 128 GiB unified memory, macOS 26.4.
Use [the native backend guide](QWEN38_NATIVE.md) for installation and semantics.
These results describe this checkpoint and workload, not a universal fastest
model or backend. No GGUF/llama.cpp or Qwen3.5 comparison was performed.

## Throughput: one state, 21 decisions

The first 21 rows of the owned `shape777` fixture share an **1812-token prefix**;
full prompts are about 1900 tokens. Each mode received an untimed warmup,
followed by two timed runs (42 decisions). The table includes score-call,
synchronization and harness/output overhead, excluding model loading and warmup.

| Mode | Decisions/s | Mean wall time for 21 decisions | Peak MLX allocation |
| --- | ---: | ---: | ---: |
| Resident mlx-serve API, direct | 0.306 | 68.56 s | not measured by the client |
| Native, independent direct | 0.359 | 58.52 s | 75.69 GB |
| Native, shared prefix | **2.123** | **9.89 s** | **75.62 GB** |

Native shared throughput was **6.93× the API** and **5.92× independent native**
on this one state. The shared prefix took 2.48–2.92 s, cache branching about
0.01 s, and all 21 sequential suffixes about 6.90 s per run. This is prefix
reuse, not simultaneous suffix evaluation. Native prefill chunk: 2048.

The API was already resident with its ordinary 4 GiB RAM prefix cache enabled.
The native model was loaded once for both modes. Model weights used 72.91 GB of
MLX allocation; the n-gram table stayed mmap-backed. Observed native load times
were roughly 8–11 s, including runs with warm OS file caches; first-use kernel
compilation adds latency. A one-shot shell invocation must pay these costs.
Peak MLX allocation is not total process or system memory.

This is an indicative local measurement, not a controlled isolated-system
certification: small regression tests overlapped part of the API timing run.
Only one of 37 shape states was timed. Do not extrapolate the multiplier to all
777 decisions, long contexts, single short questions, or text generation.

## Quality and numerical agreement

The final native default uses compiled folded RMSNorm, chunk2048, exact-state
grouping, and independent direct scoring when a state has only one decision.

| Owned fixture | Rows | API correct | Native shared correct | API/native decisions agree |
| --- | ---: | ---: | ---: | ---: |
| authored144 | 144 | 136 | 136 | 144/144 |
| perturbations108 | 108 | 108 | 108 | 108/108 |
| Combined | 252 | 244 (96.83%) | 244 (96.83%) | 252/252 |

All rows produced finite option distributions; no API top-20 errors occurred on
these fixtures. Prompts, encoded input IDs and answer-slot IDs match. This small,
project-owned suite does not establish general task accuracy or calibrated
confidence. Both backends retain the same eight authored-fixture mistakes.

**Probabilities are not identical.** The largest per-row probability difference
was 0.211 on authored144 and 0.0291 on perturbations108. In the unlabelled
21-decision shape slice, native shared agreed with the API on all 21, while
independent native disagreed on one (`015c14ff8f2bce2886b3`): API P(yes)=0.6225,
native direct P(yes)=0.4688, shared P(yes)=0.5622. The native direct/shared maximum
probability difference was 0.0934. Thus changing prefill boundaries can change a
decision; shared scoring is not bitwise equivalent to an independent forward.
Neither shape answer is declared more correct because the fixture has no gold.

## Evidence and reproduction

Raw outputs, manifests, summaries, comparisons and SHA-256 hashes are retained
under [`results/qwen38flash/2026-09-18-native/`](../results/qwen38flash/2026-09-18-native/).
`RUN_NOTES.json` distinguishes initial experiments from the final default and
corrects the early API quality manifest's erroneous warm-timing description.
The early API quality run had no explicit warmup; its latency is not used in the
throughput table. `native-quality` is the earlier unfused/chunk256 baseline.

Checkpoint:
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, revision
`7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3`.
Runtime: MLX `0.32.2`, mlx-vlm commit
`10db092733a416439fce874cd96a7835b700f43d`, or mlx-serve `26.9.4`.
KV quantization is off; API requests disable MTP, draft/PLD, and sampling penalties.
The parent project previously checked all 113 file sizes and all 103 published
payload hashes. Scoring manifests record header hashes; they do not independently
attest complete payload SHA-256.

Run from the repository root, using a fresh output directory each time:

```sh
# Start the existing API server first; use its client environment.
python benchmarks/qwen38_native.py --backend mlx-serve \
  --model /path/to/Qwen3.8-Flash-Next \
  --revision 7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3 \
  --dataset shape777 --limit 21 --runs 2 --max-tokens 20000 \
  --output results/api-shape-new

# Stop the API server, then use the native environment.
python benchmarks/qwen38_native.py --backend qwen38-native \
  --model /path/to/Qwen3.8-Flash-Next \
  --revision 7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3 \
  --dataset shape777 --limit 21 --mode direct --mode shared \
  --runs 2 --max-tokens 20000 --prefill-chunk 2048 \
  --reference results/api-shape-new/mlx-serve/shape777-direct.jsonl \
  --output results/native-shape-new

python benchmarks/qwen38_native.py --backend qwen38-native \
  --model /path/to/Qwen3.8-Flash-Next \
  --revision 7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3 \
  --dataset authored144 --dataset perturbations108 \
  --mode shared --warmup 0 --max-tokens 20000 \
  --output results/native-quality-new
```

The final combined test suite passed **75 tests**. All **19 original published
raw-file hashes** match and `verify_published.py` still verifies **69 claims**.
Original headline results and their raw files were not altered. These integrity
checks are not a fresh CUDA run.
