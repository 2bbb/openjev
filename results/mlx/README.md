# Apple Silicon evidence

These runs use an Apple M5 Max with 128 GiB unified memory, macOS 26.5.2,
Python 3.12.13, and the pinned Qwen/Qwen3.5-4B checkpoint recorded in every
manifest. The Mac also had an unrelated local-model service resident; it was
left running. Timings describe this machine and workload, not an isolated
cross-hardware comparison with the published CUDA results.

## Run history

| Directory | Status and purpose |
| --- | --- |
| `2026-09-17-bf16-pilot` | Initial seven-decision pilot on MLX-LM 0.31.3; review threshold frozen before full evaluation. Superseded runtime. |
| `2026-09-17-bf16` | Incomplete first run. Diagnostic and quality completed; process exited 137 during fresh shape scoring. No completed systems report. |
| `2026-09-17-bf16-bounded` | Complete run after limiting inactive MLX allocations to 256 MiB. Superseded runtime. Contains the before/after precision investigation. |
| `2026-09-17-q8` | Complete initial 8-bit experiment on MLX-LM 0.31.3. Superseded runtime; generated arrays were invalid. |
| `2026-09-17-bf16-fixed` | Complete BF16 evaluation with the pinned upstream normalization fix. |
| `2026-09-17-q8-fixed` | Complete 8-bit diagnostic, quality, and generation evaluation against BF16. |
| `2026-09-17-q4-fixed` | Complete 4-bit diagnostic, quality, and generation evaluation against BF16. |
| `2026-09-17-cli-smoke` | Installed CLI direct, serial, and shared modes on three retained fixture rows. |

The initial inactive-cache limit was approximately 122 GiB. The first run's
exit occurred while another model occupied substantial memory; memory pressure
is a likely explanation, not an independently confirmed OS diagnosis. With a
256 MiB inactive-cache limit, the full workload completed. The limit does not
cap active model or batch memory.

## Fixed-runtime BF16 results

All 252 quality decisions selected the same winning option as the published
Torch predictions: mean family-balanced accuracy was **81.32%** on authored144
and **77.99%** on perturbations108. Maximum probability differences from the
published predictions were 0.1052 and 0.0614 respectively. Prompt hashes and
input-token counts matched. One missing-evidence example selected a non-
insufficient answer with score at least 0.8.

The full shape workload is 37 states with 21 decisions per state. Each mode
has one warm measured pass over all 777 decisions, excluding model loading,
artifact hashing, initial warmup, and result writes. Timings include prompt
preparation and synchronized GPU execution through CPU readout.

| Mode | Total seconds | Decisions/s | Speedup vs fresh | Peak MLX GiB | Changed choices vs fresh |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 475.59 | 1.63 | 1.00x | 9.31 | 0 |
| serial | 66.84 | 11.62 | 7.12x | 9.29 | 3 |
| shared | 45.06 | 17.24 | 10.55x | 12.27 | 3 |

Serial and parallel modes each changed 3/777 choices. Every changed decision
involved a 0.5/0.5 tie in at least one execution shape. Maximum absolute
probability movement was 0.06713 for serial and 0.06028 for parallel. This is
measured speed with small numerical differences, not bit-identical reuse.
Peak MLX allocation includes model weights and intermediates, not total system
memory. The resident model itself used about 7.83 GiB; parallel batch
intermediates account for the higher peak.

The 21-decision compact-generation comparison produced valid JSON arrays in
all three repetitions and agreed with direct decisions on 18/21 questions.
Median direct shared scoring took **2.033 s**;
median generation took **2.718 s**
(1.34x as long). This separate short experiment
has its own timing variation; it is not the 777-decision throughput result.

Evidence: [quality](2026-09-17-bf16-fixed/quality.json),
[systems](2026-09-17-bf16-fixed/shape777.json),
[generation](2026-09-17-bf16-fixed/generation.json),
[manifest](2026-09-17-bf16-fixed/manifest.json).

## Precision investigation

The first complete run had 12 quality decisions exceeding the predeclared
0.05 probability-difference review threshold or changing their winning option.
The diagnostic selected exactly those cases and compared identical prompts and
source weights with PyTorch CPU FP32. This is an investigation of observed
failures, not a held-out quality evaluation.

MLX-LM 0.31.3 used RMS normalization with epsilon applied to a mean while the
reference L2 normalization applies epsilon to a sum. The pinned upstream
[fix](https://github.com/ml-explore/mlx-lm/commit/a63e24c389382619eb6d9af656e3b46024be217a)
corrects the scaling. The production dependency uses that maintained upstream
commit, with no local model fork or monkey patch.

| Maximum absolute probability difference over the selected 12 decisions | Original runtime | Fixed runtime |
| --- | ---: | ---: |
| Native MLX weights cast to FP32 vs CPU FP32 | 0.109083 | 0.009402 |
| Source RMSNorm weights folded in FP32 vs CPU FP32 | 0.101279 | 0.007598 |

Evidence: [original probe](2026-09-17-bf16-bounded/precision-probe.json),
[fixed-runtime probe](2026-09-17-bf16-bounded/precision-upstream-fixed.json).
The second probe selects cases from the same original run, then recomputes
MLX predictions using the fixed runtime. Per-prediction metadata distinguishes
runtime and diagnostic transformations. Remaining differences are measured;
this does not establish numerical identity between Metal and PyTorch.

## Quantization results

Both quantized models start from the same BF16 source and use affine weight
quantization, group size 64. All 252 direct outputs had valid distributions.
The accuracy columns are mean family-balanced accuracy; changed choices are
measured against fresh BF16 MLX scoring, not against the gold labels.

| Precision | Authored144 | Perturbations108 | Changed choices: authored / perturbations | Peak MLX GiB during quality scoring |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 81.32% | 77.99% | 0 / 0 | 8.12 |
| Q8 | 81.85% | 76.58% | 2 / 4 | 4.66 |
| Q4 | 78.92% | 79.89% | 14 / 6 | 2.82 |

8-bit changed 6/252 decisions, with maximum probability movement of 0.1761.
4-bit changed 20/252, with maximum movement of 0.6776. These small authored
fixtures do not justify a general quality ranking; quantization is a measured
memory/behavior tradeoff. BF16 remains the default. The 4-bit option is useful
for memory experiments but substantially changes some distributions.

Both quantized models generated only 20 answers for 21 questions in all three
compact-generation repetitions. Their generation timings are retained as
failed-completion evidence, not equivalent-answer performance. The direct
path returned all requested decisions. Quantized tests cover the seven-case
cache diagnostic, 252 quality cases, and three 21-decision timing repetitions;
the complete 777-decision systems comparison was run only for BF16.

Evidence: [8-bit quality](2026-09-17-q8-fixed/quality.json),
[4-bit quality](2026-09-17-q4-fixed/quality.json),
[8-bit generation](2026-09-17-q8-fixed/generation.json),
[4-bit generation](2026-09-17-q4-fixed/generation.json).

## Interpretation

Direct output is a distribution over the supplied choices. It guarantees the
output shape, not a correct decision or calibrated confidence. Changes in
precision, kernel shape, prefix splitting, and batch size can move scores and
change close decisions. All observed winning-option changes are retained in
comparison reports. Neither the authored benchmark nor the shape workload is
a general model-quality qualification.

See [MLX usage and reproduction](../../docs/MLX.md). Evidence directories are
create-only; partial and superseded runs are retained separately from the final
measurements. The original CUDA summary and raw evidence are unchanged.

## Validation and integrity

The installed CLI was exercised in direct, serial, and shared modes with the
real pinned checkpoint; each returned the three expected decision IDs and
finite normalized distributions. Commands, raw outputs, comparisons, and
hashes are retained in [CLI validation](2026-09-17-cli-smoke/validation.json).

Validation completed with 26 passing tests, including the native tiny hybrid
cache regressions and the recurrent normalization regression. All six completed
benchmark directories pass `benchmarks/verify_mlx.py`; the incomplete run is
explicitly excluded. Original CUDA checksums and all 69 published-summary checks
also pass. To check the retained Mac artifacts:

```bash
(cd results/mlx && shasum -a 256 -c SHA256SUMS)
python benchmarks/verify_mlx.py results/mlx/2026-09-17-bf16-fixed
python benchmarks/verify_mlx.py results/mlx/2026-09-17-q8-fixed
python benchmarks/verify_mlx.py results/mlx/2026-09-17-q4-fixed
```
