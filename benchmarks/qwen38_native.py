"""Create-only Qwen3.8 backend comparison harness.

The runner intentionally uses only project-owned fixtures. It can compare the
resident mlx-serve bridge against a native backend module that exposes:

    load_model(source, revision)
    score(model, tokenizer, row, metadata, max_tokens)
    score_shared(model, tokenizer, rows, metadata, max_tokens)

Run from the repository root with a fresh output directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import evaluate
from openjev_phase1.direct import encode_prompt


DATASETS = {
    "authored144": Path("benchmarks/data/authored144.jsonl"),
    "perturbations108": Path("benchmarks/data/perturbations108.jsonl"),
    "shape777": Path("benchmarks/data/shape777.jsonl"),
}
BACKENDS = {
    "mlx-serve": "openjev_phase1.mlx_serve_backend",
    "qwen38-native": "openjev_phase1.qwen38_native_backend",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose(row: dict[str, Any]) -> str | None:
    values = row.get("probabilities")
    ids = row.get("option_ids")
    if not isinstance(values, list) or not isinstance(ids, list) or len(values) != len(ids):
        return None
    return ids[max(range(len(values)), key=values.__getitem__)]


def finite_distribution(row: dict[str, Any]) -> bool:
    values = row.get("probabilities")
    return (
        isinstance(values, list)
        and len(values) >= 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
                for value in values)
        and abs(sum(values) - 1) <= 1e-4
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def summarize_latencies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row if key.endswith("_seconds")})
    summary: dict[str, Any] = {}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = {
                "n": len(values),
                "sum": sum(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p90": percentile(values, 0.90),
                "max": max(values),
            }
    return summary


def uses_mlx_runtime(module: Any) -> bool:
    return module.__name__.endswith((".mlx_backend", ".qwen38_native_backend"))


def memory_snapshot(enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None
    try:
        return {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        }
    except Exception as exc:  # pragma: no cover - depends on optional MLX runtime.
        return {"error": str(exc)}


def reset_peak_memory(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import mlx.core as mx
    except ImportError:
        return
    try:
        mx.reset_peak_memory()
    except Exception:
        return


def sync_accelerator(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import mlx.core as mx
    except ImportError:
        return
    try:
        mx.synchronize()
    except Exception:
        return


def prompt_fingerprint(tokenizer: Any, row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    return {
        "answer_token_ids": slots,
        "input_tokens": len(ids),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "prompt_sha256": prompt_hash,
    }


def error_row(row: dict[str, Any], exc: Exception, tokenizer: Any, max_tokens: int) -> dict[str, Any]:
    result = {
        "id": row.get("id"),
        "option_ids": [option.get("id") for option in row.get("options", []) if isinstance(option, dict)],
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    try:
        result.update(prompt_fingerprint(tokenizer, row, max_tokens))
    except Exception as prompt_exc:
        result["prompt_error"] = str(prompt_exc)
        result["prompt_error_type"] = type(prompt_exc).__name__
    return result


def limited(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows if limit is None else rows[:limit]


def selected_groups(rows: list[dict[str, Any]], group_limit: int | None) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # Quality fixtures use group_id for a source and its changed evidence.
        # Match the native CLI: only exactly serialized states share a cache.
        state_key = json.dumps(row["state"], ensure_ascii=False, allow_nan=False)
        groups[state_key].append(row)
    ordered = list(groups.values())
    return ordered if group_limit is None else ordered[:group_limit]


def load_backend(name: str):
    return importlib.import_module(BACKENDS.get(name, name))


def backend_label(name: str) -> str:
    return name.rsplit(".", 1)[-1].replace("_backend", "")


def backend_run_labels(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    labels = []
    for name in names:
        base = backend_label(name)
        counts[base] = counts.get(base, 0) + 1
        labels.append(base if counts[base] == 1 else f"{base}-{counts[base]}")
    return labels


def load_model(module: Any, args: argparse.Namespace):
    kwargs: dict[str, Any] = {}
    if args.base_url and module.__name__.endswith("mlx_serve_backend"):
        kwargs["base_url"] = args.base_url
    if module.__name__.endswith("qwen38_native_backend"):
        kwargs["prefill_chunk"] = args.prefill_chunk
        kwargs["cache_limit_mib"] = args.cache_limit_mib
    return module.load_model(args.model, args.revision, **kwargs)


def score_one(module: Any, model: Any, tokenizer: Any, metadata: dict[str, Any],
              row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    sync_accelerator(uses_mlx_runtime(module))
    started = time.perf_counter()
    try:
        result = module.score(model, tokenizer, row, metadata, max_tokens=max_tokens)
    except Exception as exc:
        result = error_row(row, exc, tokenizer, max_tokens)
    sync_accelerator(uses_mlx_runtime(module))
    result["harness_wall_seconds"] = time.perf_counter() - started
    result["harness_mode"] = "direct"
    result["harness_status"] = "ok" if not result.get("error") else "error"
    return result


def score_group_shared(module: Any, model: Any, tokenizer: Any, metadata: dict[str, Any],
                       rows: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not hasattr(module, "score_shared"):
        raise ValueError(f"{module.__name__} does not expose score_shared")
    sync_accelerator(uses_mlx_runtime(module))
    started = time.perf_counter()
    try:
        outputs, timing = module.score_shared(model, tokenizer, rows, metadata, max_tokens=max_tokens)
        by_id = {row["id"]: row for row in outputs}
        results = [by_id.get(row["id"], error_row(row, ValueError("missing shared output"), tokenizer, max_tokens))
                   for row in rows]
        group_error = None
    except Exception as exc:
        timing = {}
        group_error = str(exc)
        results = [error_row(row, exc, tokenizer, max_tokens) for row in rows]
    sync_accelerator(uses_mlx_runtime(module))
    wall = time.perf_counter() - started
    for result in results:
        result["harness_wall_seconds"] = wall / len(rows)
        result["harness_group_wall_seconds"] = wall
        result["harness_mode"] = "shared"
        result["harness_status"] = "ok" if not result.get("error") else "error"
    return results, {"group_id": rows[0]["group_id"], "rows": len(rows), "wall_seconds": wall,
                     "backend_timing": timing, "error": group_error}


def warmup_dataset(module: Any, model: Any, tokenizer: Any, metadata: dict[str, Any],
                   mode: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if args.warmup == 0 or not rows:
        return records
    if mode == "direct":
        warm_rows = rows[:min(args.warmup, len(rows))]
        for index, row in enumerate(warm_rows):
            result = score_one(module, model, tokenizer, metadata, row, args.max_tokens)
            records.append({"warmup_index": index, "mode": mode, "id": row["id"],
                            "status": result["harness_status"], "error": result.get("error"),
                            "wall_seconds": result["harness_wall_seconds"]})
        return records
    groups = selected_groups(rows, args.group_limit)
    for repeat in range(args.warmup):
        for group in groups:
            scored, timing = score_group_shared(module, model, tokenizer, metadata, group, args.max_tokens)
            records.append({"warmup_index": repeat, "mode": mode, "group_id": group[0]["group_id"],
                            "rows": len(group), "ok_rows": sum(row.get("harness_status") == "ok" for row in scored),
                            "error_rows": sum(row.get("harness_status") != "ok" for row in scored),
                            "error": timing.get("error"), "wall_seconds": timing["wall_seconds"]})
    return records


def score_dataset(module: Any, model: Any, tokenizer: Any, metadata: dict[str, Any],
                  dataset: str, mode: str, rows: list[dict[str, Any]],
                  output: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_path = output / f"{dataset}-{mode}.jsonl"
    group_timings: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    warmup = warmup_dataset(module, model, tokenizer, metadata, mode, rows, args)
    reset_peak_memory(uses_mlx_runtime(module))
    started = time.perf_counter()
    if mode == "direct":
        for run in range(args.runs):
            for row in rows:
                scored = score_one(module, model, tokenizer, metadata, row, args.max_tokens)
                scored["harness_run"] = run
                append_jsonl(out_path, [scored])
                all_rows.append(scored)
    elif mode == "shared":
        for run in range(args.runs):
            for group in selected_groups(rows, args.group_limit):
                scored, timing = score_group_shared(module, model, tokenizer, metadata, group, args.max_tokens)
                for row in scored:
                    row["harness_run"] = run
                timing["harness_run"] = run
                append_jsonl(out_path, scored)
                all_rows.extend(scored)
                group_timings.append(timing)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    elapsed = time.perf_counter() - started
    ok_rows = [row for row in all_rows if row.get("harness_status") == "ok"]
    return {
        "dataset": dataset,
        "mode": mode,
        "runs": args.runs,
        "warmup": {"runs": args.warmup, "timed": False, "records": warmup},
        "rows_attempted": len(all_rows),
        "rows_ok": len(ok_rows),
        "rows_error": len(all_rows) - len(ok_rows),
        "wall_seconds": elapsed,
        "decisions_per_second": len(ok_rows) / elapsed if elapsed else None,
        "latency": summarize_latencies(ok_rows),
        "group_timings": group_timings,
        "memory": memory_snapshot(uses_mlx_runtime(module)),
        "output": str(out_path),
    }


def evaluate_if_gold(dataset: str, gold: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not gold or "label" not in gold[0]:
        return None
    first_run = [row for row in predictions if row.get("harness_run") in (None, 0)]
    return evaluate.evaluate(gold, first_run)


def filter_run(rows: list[dict[str, Any]], run: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("harness_run", 0) == run]
    return selected


def compare_predictions(left: list[dict[str, Any]], right: list[dict[str, Any]],
                        *, left_run: int = 0, right_run: int = 0) -> dict[str, Any]:
    left = filter_run(left, left_run)
    right = filter_run(right, right_run)
    a = evaluate.indexed(left, "left")
    b = evaluate.indexed(right, "right")
    common = sorted(a.keys() & b.keys())
    flips = []
    probability_deltas = []
    prompt_mismatches = []
    token_mismatches = []
    slot_mismatches = []
    for key in common:
        if a[key].get("prompt_sha256") != b[key].get("prompt_sha256"):
            prompt_mismatches.append(key)
        if a[key].get("input_ids_sha256") != b[key].get("input_ids_sha256"):
            token_mismatches.append(key)
        if a[key].get("answer_token_ids") != b[key].get("answer_token_ids"):
            slot_mismatches.append(key)
        left_choice, right_choice = choose(a[key]), choose(b[key])
        if left_choice != right_choice:
            flips.append({"id": key, "left": left_choice, "right": right_choice})
        if finite_distribution(a[key]) and finite_distribution(b[key]):
            left_probs = dict(zip(a[key]["option_ids"], a[key]["probabilities"]))
            right_probs = dict(zip(b[key]["option_ids"], b[key]["probabilities"]))
            if left_probs.keys() == right_probs.keys():
                probability_deltas.append(max(abs(left_probs[item] - right_probs[item]) for item in left_probs))
    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "left_run": left_run,
        "right_run": right_run,
        "run_policy": "matched harness_run values; default compares run 0 only",
        "common_rows": len(common),
        "agreement": (len(common) - len(flips)) / len(common) if common else None,
        "argmax_flips": flips,
        "prompt_mismatches": prompt_mismatches,
        "token_mismatches": token_mismatches,
        "slot_mismatches": slot_mismatches,
        "max_probability_delta": max(probability_deltas) if probability_deltas else None,
        "mean_probability_delta": statistics.mean(probability_deltas) if probability_deltas else None,
    }


def read_reference(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    return read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", choices=sorted(BACKENDS), required=True,
                        help="Backend to run. Use separate processes and --reference when switching between resident API and native weights; this harness does not stop servers.")
    parser.add_argument("--model", required=True, help="Local checkpoint directory or backend-supported source.")
    parser.add_argument("--revision", required=True, help="Pinned revision or explicit local manifest label.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18082",
                        help="Loopback mlx-serve URL for --backend mlx-serve.")
    parser.add_argument("--prefill-chunk", type=int, default=2048,
                        help="Native Qwen3.8 prefill chunk size; passed only to qwen38-native.")
    parser.add_argument("--cache-limit-mib", type=int, default=256,
                        help="Native MLX inactive allocation cache in MiB; passed only to qwen38-native.")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS), default=None,
                        help="Dataset to run; repeatable. Default: authored144, perturbations108, shape777.")
    parser.add_argument("--mode", action="append", choices=("direct", "shared"), default=None,
                        help="Scoring mode; repeatable. Default: direct.")
    parser.add_argument("--limit", type=int, help="Maximum rows per dataset before grouping.")
    parser.add_argument("--group-limit", type=int, help="Maximum shared-state groups for shared mode.")
    parser.add_argument("--runs", type=int, default=1, help="Timed repetitions after model load.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Untimed warmup operations before each dataset/mode. Direct warms up this many rows; shared warms up each selected group this many times. Outputs are not written.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reference", type=Path, help="Prior prediction JSONL for agreement comparison.")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; create-only.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.group_limit is not None and args.group_limit <= 0:
        parser.error("--group-limit must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.prefill_chunk <= 0:
        parser.error("--prefill-chunk must be positive")
    if args.cache_limit_mib < 0:
        parser.error("--cache-limit-mib must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)

    datasets = args.dataset or ["authored144", "perturbations108", "shape777"]
    modes = args.mode or ["direct"]
    source_rows = {name: limited(read_jsonl(DATASETS[name]), args.limit) for name in datasets}
    reference = read_reference(args.reference)
    manifest = {
        "created_at_epoch": time.time(),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "fixtures": {name: {"path": str(path), "sha256": sha256(path), "rows_selected": len(source_rows[name])}
                     for name, path in DATASETS.items() if name in source_rows},
        "source_sha256": {
            "benchmarks/qwen38_native.py": sha256(Path("benchmarks/qwen38_native.py")),
            "benchmarks/evaluate.py": sha256(Path("benchmarks/evaluate.py")),
            "src/openjev_phase1/qwen38_native_backend.py": sha256(Path("src/openjev_phase1/qwen38_native_backend.py")),
            "src/openjev_phase1/mlx_serve_backend.py": sha256(Path("src/openjev_phase1/mlx_serve_backend.py")),
        },
        "timing_scope": "Timed runs after model load and after the configured untimed warmup. Per-row harness wall includes backend score call and MLX synchronization for native MLX backends. qwen38-native shared mode uses one prefill followed by isolated sequential suffix branches.",
        "outputs_are_create_only": True,
    }
    write_json(args.output / "manifest.json", manifest)

    reports: dict[str, Any] = {}
    predictions_by_key: dict[str, list[dict[str, Any]]] = {}
    for backend_name, key in zip(args.backend, backend_run_labels(args.backend)):
        module = load_backend(backend_name)
        model, tokenizer, metadata = load_model(module, args)
        backend_output = args.output / key
        backend_output.mkdir()
        reports[key] = {"metadata": metadata, "datasets": {}}
        for dataset in datasets:
            rows = source_rows[dataset]
            for mode in modes:
                report_key = f"{dataset}-{mode}"
                summary = score_dataset(module, model, tokenizer, metadata, dataset, mode, rows, backend_output, args)
                predictions = read_jsonl(Path(summary["output"]))
                summary["quality"] = evaluate_if_gold(dataset, rows, predictions)
                if reference is not None:
                    summary["vs_reference"] = compare_predictions(reference, predictions)
                reports[key]["datasets"][report_key] = summary
                predictions_by_key[f"{key}:{report_key}"] = predictions
        del model, tokenizer
        gc.collect()
        if uses_mlx_runtime(module):
            import mlx.core as mx
            mx.synchronize()
            mx.clear_cache()

    comparisons: dict[str, Any] = {}
    labels = list(predictions_by_key)
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            left_dataset_mode = left.split(":", 1)[1]
            if left_dataset_mode == right.split(":", 1)[1]:
                comparisons[f"{left} vs {right}"] = compare_predictions(predictions_by_key[left], predictions_by_key[right])
    write_json(args.output / "summary.json", {"reports": reports, "comparisons": comparisons})


if __name__ == "__main__":
    main()
