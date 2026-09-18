"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, validate_row
from .direct import score as direct_score
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--backend", choices=("torch", "mlx", "mlx-serve", "qwen38-native"), default="torch")
    parser.add_argument("--prefill-chunk", type=int, help="Native Qwen3.8 prefill chunk (default: 2048)")
    parser.add_argument("--server-url", help="mlx-serve loopback base URL (default: http://127.0.0.1:18082)")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8), help="Quantize MLX weights in memory; default preserves source precision")
    parser.add_argument("--mlx-cache-limit-mib", type=int,
                        help="MLX inactive allocation cache in MiB (default: 256; 0 disables caching)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.mlx_bits and args.backend != "mlx":
        parser.error("--mlx-bits requires --backend mlx")
    if args.mlx_cache_limit_mib is not None:
        if args.backend != "mlx":
            parser.error("--mlx-cache-limit-mib requires --backend mlx")
        if args.mlx_cache_limit_mib < 0:
            parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.backend == "mlx" and args.mode == "reranker":
        parser.error("MLX supports direct, serial, and shared modes; reranker requires torch")
    if args.server_url is not None and args.backend != "mlx-serve":
        parser.error("--server-url requires --backend mlx-serve")
    if args.backend == "mlx-serve" and args.mode != "direct":
        parser.error("mlx-serve supports direct mode only; native serial/shared/reranker are unsupported")
    if args.backend == "qwen38-native" and args.mode not in ("direct", "shared"):
        parser.error("qwen38-native supports direct and shared modes only")
    if args.prefill_chunk is not None and (args.backend != "qwen38-native" or args.prefill_chunk < 1):
        parser.error("--prefill-chunk requires qwen38-native and a positive value")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    if args.backend in ("mlx-serve", "qwen38-native") and len({row["id"] for row in rows}) != len(rows):
        parser.error("Decision IDs must be unique")
    direct, serial, shared = direct_score, SerialPrefixScorer, score_shared
    if args.backend == "qwen38-native":
        from . import qwen38_native_backend

        model, tokenizer, metadata = qwen38_native_backend.load_model(
            args.model, args.revision, prefill_chunk=args.prefill_chunk or 2048)
        direct, shared = qwen38_native_backend.score, qwen38_native_backend.score_shared
    elif args.backend == "mlx-serve":
        from . import mlx_serve_backend

        model, tokenizer, metadata = mlx_serve_backend.load_model(
            args.model, args.revision, base_url=args.server_url or "http://127.0.0.1:18082")
        direct = mlx_serve_backend.score
    elif args.backend == "mlx":
        from . import mlx_backend

        cache_limit_mib = (mlx_backend.DEFAULT_CACHE_LIMIT_MIB if args.mlx_cache_limit_mib is None
                           else args.mlx_cache_limit_mib)
        model, tokenizer, metadata = mlx_backend.load_model(
            args.model, args.revision, args.mlx_bits, cache_limit_mib=cache_limit_mib)
        direct, serial, shared = mlx_backend.score, mlx_backend.SerialPrefixScorer, mlx_backend.score_shared
    else:
        model, tokenizer, metadata = load_causal_model(args.model, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            if args.backend == "qwen38-native":
                groups = {}
                for row in rows:
                    state_key = json.dumps(row["state"], ensure_ascii=False, allow_nan=False)
                    groups.setdefault(state_key, []).append(row)
                by_id = {}
                for group in groups.values():
                    results, timing = shared(model, tokenizer, group, metadata, args.max_tokens)
                    by_id.update({result["id"]: {**result, "shared_timing": timing} for result in results})
                results = [by_id[row["id"]] for row in rows]
            else:
                results, timing = shared(model, tokenizer, rows, metadata, args.max_tokens)
                results = [{**result, "shared_timing": timing} for result in results]
            for result in results:
                destination.write(json.dumps(result, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = serial(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct if args.mode == "direct" else reranker_score
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
