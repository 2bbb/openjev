"""Direct, text-only scoring of the existing mlx-serve Qwen3.8 packed weights.

Use pinned mlx-vlm computation, preserve packed RMSNorm weights, infer each
affine quantizer from its stored shape, and mmap the merged n-gram table.
No HTTP requests, weight conversion on disk, or generated tokens are involved.
"""
from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from importlib.metadata import distribution, version
import json
from pathlib import Path
import struct
import tempfile
import time

from .core import softmax
from .direct import PROMPT_VERSION, encode_prompt
from .shared import _state_prefix

VLM_REVISION = "10db092733a416439fce874cd96a7835b700f43d"


def runtime_source():
    source = json.loads(distribution("mlx-vlm").read_text("direct_url.json") or "null")
    if not source or source.get("vcs_info", {}).get("commit_id") != VLM_REVISION:
        raise RuntimeError(f"Native backend requires mlx-vlm git revision {VLM_REVISION}")
    if version("mlx") != "0.32.2":
        raise RuntimeError("Native backend requires the verified MLX 0.32.2 runtime")
    return source


def tensor_header(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 1 <= size <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(size))
    return {k: v for k, v in header.items() if k != "__metadata__"}, size + 8


def ple_manifest(path, config):
    """Describe the existing merged table as one read-only upstream mmap shard."""
    table = config.get("ngram_table", {})
    if table != {"file": "ngram_table.bin", "bits": 4, "group_size": 32}:
        raise ValueError("Expected the mlx-serve merged affine Q4/group32 n-gram pack")
    file = path / "ngram_table.bin"
    header, base = tensor_header(file)
    if set(header) != {"weight", "scales", "biases"}:
        raise ValueError("Unexpected n-gram tensors")
    rows, packed_width = header["weight"]["shape"]
    entry = {"row_start": 0, "row_count": rows}
    for name, dtype in (("weight", "U32"), ("scales", "BF16"), ("biases", "BF16")):
        desc = header[name]
        if desc["dtype"] != dtype or desc["data_offsets"][1] + base > file.stat().st_size:
            raise ValueError(f"Invalid n-gram descriptor: {name}")
        entry[name] = {"file": file.name, "offset": base + desc["data_offsets"][0],
                       "shape": desc["shape"], "dtype": dtype}
    return {"version": 2, "source_root": str(path), "row_width": packed_width * 8,
            "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
            "cache_rows": 0, "shards": [entry]}


def quantizer_for(header, name, width):
    if name + ".scales" not in header:
        return False
    weight, scales = header[name + ".weight"], header[name + ".scales"]
    if weight["dtype"] != "U32" or name + ".biases" not in header:
        raise ValueError(f"Unsupported affine packing for {name}")
    packed = weight["shape"][-1] * 32
    groups = scales["shape"][-1]
    if packed % width or width % groups:
        raise ValueError(f"Inconsistent quantized tensor shape for {name}")
    bits, group_size = packed // width, width // groups
    if bits not in (4, 8) or group_size != 64:
        raise ValueError(f"Unsupported quantizer for {name}: {bits}/{group_size}")
    return {"bits": bits, "group_size": group_size, "mode": "affine"}


def text_weight(key):
    return key.startswith(("language_model.model.", "language_model.lm_head."))


@lru_cache(maxsize=None)
def packed_norm(group_size, eps):
    import mlx.core as mx

    @mx.compile
    def normalize(x, weight):
        y, weight = x.astype(mx.float32), weight.astype(mx.float32)
        if group_size is not None:
            y = y.reshape(*y.shape[:-1], -1, group_size)
            weight = weight.reshape(-1, group_size)
        y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
        return (y * weight).reshape(x.shape).astype(x.dtype)

    return normalize


def load_model(source, revision, *, prefill_chunk=2048, cache_limit_mib=256):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm.models.qwen4_exp import LanguageModel, TextConfig
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm
    from transformers import AutoTokenizer

    vlm_source = runtime_source()
    path = Path(source).resolve()
    if not path.is_dir() or not revision:
        raise ValueError("Native packed loading requires a local directory and revision label")
    if prefill_chunk < 1 or cache_limit_mib < 0:
        raise ValueError("prefill_chunk must be positive and cache_limit_mib nonnegative")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or config.get("model_file"):
        raise ValueError("Expected native qwen4_exp, without custom model code")
    if not mx.metal.is_available():
        raise RuntimeError("Native Qwen3.8 scoring requires Apple Silicon / Metal")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(cache_limit_mib * 1024 * 1024)
    started = time.perf_counter()
    files = sorted(path.glob("model-[0-9]*.safetensors"))
    if not files:
        raise ValueError("No numbered mlx-serve weight shards found")
    headers, file_keys, artifacts = {}, {}, {}
    for file in files:
        header, _ = tensor_header(file)
        mapped = {key.removeprefix("language_model."): desc for key, desc in header.items()
                  if text_weight(key)}
        if set(headers) & set(mapped):
            raise ValueError("Duplicate model tensor keys")
        headers.update(mapped)
        file_keys[file] = mapped
        artifacts[file.name] = {"bytes": file.stat().st_size,
            "header_sha256": hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()}
    import psutil
    weight_bytes = sum(d["data_offsets"][1] - d["data_offsets"][0] for d in headers.values())
    required = weight_bytes + 5 * 1024**3
    if psutil.virtual_memory().available < required:
        raise MemoryError(f"Need approximately {required / 1024**3:.1f} GiB available; "
                          "stop the other model server before native loading")
    manifest = ple_manifest(path, config)
    temporary = tempfile.TemporaryDirectory(prefix="openjev-qwen38-")
    manifest_path = Path(temporary.name) / "ple.json"
    manifest_path.write_text(json.dumps(manifest))
    text_config = dict(config["text_config"])
    text_config["ple_storage"] = {"manifest": str(manifest_path), "cache_rows": 0}
    lm = LanguageModel(TextConfig.from_dict(text_config))

    # mlx-serve has already folded (1+w). Keep those values exactly; subtracting
    # and adding one in BF16 would introduce an unnecessary second rounding.
    class PackedRMSNorm(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.weight = original.weight
            self.eps, self.group_size = original.eps, original.group_size

        def __call__(self, x):
            return packed_norm(self.group_size, self.eps)(x, self.weight)

    replacements = [(name, PackedRMSNorm(module)) for name, module in lm.named_modules()
                    if isinstance(module, Qwen4ExpRMSNorm)]
    lm.update_modules(tree_unflatten(replacements))
    quantizers = {}

    def predicate(name, module):
        if name + ".scales" not in headers:
            return False
        params = quantizer_for(headers, name, module.weight.shape[-1])
        quantizers[name] = dict(params)
        return params

    nn.quantize(lm, class_predicate=predicate)
    expected = dict(tree_flatten(lm.parameters()))
    if set(headers) != set(expected):
        raise ValueError(f"Packed model keys differ: missing={sorted(set(expected)-set(headers))}, "
                         f"unexpected={sorted(set(headers)-set(expected))}")
    for name, value in expected.items():
        if list(value.shape) != headers[name]["shape"]:
            raise ValueError(f"Shape mismatch for {name}: {value.shape} vs {headers[name]['shape']}")
    del expected
    # Evaluate one shard at a time. No duplicate full unquantized model or
    # n-gram payload is ever materialized. MTP/vision shards are excluded.
    for file, keys in file_keys.items():
        if not keys:
            continue
        loaded = mx.load(str(file))
        weights = [(key.removeprefix("language_model."), value) for key, value in loaded.items()
                   if text_weight(key)]
        if {name for name, _ in weights} != set(keys):
            raise ValueError(f"Loaded shard keys differ from inspected header: {file}")
        for name, value in weights:
            if list(value.shape) != keys[name]["shape"]:
                raise ValueError(f"Loaded tensor shape differs: {name}")
        lm.load_weights(weights, strict=False)
        mx.eval([value for _, value in weights])
        del loaded, weights
    lm.eval()
    mx.synchronize()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = NativeModel(lm, temporary, prefill_chunk)
    metadata = {"source": str(path), "revision": revision, "backend": "qwen38-native-v1",
                "mlx_version": version("mlx"), "mlx_vlm_version": version("mlx-vlm"),
                "mlx_vlm_source": vlm_source,
                "mlx_vlm_expected_revision": VLM_REVISION,
                "quantizers": {str(bits): sum(q["bits"] == bits for q in quantizers.values()) for bits in (4, 8)},
                "folded_norms": len(replacements), "ngram_storage": "mmap existing merged Q4 table",
                "norm_execution": "compiled folded-weight RMSNorm",
                "prefill_chunk": prefill_chunk, "load_seconds": time.perf_counter() - started,
                "weight_artifacts": artifacts,
                "weight_identity": "header hashes and caller revision; payload SHA not recomputed",
                "memory_active_bytes": mx.get_active_memory(), "generated_tokens": 0}
    return model, tokenizer, metadata


class NativeModel:
    def __init__(self, language_model, temporary, prefill_chunk):
        self.lm, self.temporary, self.prefill_chunk = language_model, temporary, prefill_chunk

    def prefill(self, ids, cache):
        import mlx.core as mx
        hidden = None
        for offset in range(0, len(ids), self.prefill_chunk):
            chunk = ids[offset:offset + self.prefill_chunk]
            start = cache[self.lm.model.fa_idx].offset
            positions = mx.arange(start, start + len(chunk))[None]
            hidden = self.lm.model(mx.array([chunk]), cache=cache, position_ids=positions)[:, -1:, :]
            mx.eval(hidden, [entry.state for entry in cache])
        return hidden

    def readout(self, hidden, slots):
        import mlx.core as mx
        logits = self.lm.lm_head(hidden)[0, -1].astype(mx.float32)
        selected = logits[mx.array(slots)]
        mx.eval(selected)
        return selected.tolist()


def _result(row, encoded, selected, metadata, mode):
    ids, slots, prompt_hash = encoded
    return {"id": row["id"], "option_ids": [o["id"] for o in row["options"]],
            "probabilities": softmax(selected), "option_logits": selected,
            "answer_token_ids": slots, "input_tokens": len(ids),
            "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "prompt_sha256": prompt_hash, "prompt_version": PROMPT_VERSION,
            "model": {**metadata, "serving_config": f"qwen38-native-{mode}-v1"},
            "readout": "native last-position logits restricted to declared slots; zero generated tokens",
            "probability_status": "conditional option score; uncalibrated as decision confidence"}


def score(model, tokenizer, row, metadata, max_tokens=4096):
    import mlx.core as mx
    mx.synchronize()
    started = time.perf_counter()
    encoded = encode_prompt(tokenizer, row, max_tokens)
    ids, slots, _ = encoded
    cache = model.lm.make_cache()
    mark = time.perf_counter()
    hidden = model.prefill(ids, cache)
    selected = model.readout(hidden, slots)
    mx.synchronize()
    result = _result(row, encoded, selected, metadata, "direct")
    result.update(forward_seconds=time.perf_counter() - mark, total_seconds=time.perf_counter() - started)
    return result


def score_shared(model, tokenizer, rows, metadata, max_tokens=4096):
    """Prefill once, then evaluate isolated native cache branches.

    The pinned Qwen4 runtime itself serializes batched multi-token suffixes for
    numerical stability. Keep that policy explicit instead of claiming SIMD
    parallel suffix evaluation. All attention, recurrent, PLE and QSA states
    are copied; only immutable MLX arrays may share backing storage.
    """
    import mlx.core as mx
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Shared scoring requires nonempty rows with unique IDs")
    if len(rows) == 1:
        # There is nothing to share: avoid an extra prefill boundary and copy.
        result = score(model, tokenizer, rows[0], metadata, max_tokens)
        return [result], {"total_seconds": result["total_seconds"], "prefix_tokens": 0,
                          "prefill_seconds": result["forward_seconds"], "copy_seconds": 0.0,
                          "suffix_forward_seconds": 0.0, "batch_size": 1,
                          "suffix_execution": "single decision; direct full prefill"}
    mx.synchronize()
    started = time.perf_counter()
    encoded = [encode_prompt(tokenizer, row, max_tokens) for row in rows]
    prefix = _state_prefix(tokenizer, rows[0]["state"])
    if any(_state_prefix(tokenizer, row["state"]) != prefix for row in rows):
        raise ValueError("Shared scoring requires one exact tokenized state")
    if not prefix or any(ids[:len(prefix)] != prefix or len(ids) <= len(prefix) for ids, _, _ in encoded):
        raise ValueError("Shared prefix does not match the full prompts")
    mark = time.perf_counter()
    cache = model.lm.make_cache()
    model.prefill(prefix, cache)
    mx.synchronize()
    prefill_seconds = time.perf_counter() - mark
    results, copy_seconds, suffix_seconds = [], 0.0, 0.0
    for row, enc in zip(rows, encoded):
        mark = time.perf_counter()
        branch = copy.deepcopy(cache)
        copy_seconds += time.perf_counter() - mark
        mark = time.perf_counter()
        ids, slots, _ = enc
        hidden = model.prefill(ids[len(prefix):], branch)
        selected = model.readout(hidden, slots)
        mx.synchronize()
        suffix_seconds += time.perf_counter() - mark
        results.append(_result(row, enc, selected, metadata, "shared"))
        del branch, hidden
    timing = {"total_seconds": time.perf_counter() - started, "prefix_tokens": len(prefix),
              "prefill_seconds": prefill_seconds, "copy_seconds": copy_seconds,
              "suffix_forward_seconds": suffix_seconds, "batch_size": len(rows),
              "suffix_execution": "isolated sequential native branches"}
    return results, timing
