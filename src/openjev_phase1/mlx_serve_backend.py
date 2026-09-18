"""Qwen3.8 Flash Next option scores through a resident mlx-serve process.

This bounded direct-mode bridge reuses native packed weights. It does not load
them with MLX-LM. All answer slots must appear in the first-position top-20
logprobs, or the row is refused. One transport token is generated and discarded.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import urllib.parse
import urllib.request

from .core import LETTERS, direct_messages, softmax
from .direct import PROMPT_VERSION, encode_prompt

RUNTIME_VERSION = "26.9.4"
BRIDGE_VERSION = "mlx-serve-completion-logprobs-v1"


class Client:
    def __init__(self, base_url):
        url = urllib.parse.urlsplit(base_url)
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.username or url.password or url.path not in {"", "/"}
                or url.query or url.fragment):
            raise ValueError("mlx-serve requires a loopback HTTP base URL without /v1")
        self.base_url = base_url.rstrip("/")
        # Do not send local evidence through environment-configured HTTP proxies.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.base_url + path, data=body,
                                         headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=180) as response:
            return json.load(response)


def load_model(source, revision, *, base_url="http://127.0.0.1:18082"):
    """Load only local tokenizer assets; the existing server owns model weights."""
    path = Path(source)
    if not path.is_dir() or not revision:
        raise ValueError("mlx-serve requires a local model directory and explicit revision label")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or config.get("model_file"):
        raise ValueError("mlx-serve bridge supports native Qwen3.8 Flash Next (qwen4_exp) only")
    client = Client(base_url)
    props = client.request("/props")
    settings = props.get("settings", {})
    if settings.get("version") != RUNTIME_VERSION or settings.get("engine") != "mlx":
        raise ValueError(f"mlx-serve bridge requires pinned runtime {RUNTIME_VERSION} / MLX")
    if props.get("default_generation_settings", {}).get("model") != "qwen4_exp":
        raise ValueError("Server architecture does not match the local checkpoint")
    if settings.get("kv_quant") != "off" or settings.get("decode_attn_quant") is not False:
        raise ValueError("Scoring requires KV and decode-attention quantization off")
    models = client.request("/v1/models")["data"]
    if len(models) != 1 or models[0].get("meta", {}).get("architecture") != "qwen4_exp":
        raise ValueError("Expected exactly one resident Qwen3.8 Flash Next model")
    client.model_id = models[0]["id"]
    client.context_length = props["default_generation_settings"]["n_ctx"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    assets = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        with (path / name).open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            assets[name] = digest.hexdigest()
    metadata = {
        "source": str(path), "revision": revision,
        "revision_basis": "caller-provided; server API does not attest loaded weight hashes",
        "backend": BRIDGE_VERSION, "runtime_version": RUNTIME_VERSION,
        "base_url": client.base_url, "server_model": client.model_id,
        "tokenizer_artifact_sha256": assets, "quantization": config.get("quantization"),
    }
    return client, tokenizer, metadata


def read_scores(response, count, input_tokens, model_id):
    if response.get("model") != model_id or response.get("system_fingerprint") != "mlx-serve":
        raise ValueError("Unexpected model or runtime in completion response")
    usage = response.get("usage", {})
    if usage.get("prompt_tokens") != input_tokens or usage.get("completion_tokens") != 1:
        raise ValueError("Prompt token count changed or completion was not exactly one transport token")
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Expected exactly one completion choice")
    records = (choices[0].get("logprobs") or {}).get("top_logprobs", [])
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("Missing first-position logprobs; this server cannot score the row")
    missing = [letter for letter in LETTERS[:count] if letter not in records[0]]
    if missing:
        raise ValueError(f"Answer slots {missing} absent from top-20 logprobs; refusing incomplete distribution")
    selected = [records[0][letter] for letter in LETTERS[:count]]
    if any(type(value) not in (float, int) or not math.isfinite(value) or value > 0 for value in selected):
        raise ValueError("Nonfinite or invalid answer-slot logprobs")
    return selected, softmax(selected)


def score(model, tokenizer, row, metadata, max_tokens=4096):
    started = time.perf_counter()
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, min(max_tokens, model.context_length - 1))
    prompt = tokenizer.apply_chat_template(direct_messages(row), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    # Verify the actual server tokenizer including every answer-slot boundary.
    if model.request("/tokenize", {"content": prompt}).get("tokens") != ids:
        raise ValueError(f"Row {row['id']}: server and OpenJev prompt tokenization differ")
    for letter, slot in zip(LETTERS, slots):
        if model.request("/tokenize", {"content": prompt + letter}).get("tokens") != ids + [slot]:
            raise ValueError(f"Row {row['id']}: server answer boundary differs for {letter}")
    forward_start = time.perf_counter()
    response = model.request("/v1/completions", {
        "model": model.model_id, "prompt": prompt, "max_tokens": 1, "logprobs": 20,
        "temperature": 0, "top_p": 1, "top_k": 0,
        "repeat_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0,
        "enable_mtp": False, "enable_drafter": False, "enable_pld": False,
    })
    selected, probabilities = read_scores(response, len(slots), len(ids), model.model_id)
    return {
        "id": row["id"], "option_ids": [option["id"] for option in row["options"]],
        "probabilities": probabilities, "option_logprobs": selected,
        "answer_token_ids": slots, "input_tokens": len(ids),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "prompt_sha256": prompt_hash, "prompt_version": PROMPT_VERSION,
        "model": metadata, "request_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "readout": "softmax of complete answer-slot first-position logprobs; one transport token discarded",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
        "generated_transport_tokens": 1, "api_response": response,
    }
