import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
import qwen38_native as harness


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, turns, **kwargs):
        return "\n".join(turn["content"] for turn in turns) + "\nAssistant:"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_backend():
    module = types.ModuleType("fake_qwen38_backend")
    module.calls = []

    def load_model(source, revision):
        return object(), Tokenizer(), {"source": source, "revision": revision, "backend": "fake"}

    def score(model, tokenizer, row, metadata, max_tokens=4096):
        module.calls.append(("direct", row["id"]))
        if row["id"] == "f40beba9088c8db8bbd6":
            raise ValueError("Answer slots ['B'] absent from top-20 logprobs; refusing incomplete distribution")
        ids, slots, prompt_hash = harness.encode_prompt(tokenizer, row, max_tokens)
        return {
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "probabilities": [0.8, 0.1, 0.1],
            "option_logits": [2.0, 0.0, 0.0],
            "answer_token_ids": slots,
            "input_tokens": len(ids),
            "input_ids_sha256": "fixture",
            "prompt_sha256": prompt_hash,
            "prompt_version": "direct-options-v1",
            "model": metadata,
            "forward_seconds": 0.01,
            "total_seconds": 0.02,
        }

    def score_shared(model, tokenizer, rows, metadata, max_tokens=4096):
        module.calls.append(("shared", [row["id"] for row in rows]))
        outputs = []
        for row in rows:
            ids, slots, prompt_hash = harness.encode_prompt(tokenizer, row, max_tokens)
            outputs.append({
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": [0.55, 0.45],
                "option_logits": [0.2, 0.0],
                "answer_token_ids": slots,
                "input_tokens": len(ids),
                "input_ids_sha256": "fixture",
                "prompt_sha256": prompt_hash,
                "prompt_version": "direct-options-v1",
                "model": metadata,
            })
        return outputs, {"batch_size": len(rows), "total_seconds": 0.03}

    module.load_model = load_model
    module.score = score
    module.score_shared = score_shared
    return module


def test_direct_harness_writes_successes_errors_quality_and_create_only(tmp_path, monkeypatch):
    backend = make_backend()
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(harness.BACKENDS, "qwen38-native", backend.__name__)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "qwen38_native.py",
        "--backend", "qwen38-native",
        "--model", "local-model",
        "--revision", "local-fixture",
        "--dataset", "authored144",
        "--limit", "2",
        "--output", str(output),
    ])

    harness.main()

    rows = read_jsonl(output / "qwen38-native" / "authored144-direct.jsonl")
    assert [row["id"] for row in rows] == ["a3f18f3a63d45345942b", "f40beba9088c8db8bbd6"]
    assert backend.calls[0] == ("direct", "a3f18f3a63d45345942b")
    assert len(backend.calls) == 3  # one untimed warmup plus two timed rows
    assert rows[0]["harness_status"] == "ok"
    assert rows[0]["prompt_sha256"]
    assert rows[1]["harness_status"] == "error"
    assert "absent from top-20" in rows[1]["error"]
    assert rows[1]["prompt_sha256"]
    summary = json.loads((output / "summary.json").read_text())
    report = summary["reports"]["qwen38-native"]["datasets"]["authored144-direct"]
    assert "configured untimed warmup" in json.loads((output / "manifest.json").read_text())["timing_scope"]
    assert report["warmup"]["timed"] is False
    assert report["warmup"]["records"][0]["id"] == "a3f18f3a63d45345942b"
    assert report["rows_attempted"] == 2
    assert report["rows_ok"] == 1
    assert report["quality"]["available_gold"] == 2
    with pytest.raises(FileExistsError):
        harness.main()


def test_shared_harness_groups_rows_and_compares_backends(tmp_path, monkeypatch):
    backend = make_backend()
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(harness.BACKENDS, "qwen38-native", backend.__name__)
    output = tmp_path / "shared-run"
    monkeypatch.setattr(sys, "argv", [
        "qwen38_native.py",
        "--backend", "qwen38-native",
        "--backend", "qwen38-native",
        "--model", "local-model",
        "--revision", "local-fixture",
        "--dataset", "shape777",
        "--mode", "shared",
        "--group-limit", "1",
        "--max-tokens", "20000",
        "--output", str(output),
    ])

    harness.main()

    left = read_jsonl(output / "qwen38-native" / "shape777-shared.jsonl")
    right = read_jsonl(output / "qwen38-native-2" / "shape777-shared.jsonl")
    assert len(left) == len(right) == 21
    assert left[0]["harness_mode"] == "shared"
    assert all(row["prompt_sha256"] for row in left)
    assert ("shared", [row["id"] for row in left]) in backend.calls
    summary = json.loads((output / "summary.json").read_text())
    report = summary["reports"]["qwen38-native"]["datasets"]["shape777-shared"]
    assert "qwen38-native-2" in summary["reports"]
    assert report["group_timings"][0]["rows"] == 21
    assert report["quality"] is None
    comparison = summary["comparisons"]["qwen38-native:shape777-shared vs qwen38-native-2:shape777-shared"]
    assert comparison["common_rows"] == 21
    assert comparison["run_policy"] == "matched harness_run values; default compares run 0 only"


def test_repeated_timed_runs_compare_run_zero_without_duplicate_id_crash(tmp_path, monkeypatch):
    backend = make_backend()
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(harness.BACKENDS, "qwen38-native", backend.__name__)
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", [
        "qwen38_native.py",
        "--backend", "qwen38-native",
        "--backend", "qwen38-native",
        "--model", "local-model",
        "--revision", "local-fixture",
        "--dataset", "authored144",
        "--limit", "1",
        "--runs", "2",
        "--warmup", "0",
        "--output", str(output),
    ])

    harness.main()

    left = read_jsonl(output / "qwen38-native" / "authored144-direct.jsonl")
    right = read_jsonl(output / "qwen38-native-2" / "authored144-direct.jsonl")
    assert [row["harness_run"] for row in left] == [0, 1]
    assert [row["harness_run"] for row in right] == [0, 1]
    summary = json.loads((output / "summary.json").read_text())
    comparison = summary["comparisons"]["qwen38-native:authored144-direct vs qwen38-native-2:authored144-direct"]
    assert comparison["left_run"] == 0
    assert comparison["right_run"] == 0
    assert comparison["left_rows"] == 1
    assert comparison["right_rows"] == 1
