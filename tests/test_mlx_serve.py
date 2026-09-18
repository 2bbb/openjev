import json
import math
from types import SimpleNamespace

import pytest

from openjev_phase1 import mlx_serve_backend as bridge


def response():
    return {
        "model": "local-qwen", "system_fingerprint": "mlx-serve",
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        "choices": [{"text": "ignored", "logprobs": {"top_logprobs": [{"A": -2.0, "B": -3.0}]}}],
    }


def test_conditional_scores_preserve_logit_ratios_and_ignore_generated_text():
    raw = response()
    selected, probabilities = bridge.read_scores(raw, 2, 3, "local-qwen")
    assert selected == [-2, -3]
    assert probabilities == pytest.approx([1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))])
    assert sum(probabilities) == pytest.approx(1)
    raw["choices"][0]["text"] = "arbitrary explanation that is not a valid answer"
    assert bridge.read_scores(raw, 2, 3, "local-qwen")[1] == probabilities
    # Subtracting the same vocabulary log-normalizer has no effect.
    raw["choices"][0]["logprobs"]["top_logprobs"][0] = {"A": -102.0, "B": -103.0}
    assert bridge.read_scores(raw, 2, 3, "local-qwen")[1] == pytest.approx(probabilities)


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), float("-inf"), 0.1])
def test_invalid_scores_refused(value):
    raw = response()
    raw["choices"][0]["logprobs"]["top_logprobs"][0]["B"] = value
    with pytest.raises(ValueError, match="invalid answer-slot"):
        bridge.read_scores(raw, 2, 3, "local-qwen")


def test_missing_slot_is_not_zero_filled():
    raw = response()
    del raw["choices"][0]["logprobs"]["top_logprobs"][0]["B"]
    with pytest.raises(ValueError, match="refusing incomplete"):
        bridge.read_scores(raw, 2, 3, "local-qwen")


@pytest.mark.parametrize("changes", [
    {"model": "wrong"}, {"system_fingerprint": "wrong"}, {"choices": []},
    {"usage": {"prompt_tokens": 2, "completion_tokens": 1}},
    {"usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    {"choices": [{"logprobs": None}]},
])
def test_wrong_response_contract_refused(changes):
    raw = response()
    raw.update(changes)
    with pytest.raises(ValueError):
        bridge.read_scores(raw, 2, 3, "local-qwen")


@pytest.mark.parametrize("url", ["https://example.com", "http://192.168.1.3", "http://localhost/v1",
                                 "http://user:password@localhost", "http://localhost?x=y"])
def test_only_explicit_local_server_urls(url):
    with pytest.raises(ValueError, match="loopback"):
        bridge.Client(url)


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(token) for token in ids)

    def apply_chat_template(self, *args, **kwargs):
        return "p: "


def test_score_checks_native_token_boundaries_and_transport_flags():
    calls = []

    def request(path, payload):
        calls.append((path, payload))
        if path == "/tokenize":
            return {"tokens": Tokenizer().encode(payload["content"])}
        return response()

    model = SimpleNamespace(request=request, model_id="local-qwen", context_length=4096)
    row = {"id": "a", "state": "evidence", "question": "which?",
           "options": [{"id": "yes", "description": "yes"}, {"id": "no", "description": "no"}]}
    result = bridge.score(model, Tokenizer(), row, {})
    assert result["answer_token_ids"] == [65, 66]
    assert result["generated_transport_tokens"] == 1
    assert [item[1]["content"] for item in calls[:-1]] == ["p: ", "p: A", "p: B"]
    payload = calls[-1][1]
    assert payload["max_tokens"] == 1 and payload["logprobs"] == 20
    assert all(payload[flag] is False for flag in ("enable_mtp", "enable_drafter", "enable_pld"))
    assert "option_logits" not in result  # logprobs are not raw logits
    assert result["probabilities"] == pytest.approx(bridge.read_scores(response(), 2, 3, "local-qwen")[1])
    with pytest.raises(ValueError, match="no truncation"):
        bridge.score(model, Tokenizer(), row, {}, max_tokens=2)
    model.request = lambda path, payload: {"tokens": []}
    with pytest.raises(ValueError, match="tokenization differ"):
        bridge.score(model, Tokenizer(), row, {})


def test_loader_rejects_incompatible_checkpoint_before_contacting_server(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(bridge, "Client", lambda *args: pytest.fail("must not contact server"))
    with pytest.raises(ValueError, match="qwen4_exp"):
        bridge.load_model(str(tmp_path), "pinned")


@pytest.mark.parametrize("mode", ["serial", "shared", "reranker"])
def test_cli_unsupported_modes_fail_before_input_read(tmp_path, monkeypatch, capsys, mode):
    import sys
    from openjev_phase1.cli import main
    monkeypatch.setattr(sys, "argv", ["openjev-score", "--backend", "mlx-serve", "--mode", mode,
                                    "--model", "unused", "--revision", "unused", "--input", "missing.jsonl",
                                    "--output", str(tmp_path / "output.jsonl")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "direct mode only" in capsys.readouterr().err
    assert not (tmp_path / "output.jsonl").exists()
