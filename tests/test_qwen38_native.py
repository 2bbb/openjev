import copy
import json
import struct

import pytest

from openjev_phase1.qwen38_native_backend import ple_manifest, quantizer_for, tensor_header


def test_mmap_manifest_addresses_existing_payload_without_copy(tmp_path):
    header = {
        "weight": {"dtype": "U32", "shape": [2, 20], "data_offsets": [0, 160]},
        "scales": {"dtype": "BF16", "shape": [2, 5], "data_offsets": [160, 180]},
        "biases": {"dtype": "BF16", "shape": [2, 5], "data_offsets": [180, 200]},
    }
    encoded = json.dumps(header).encode()
    payload = struct.pack("<Q", len(encoded)) + encoded + bytes(200)
    source = tmp_path / "ngram_table.bin"
    source.write_bytes(payload)
    config = {"ngram_table": {"file": "ngram_table.bin", "bits": 4, "group_size": 32}}
    manifest = ple_manifest(tmp_path, config)
    assert manifest["row_width"] == 160
    assert manifest["shards"][0]["scales"]["offset"] == len(encoded) + 8 + 160
    assert source.read_bytes() == payload
    source.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="Invalid n-gram descriptor"):
        ple_manifest(tmp_path, config)


@pytest.mark.parametrize("bits", [4, 8])
def test_mixed_quantizer_inferred_from_shapes(bits):
    header = {"head.weight": {"dtype": "U32", "shape": [100, 256 * bits // 32]},
              "head.scales": {"dtype": "BF16", "shape": [100, 4]},
              "head.biases": {"dtype": "BF16", "shape": [100, 4]}}
    assert quantizer_for(header, "head", 256) == {"bits": bits, "group_size": 64, "mode": "affine"}
    assert quantizer_for(header, "dense", 256) is False
    with pytest.raises(ValueError):
        quantizer_for(header, "head", 255)


def test_invalid_safetensors_header_rejected(tmp_path):
    source = tmp_path / "bad.safetensors"
    source.write_bytes(struct.pack("<Q", 2**63))
    with pytest.raises(ValueError, match="header length"):
        tensor_header(source)


def tiny_model():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.qwen4_exp import LanguageModel, TextConfig
    config = TextConfig(
        model_type="qwen4_exp_text", hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=3,
        num_experts=4, num_experts_per_tok=2, shared_expert_intermediate_size=16,
        moe_intermediate_size=16, rms_norm_eps=1e-6, vocab_size=64,
        num_key_value_heads=2, max_position_embeddings=128, hc_count=2, hc_lowrank=8,
        head_dim=8, layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1], ple_embed_dim=32, ple_conv_kernel_size=3,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4, split_ngram_parts=4,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=2, eos_token_id=1,
        rope_parameters={"rope_type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 10000, "partial_rotary_factor": 1.0})
    mx.random.seed(73)
    lm = LanguageModel(config)
    mx.eval(lm.parameters())
    return mx, lm


def test_native_cache_branch_isolation_and_chunked_scoring():
    from openjev_phase1.qwen38_native_backend import NativeModel
    mx, lm = tiny_model()
    model = NativeModel(lm, None, prefill_chunk=5)
    prefix, suffixes = list(range(2, 14)), [[14, 15], [20, 21, 22, 23]]
    cache = lm.make_cache()
    model.prefill(prefix, cache)
    original = copy.deepcopy(cache)
    for suffix in suffixes + suffixes[::-1]:
        branch = copy.deepcopy(cache)
        reused = model.readout(model.prefill(suffix, branch), [7, 8])
        fresh = model.readout(model.prefill(prefix + suffix, lm.make_cache()), [7, 8])
        # Prefix splitting changes QSA/GDN reduction shapes. Bound the small
        # floating-point drift as well as the winning option on this fixture.
        assert reused == pytest.approx(fresh, abs=1e-3)
        assert (reused[0] > reused[1]) == (fresh[0] > fresh[1])
    # Includes PLE history/convolution and sparse indexer state, beyond KV alone.
    def flatten(value):
        if isinstance(value, (list, tuple)):
            return [x for item in value for x in flatten(item)]
        return [value]
    for old, kept in zip(original, cache):
        for left, right in zip(flatten(old.state), flatten(kept.state)):
            if isinstance(left, mx.array):
                assert bool(mx.array_equal(left, right))
            else:
                assert left == right


@pytest.mark.parametrize('commit,mlx_version', [('wrong', '0.32.2'), ('expected', '0.32.1')])
def test_runtime_pin_rejects_unverified_versions(monkeypatch, commit, mlx_version):
    from types import SimpleNamespace
    from openjev_phase1 import qwen38_native_backend as backend
    value = backend.VLM_REVISION if commit == 'expected' else commit
    monkeypatch.setattr(backend, 'distribution', lambda name: SimpleNamespace(
        read_text=lambda file: json.dumps({'vcs_info': {'commit_id': value}})))
    monkeypatch.setattr(backend, 'version', lambda name: mlx_version)
    with pytest.raises(RuntimeError, match='requires'):
        backend.runtime_source()
