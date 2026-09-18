import sys

import pytest

from openjev_phase1.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--backend", "mlx", "--mode", "reranker"], "reranker requires torch"),
    (["--mode", "direct", "--mlx-bits", "4"], "requires --backend mlx"),
    (["--mode", "direct", "--mlx-cache-limit-mib", "0"], "requires --backend mlx"),
    (["--mode", "direct", "--backend", "mlx", "--mlx-cache-limit-mib", "-1"], "must be nonnegative"),
])
def test_invalid_backend_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["openjev-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize('limit', [None, 0, 512])
def test_cli_passes_cache_limit_to_loader(tmp_path, monkeypatch, limit):
    import json
    from types import SimpleNamespace
    import openjev_phase1

    fake_backend = SimpleNamespace(
        DEFAULT_CACHE_LIMIT_MIB=256,
        load_model=lambda source, revision, bits, *, cache_limit_mib:
            (None, None, {'limit': cache_limit_mib}),
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(openjev_phase1, 'mlx_backend', fake_backend, raising=False)
    source, output = tmp_path / 'input.jsonl', tmp_path / 'output.jsonl'
    source.write_text(json.dumps({'id': 'test', 'state': 'Evidence', 'question': 'Supported?',
                                 'options': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}) + '\n')
    args = ['openjev-score', '--backend', 'mlx', '--mode', 'direct', '--model', 'unused',
            '--revision', 'unused', '--input', str(source), '--output', str(output)]
    if limit is not None:
        args += ['--mlx-cache-limit-mib', str(limit)]
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert json.loads(output.read_text())['limit'] == (256 if limit is None else limit)


def test_native_shared_groups_exact_serialized_states_and_preserves_order(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import openjev_phase1

    groups = []
    def shared(model, tokenizer, rows, metadata, max_tokens):
        groups.append([row['id'] for row in rows])
        return [{'id': row['id']} for row in rows], {'batch_size': len(rows)}

    fake = SimpleNamespace(load_model=lambda *a, **kw: (None, None, {}),
                           score=None, score_shared=shared)
    monkeypatch.setattr(openjev_phase1, 'qwen38_native_backend', fake, raising=False)
    rows = [{'id': str(i), 'state': state, 'question': 'Supported?',
             'options': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}
            for i, state in enumerate([{'a': 1}, ['different'], {'a': 1}])]
    source, output = tmp_path / 'in.jsonl', tmp_path / 'out.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    monkeypatch.setattr(sys, 'argv', ['openjev-score', '--backend', 'qwen38-native',
        '--mode', 'shared', '--model', 'unused', '--revision', 'unused',
        '--input', str(source), '--output', str(output)])
    main()
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert groups == [['0', '2'], ['1']]
    assert [row['id'] for row in results] == ['0', '1', '2']
    assert [row['shared_timing']['batch_size'] for row in results] == [2, 1, 2]
