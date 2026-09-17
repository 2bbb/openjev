import sys

import pytest

from openjev_phase1.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--backend", "mlx", "--mode", "reranker"], "reranker requires torch"),
    (["--mode", "direct", "--mlx-bits", "4"], "requires --backend mlx"),
])
def test_invalid_backend_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["openjev-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()
