import re
from pathlib import Path


WEBGPU = Path(__file__).resolve().parent


def test_browser_demo_files_and_pins_exist():
    html = (WEBGPU / "index.html").read_text()
    worker = (WEBGPU / "worker.js").read_text()
    assert 'src="app.js"' in html
    assert 'href="style.css"' in html
    assert "@huggingface/transformers@4.3.0" in worker
    assert "onnx-community/Qwen3-0.6B-ONNX" in worker
    assert "da1453100cf3ff33ef56d17983fc7a8648706db6" in worker
    assert "vue@3.5.21" in (WEBGPU / "app.js").read_text()
    assert "Material+Symbols+Rounded" in html
    assert "icon_names=" in html and "display=block" in html


def test_demo_has_no_canned_benchmark_results_or_backend_calls():
    sources = "\n".join((WEBGPU / name).read_text() for name in ("index.html", "app.js", "worker.js", "README.md"))
    assert "performance.now()" in sources
    assert "WebSocket" not in sources
    assert re.search(r"https?://", sources)
    assert not re.search(r"(?:fetch|axios)\s*\(\s*['\"]/(?:api|generate|score)", sources)


def test_demo_discloses_measurement_and_probability_limits():
    text = ((WEBGPU / "index.html").read_text() + (WEBGPU / "README.md").read_text()).lower()
    for phrase in ("conditional probabilities", "not calibrated", "sequential", "warmup", "no backend"):
        assert phrase in text


def test_demo_checks_required_gpu_feature_and_generated_format():
    html = (WEBGPU / "index.html").read_text()
    app = (WEBGPU / "app.js").read_text()
    worker = (WEBGPU / "worker.js").read_text()
    assert 'id="generation-validity"' in html
    assert 'adapter.features.has("shader-f16")' in app
    assert "JSON.parse" in worker
    assert 'keys.length !== 1 || keys[0] !== "choice"' in worker
    assert "min_new_tokens: 2" in worker
    assert "format failure" in app
