import re
from pathlib import Path


WEBGPU = Path(__file__).resolve().parent


def test_browser_demo_files_and_pins_exist():
    html = (WEBGPU / "index.html").read_text()
    worker = (WEBGPU / "worker.js").read_text()
    assert 'src="app.js"' in html
    assert 'href="style.css"' in html
    assert "@mlc-ai/web-llm@0.2.85" in worker
    for model in (
        "Qwen3-0.6B-q4f16_1-MLC",
        "Qwen3.5-0.8B-q4f16_1-MLC",
    ):
        assert model in worker
        assert model in html
    assert "vue@3.5.21" in (WEBGPU / "app.js").read_text()
    assert "Material+Symbols+Rounded" in html
    assert "icon_names=" in html and "display=block" in html
    assert 'id="model-select"' in html
    assert "There is no waitlist!" in html
    assert 'data-preset="account"' in html
    assert 'data-preset="email"' in html
    assert 'options: ["Legitimate", "Spam", "Phishing"]' in (WEBGPU / "app.js").read_text()
    app = (WEBGPU / "app.js").read_text()
    assert 'document.querySelectorAll("[data-preset]")' in app
    assert 'id="add-option"' in html and 'id="remove-option"' in html
    assert "const MAX_OPTIONS = 20" in app
    assert "const MIN_OPTIONS = 2" in app
    assert "setOptions(preset.options)" in app
    assert "join the waitlist" not in html.lower()
    assert html.index('class="machine"') < html.index('class="workbench"') < html.index('class="method-map"')
    assert "Can we run something like Jev in your browser?" in html
    assert "Decision model <mark>in your browser.</mark>" in html
    assert "Works best on a laptop or desktop" in html
    assert "isMobileDevice" in (WEBGPU / "app.js").read_text()
    assert "Qwen3-0.6B-q4f16_1-MLC" in (WEBGPU / "app.js").read_text()
    assert "Qwen3.5-0.8B-q4f16_1-MLC" in (WEBGPU / "app.js").read_text()
    assert '<meta name="referrer" content="no-referrer"' in html
    assert 'referrerPolicy: "no-referrer"' in worker
    assert "modelId: modelSelect.value" in (WEBGPU / "app.js").read_text()
    assert "load(data.modelId)" in worker


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
    readme = (WEBGPU / "README.md").read_text()
    assert 'id="generation-validity"' not in html
    assert 'adapter.features.has("shader-f16")' in app
    assert "JSON.parse" in worker
    assert "<think>" in worker and "<\\/think>" in worker
    assert "exactKeys(parsed, expectedKeys)" in worker
    assert "probabilities must sum to 1" in worker
    assert "JSON probabilities" in html
    assert "Estimate the probability" in worker
    assert '{"A: Route north": 0.65, "B: Route south": 0.35}' in worker
    assert "Account access support" not in worker
    assert "generated, self-reported probabilities" in readme
    assert "max_tokens: 512" in worker
    assert "verbose" not in html.lower() + readme.lower()
    assert "Qwen3.5-2B" not in html + app + worker + readme
    assert "top_logprobs: group.length" in worker
    assert "group.map((label)" in worker
    assert "groups.length" in worker
    assert "relativeLogits" in worker
    assert "optionLogprobs" in worker
    assert "CreateMLCEngine" in worker
    assert "format failure" not in app
