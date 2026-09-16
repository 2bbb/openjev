# Third-party material

No model weights or third-party raw evaluation records are distributed here.

| Item | Upstream | Pinned revision | Note |
|---|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Direct-logit baseline; check upstream model license. |
| Qwen3-Reranker-4B | https://huggingface.co/Qwen/Qwen3-Reranker-4B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | Native reranker baseline; Apache-2.0 on its model card. |
| TypeSafe public evaluations | https://evals.typesafe.ai/ | Exact case URLs and snapshot hashes are in `benchmarks/fetch_sources.py`; no redistribution grant inferred. |
| Every parallel judgment lab | https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/ | [Experiment JSON](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/experiments.json) and [source archive](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/typesafe-lab-source.zip) are public downloads. |
| WANLI | https://huggingface.co/datasets/alisawuffles/WANLI | Pinned at `61c95318fd71c55b6ba355d76253254615f387ec`; CC-BY-4.0. |

These URLs returned HTTP 200 on 2026-09-16. `benchmarks/fetch_sources.py` refuses content whose SHA-256 differs from the evaluated snapshot.

TypeSafe and Jev are marks of their respective owner. OpenJev is unaffiliated. OpenJev code is provided under the repository's MIT License; third-party models and material retain their upstream terms.
