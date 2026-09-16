"""Download the exact public source snapshots used by the evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request


SOURCES = {
    "wanli-test.jsonl": (
        "https://huggingface.co/datasets/alisawuffles/WANLI/resolve/"
        "61c95318fd71c55b6ba355d76253254615f387ec/test.jsonl",
        "4276e0af7fcdf657d1ab7beb54eaf025fda592a76c9ee86b63b7871953fc74fd",
    ),
    "typesafe-security_incidents-cases.js": (
        "https://evals.typesafe.ai/security_incidents-cases.js",
        "6c96b19f192d07004613afc281a31712aa7d04bd75e31a3afe2322174ade0645",
    ),
    "typesafe-agent_trace_observability-cases.js": (
        "https://evals.typesafe.ai/agent_trace_observability-cases.js",
        "4a3821a24366dc9830e12b481a23c96d0ab70e3c473b59551b88e8d0c4b615b3",
    ),
    "typesafe-invoice_processing-cases.js": (
        "https://evals.typesafe.ai/invoice_processing-cases.js",
        "8c2f886978a30e637d577cd0713b3cf12bb622ef7210fcc9a4be47540a6df27f",
    ),
    "typesafe-customer_service-cases.js": (
        "https://evals.typesafe.ai/customer_service-cases.js",
        "066f789bcc17ae906a17fc37ad4fd2f2bb1e881245aeb76b524715bc962a2493",
    ),
    "every-experiments.json": (
        "https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/experiments.json",
        "32311398e800e2e22e8cb945d41ae5328aedebb46cc57193e2b332ec17ed78a0",
    ),
    "every-source.zip": (
        "https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/typesafe-lab-source.zip",
        "9fbf42e9d9e7cd3b072e0271a0959dbfdca85618e93fe4f0ca519d683c418bf0",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        destination = args.output / name
        if destination.exists():
            raise ValueError(f"Refusing to replace {destination}")
        request = urllib.request.Request(url, headers={"User-Agent": "openjev-research-fetch/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError(f"{url} exceeded the 64 MiB download limit")
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"{url} changed: expected {expected}, received {actual}")
        destination.write_bytes(data)
        print(f"{actual}  {destination}")


if __name__ == "__main__":
    main()
