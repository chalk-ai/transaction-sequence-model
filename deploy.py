#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Serve the model that train.py produced, behind a small page, on a CPU scaling group.

    ./deploy.py

The replica reads `model.pt` from the `transaction-sequence-model` volume at startup with
its own workload identity, so redeploying (or just restarting) picks up the newest weights.
"""

from __future__ import annotations

import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        # The model is ~350k parameters: CPU inference takes milliseconds, so skip CUDA torch.
        .run_commands("pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu")
        .pip_install(["chalkcompute==2.13.4", "fastapi==0.116.1", "uvicorn[standard]==0.35.0"])
        .add_local_file(str(HERE / "data.py"), "/app/data.py", strategy="copy")
        .add_local_file(str(HERE / "model.py"), "/app/model.py", strategy="copy")
        .add_local_file(str(HERE / "serve.py"), "/app/serve.py", strategy="copy")
        # Inline copies share a 32 KiB budget, so the page rides on a volume. It gets its
        # own directory because a volume mount hides whatever the image has at that path.
        .add_local_file(str(HERE / "index.html"), "/web/index.html", strategy="volume")
        .workdir("/app")
    )
    sg = ScalingGroup(
        image=image,
        name='transaction-sequence-model',
        cpu="2",
        memory="4Gi",
        env={
            "INDEX_HTML": "/web/index.html",
        },
        entrypoint=["python", "/app/serve.py"],
        port=8080,
        # A workload identity lets the replica read the model volume without an API key.
        chalk_identity=True,
        readiness_probe=ReadinessProbe.http("/api/model"),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    sg.deploy(build_timeout=1800.0, ready_timeout=900.0)
    print(f"ready in {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
