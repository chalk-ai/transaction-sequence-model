#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Serve the model that train.py produced, behind a small page, on a CPU scaling group.

    ./deploy.py [--env-id <environment>]

The replica reads `model.pt` from the `transaction-sequence-model` volume at startup with
its own workload identity, so redeploying (or just restarting) picks up the newest weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="transaction-sequence-model")
    parser.add_argument("--env-id", default=None)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
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
        name=args.name,
        cpu="2",
        memory="4Gi",
        env={
            "CHALK_API_SERVER": os.environ["CHALK_API_SERVER"],
            "CHALK_ENVIRONMENT_ID": os.environ["CHALK_ENVIRONMENT_ID"],
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
