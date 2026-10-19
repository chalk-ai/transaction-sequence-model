#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Generate synthetic transactions, upload them as a Chalk dataset, and train on a GPU.

    ./train.py # upload the dataset, run training on an L4, wait
    ./train.py # more data
    ./train.py # reuse the dataset's latest revision

The training run receives the dataset as a Polars LazyFrame and the config as a dict. It
writes the trained weights to the Chalk volume `transaction-sequence-model`, where the
serving app (deploy.py) reads them.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# The training run mounts this script's source over /app, which would hide anything the
# image copies there, so the shared modules live in their own directory.
LIB_DIR = "/opt/txnmodel"
DATASET = "fraud-transaction-sequences"
MODEL_VOLUME = "transaction-sequence-model"


def train_sequence_model(df: Any, config: dict[str, Any]) -> None:
    """Runs inside the training container: `df` is the dataset, `config` the run config."""
    import dataclasses
    import io
    import sys

    import torch
    from chalkcompute import Volume

    sys.path.insert(0, LIB_DIR)
    import model

    rows = df.collect().to_dicts() if hasattr(df, "collect") else df.to_dicts()
    # The run config travels as a protobuf Struct, which stores every number as a float.
    types = {f.name: f.type for f in dataclasses.fields(model.TrainConfig)}
    cfg = model.TrainConfig(**{k: int(v) if types[k] == "int" else v for k, v in config.items()})
    trained, summary = model.train_model(rows, cfg, lambda line: print(line, flush=True))

    buf = io.BytesIO()
    torch.save(
        {"config": dataclasses.asdict(trained.cfg), "state_dict": trained.state_dict(), "summary": summary},
        buf,
    )
    checkpoint_dir = Path(os.environ.get("CHALK_CHECKPOINT_DIR", "/chalk/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.pt").write_bytes(buf.getvalue())
    # The serving app loads the newest weights from this volume at startup.
    volume = Volume(MODEL_VOLUME)
    volume.put_file("model.pt", buf.getvalue())
    volume.put_file("summary.json", json.dumps(summary, indent=2))
    print(f"wrote {len(buf.getvalue()):,} bytes to volume {MODEL_VOLUME}", flush=True)


def main() -> None:
    import chalkcompute
    from chalkcompute import DatasetClient, Image, Volume

    import data

    t0 = time.monotonic()
    txns = data.generate(20000, 0.12, seed=7)
    ref = DatasetClient().upload(DATASET, data.to_columns(txns))
    n_fraud = sum(t.is_fraud for t in txns)
    print(f"uploaded {len(txns):,} transactions ({n_fraud:,} fraud) as {ref} "
          f"in {time.monotonic() - t0:.0f}s")  # fmt: skip  # fmt: skip

    image = (
        # PyTorch's CUDA runtime image already has torch, so the build only adds polars.
        Image.base("pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime")
        .pip_install(["polars==1.39.3"])
        .add_local_file(str(HERE / "model.py"), f"{LIB_DIR}/model.py", strategy="copy")
        .add_local_file(str(HERE / "data.py"), f"{LIB_DIR}/data.py", strategy="copy")
    )
    trainer = chalkcompute.training(
        dataset=DATASET,
        image=image,
        name="transaction-sequence-model",
        gpu='nvidia-l4',
        cpu="4",
        memory="16Gi",
    )(train_sequence_model)

    config = {"epochs": 8, "batch_size": 128, "lr": 2e-3, "d_model": 96,
              "n_layers": 3, "n_heads": 4, "fraud_weight": 1.0}  # fmt: skip
    t0 = time.monotonic()
    try:
        run = trainer.run_training(config=config)
    except Exception as exc:
        # The SDK stops waiting on an image build after 600s, but the build carries on
        # server-side; a second submission picks up the finished image from cache.
        if "Image build" not in str(exc) or "timed out" not in str(exc):
            raise
        print(f"{exc}; resubmitting once the build finishes")
        run = trainer.run_training(config=config)
    print(f"training run {run.run_id} submitted; waiting...")
    run.wait(timeout=3600)
    print(f"training run finished in {time.monotonic() - t0:.0f}s")
    summary = json.loads(Volume(MODEL_VOLUME).read_file("summary.json"))
    for row in summary["history"]:
        print(row)


if __name__ == "__main__":
    main()
