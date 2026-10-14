"""Serve the trained transaction model behind a small page.

Loads `model.pt` from the Chalk volume the training run wrote (or `MODEL_PATH`, for local
runs), then scores freshly generated customers the model has never seen.

  GET /                  the page (index.html)
  GET /api/model         training summary: config, parameter count, per-epoch metrics
  GET /api/sample        ?scenario=none|card_testing|account_takeover|atm_drain&seed=N
                         one synthetic customer, each transaction annotated with the model's
                         fraud score, surprise and next-transaction forecast
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

import data
import model

MODEL_VOLUME = os.environ.get("MODEL_VOLUME", "transaction-sequence-model")
INDEX_HTML = Path(os.environ.get("INDEX_HTML", str(Path(__file__).with_name("index.html"))))
PORT = int(os.environ.get("PORT", "8080"))
# Demo customers use seeds far from the training data's, so the page only shows unseen users.
SEED_OFFSET = 10_000_000

log = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
state: dict[str, Any] = {}


def _fetch_weights() -> str:
    local = os.environ.get("MODEL_PATH")
    if local:
        return local
    from chalkcompute import Volume

    blob = Volume(MODEL_VOLUME, create_if_missing=False).read_file("model.pt")
    path = Path(tempfile.mkdtemp()) / "model.pt"
    path.write_bytes(blob)
    log.info("fetched %d bytes of weights from volume %s", len(blob), MODEL_VOLUME)
    return str(path)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    t0 = time.monotonic()
    state["model"], state["summary"] = model.load(_fetch_weights())
    log.info("model ready in %.1fs", time.monotonic() - t0)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/model")
async def model_info() -> JSONResponse:
    return JSONResponse(state["summary"])


@app.get("/api/sample")
async def sample(scenario: str = "none", seed: int = 0) -> JSONResponse:
    if scenario not in ("none", *data.SCENARIOS):
        raise HTTPException(400, f"unknown scenario {scenario!r}")
    scenario_index = ("none", *data.SCENARIOS).index(scenario)
    rng = random.Random(SEED_OFFSET + seed * 8 + scenario_index)
    txns = data.generate_customer(rng, seed, 40, None if scenario == "none" else scenario)
    rows = [dataclasses.asdict(t) for t in txns]
    t0 = time.perf_counter()
    scored = model.explain(state["model"], rows)
    infer_ms = (time.perf_counter() - t0) * 1000
    start_day = rows[0]["ts"] // 86400
    out = []
    for r, s in zip(rows, scored):
        day, secs = divmod(r["ts"], 86400)
        when = f"Day {day - start_day + 1} · {secs // 3600:02d}:{secs % 3600 // 60:02d}"
        out.append({"when": when, "category": r["category"], "amount": r["amount"],
                    "channel": r["channel"], "is_fraud": r["is_fraud"], **s})  # fmt: skip
    return JSONResponse({"scenario": scenario, "persona": rows[0]["persona"], "seed": seed,
                         "infer_ms": round(infer_ms, 1), "transactions": out})  # fmt: skip


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
