"""A small causal transformer over card transactions: predict the next one, flag fraud.

Every transaction becomes five tokens (category, amount bucket, channel, hour of day, time
since the previous transaction) whose embeddings are summed into one vector. A causal
transformer reads the sequence left to right, and at position t it has seen transactions
0..t only. Two kinds of head sit on top:

  next-action heads  predict transaction t+1's category, amount bucket and channel
  fraud head         scores whether transaction t is fraudulent

The next-action heads are trained self-supervised on every customer, and give a "surprise"
signal for free: how unlikely each transaction was given the customer's own history. The
fraud head is trained on the labelled bursts.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import asdict, dataclass
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from data import CATEGORIES, CHANNELS

AMOUNT_EDGES = [1, 5, 15, 40, 100, 250, 600, 1500]
AMOUNT_LABELS = ["<$1", "$1–5", "$5–15", "$15–40", "$40–100", "$100–250", "$250–600", "$600–1.5k", "$1.5k+"]
GAP_EDGES = [120, 3600, 6 * 3600, 86400, 7 * 86400]  # 2 min, 1 h, 6 h, 1 day, 1 week
GAP_LABELS = ["<2m", "<1h", "<6h", "<1d", "<1w", "1w+"]
N_HOURS = 6  # four-hour bands
MAX_LEN = 64
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
CHANNEL_INDEX = {c: i for i, c in enumerate(CHANNELS)}


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    d_model: int
    n_layers: int
    n_heads: int
    fraud_weight: float  # weight of the fraud loss relative to the next-action losses


def tokenize(txns: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Per-field token ids for one customer's transactions, oldest first."""
    out: dict[str, list[int]] = {"cat": [], "amt": [], "chan": [], "hour": [], "gap": []}
    prev_ts: int | None = None
    for t in txns:
        out["cat"].append(CAT_INDEX[t["category"]])
        out["amt"].append(bisect.bisect_right(AMOUNT_EDGES, t["amount"]))
        out["chan"].append(CHANNEL_INDEX[t["channel"]])
        out["hour"].append((t["ts"] % 86400) // (4 * 3600))
        gap = len(GAP_EDGES) if prev_ts is None else bisect.bisect_right(GAP_EDGES, t["ts"] - prev_ts)
        out["gap"].append(gap)
        prev_ts = t["ts"]
    return out


class TransactionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.cat = nn.Embedding(len(CATEGORIES), d)
        self.amt = nn.Embedding(len(AMOUNT_LABELS), d)
        self.chan = nn.Embedding(len(CHANNELS), d)
        self.hour = nn.Embedding(N_HOURS, d)
        self.gap = nn.Embedding(len(GAP_LABELS), d)
        self.pos = nn.Embedding(MAX_LEN, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.next_cat = nn.Linear(d, len(CATEGORIES))
        self.next_amt = nn.Linear(d, len(AMOUNT_LABELS))
        self.next_chan = nn.Linear(d, len(CHANNELS))
        self.fraud = nn.Linear(d, 1)

    def forward(self, x: dict[str, torch.Tensor], pad: torch.Tensor) -> dict[str, torch.Tensor]:
        """`x[field]` is (batch, seq) token ids; `pad` is True at padding positions."""
        seq = pad.shape[1]
        pos = torch.arange(seq, device=pad.device)
        h = (
            self.cat(x["cat"]) + self.amt(x["amt"]) + self.chan(x["chan"])
            + self.hour(x["hour"]) + self.gap(x["gap"]) + self.pos(pos)
        )  # fmt: skip
        # Causal: position t attends to 0..t, so every head output uses only the past.
        causal = torch.ones(seq, seq, dtype=torch.bool, device=pad.device).triu(1)
        h = self.norm(self.encoder(h, mask=causal, src_key_padding_mask=pad, is_causal=True))
        return {
            "next_cat": self.next_cat(h),
            "next_amt": self.next_amt(h),
            "next_chan": self.next_chan(h),
            "fraud": self.fraud(h).squeeze(-1),
        }


def collate(
    seqs: list[dict[str, list[int]]], labels: list[list[int]], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Right-pads a batch; returns (tokens, pad mask, fraud labels)."""
    n = max(len(s["cat"]) for s in seqs)
    x = {f: torch.zeros(len(seqs), n, dtype=torch.long) for f in seqs[0]}
    y = torch.zeros(len(seqs), n)
    pad = torch.ones(len(seqs), n, dtype=torch.bool)
    for i, (s, lab) in enumerate(zip(seqs, labels)):
        k = len(s["cat"])
        for f in s:
            x[f][i, :k] = torch.tensor(s[f])
        y[i, :k] = torch.tensor(lab, dtype=torch.float)
        pad[i, :k] = False
    return {f: v.to(device) for f, v in x.items()}, pad.to(device), y.to(device)


def losses(out: dict[str, torch.Tensor], x: dict[str, torch.Tensor], pad: torch.Tensor,
           y: torch.Tensor, pos_weight: float) -> dict[str, torch.Tensor]:  # fmt: skip
    # Position t predicts transaction t+1, so drop the last position and the first target.
    nxt = ~pad[:, 1:]
    out_ = {k: out[k][:, :-1][nxt] for k in ("next_cat", "next_amt", "next_chan")}
    real = ~pad
    return {
        "cat": F.cross_entropy(out_["next_cat"], x["cat"][:, 1:][nxt]),
        "amt": F.cross_entropy(out_["next_amt"], x["amt"][:, 1:][nxt]),
        "chan": F.cross_entropy(out_["next_chan"], x["chan"][:, 1:][nxt]),
        "fraud": F.binary_cross_entropy_with_logits(
            out["fraud"][real], y[real], pos_weight=torch.tensor(pos_weight, device=y.device)
        ),
    }


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Area under the precision-recall curve, the usual metric for rare positives."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    hits, total, ap = 0, sum(labels), 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            hits += 1
            ap += hits / rank
    return ap / max(total, 1)


@torch.no_grad()
def evaluate(
    model: TransactionTransformer,
    seqs: list[dict[str, list[int]]],
    labels: list[list[int]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = count = 0
    scores: list[float] = []
    flat: list[int] = []
    for i in range(0, len(seqs), 256):
        x, pad, y = collate(seqs[i : i + 256], labels[i : i + 256], device)
        out = model(x, pad)
        nxt = ~pad[:, 1:]
        pred = out["next_cat"][:, :-1].argmax(-1)
        correct += (pred == x["cat"][:, 1:])[nxt].sum().item()
        count += nxt.sum().item()
        real = ~pad
        scores += torch.sigmoid(out["fraud"])[real].tolist()
        flat += y[real].int().tolist()
    model.train()
    return {
        "next_category_accuracy": round(correct / max(count, 1), 4),
        "fraud_average_precision": round(average_precision(scores, flat), 4),
        "fraud_base_rate": round(sum(flat) / max(len(flat), 1), 4),
    }


def group_customers(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_user: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r)
    return [sorted(v, key=lambda r: r["seq_idx"])[:MAX_LEN] for _, v in sorted(by_user.items())]


def train_model(
    rows: list[dict[str, Any]], cfg: TrainConfig, log: Callable[[str], None]
) -> tuple[TransactionTransformer, dict[str, Any]]:
    """Trains on 90% of customers and reports metrics on the held-out 10%."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    customers = group_customers(rows)
    seqs = [tokenize(c) for c in customers]
    labels = [[r["is_fraud"] for r in c] for c in customers]
    split = int(len(seqs) * 0.9)
    train_s, train_y, val_s, val_y = seqs[:split], labels[:split], seqs[split:], labels[split:]
    pos_rate = sum(map(sum, train_y)) / sum(map(len, train_y))
    pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)

    model_cfg = ModelConfig(cfg.d_model, cfg.n_layers, cfg.n_heads, 0.1)
    model = TransactionTransformer(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    steps = cfg.epochs * math.ceil(len(train_s) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{len(train_s)} train / {len(val_s)} val customers, {n_params:,} parameters, {device}")

    t0 = time.time()
    history: list[dict] = []
    gen = torch.Generator().manual_seed(0)
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(train_s), generator=gen).tolist()
        totals = {"cat": 0.0, "amt": 0.0, "chan": 0.0, "fraud": 0.0}
        for b in range(0, len(order), cfg.batch_size):
            idx = order[b : b + cfg.batch_size]
            x, pad, y = collate([train_s[i] for i in idx], [train_y[i] for i in idx], device)
            parts = losses(model(x, pad), x, pad, y, pos_weight)
            loss = parts["cat"] + parts["amt"] + parts["chan"] + cfg.fraud_weight * parts["fraud"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                totals[k] += v.item() * len(idx)
        metrics = evaluate(model, val_s, val_y, device)
        row = {"epoch": epoch + 1, **{f"loss_{k}": round(v / len(train_s), 4) for k, v in totals.items()},
               **metrics, "seconds": round(time.time() - t0, 1)}  # fmt: skip
        history.append(row)
        log(str(row))
    summary = {"config": asdict(cfg), "model": asdict(model_cfg), "parameters": n_params,
               "history": history, "device": str(device)}  # fmt: skip
    return model.cpu(), summary


@torch.no_grad()
def explain(model: TransactionTransformer, txns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-transaction fraud score, surprise, and the model's forecast of what comes next."""
    model.eval()
    s = tokenize(txns[-MAX_LEN:])
    x, pad, _ = collate([s], [[0] * len(s["cat"])], torch.device("cpu"))
    out = model(x, pad)
    p_cat = out["next_cat"][0].softmax(-1)
    p_amt = out["next_amt"][0].softmax(-1)
    fraud = torch.sigmoid(out["fraud"][0])
    rows = []
    for t in range(len(s["cat"])):
        # Surprise of transaction t is judged by the forecast made at t-1.
        surprise = None
        if t > 0:
            surprise = -math.log(p_cat[t - 1, s["cat"][t]].item() + 1e-9) - math.log(
                p_amt[t - 1, s["amt"][t]].item() + 1e-9
            )
        top = torch.topk(p_cat[t], 3)
        rows.append({
            "fraud": round(fraud[t].item(), 4),
            "surprise": None if surprise is None else round(surprise, 2),
            "amount_bucket": AMOUNT_LABELS[s["amt"][t]],
            "gap": GAP_LABELS[s["gap"][t]],
            "next": [{"category": CATEGORIES[i], "p": round(p, 3)}
                     for p, i in zip(top.values.tolist(), top.indices.tolist())],
            "next_amount": AMOUNT_LABELS[int(p_amt[t].argmax())],
        })  # fmt: skip
    return rows


def save(model: TransactionTransformer, summary: dict[str, Any], path: str) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "summary": summary}, path)


def load(path: str) -> tuple[TransactionTransformer, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TransactionTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["summary"]
