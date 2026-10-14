"""Synthetic card transactions: everyday spending per persona, with injected fraud bursts.

Each customer has a persona (what they buy, how much, when) and a sequence of transactions
spaced hours to days apart. About 12% of customers are hit by one fraud scenario partway
through their history, which inserts a short burst of fraudulent transactions labelled
`is_fraud = 1`:

  card_testing      several sub-$3 online "digital goods" charges minutes apart, then large
                    electronics or gift-card purchases
  account_takeover  late-night peer-to-peer and crypto transfers of a few hundred dollars
  atm_drain         back-to-back ATM withdrawals in the small hours

Legitimate customers also produce look-alikes of each pattern: app-store binges of tiny
charges with no big purchase after, the occasional late-night crypto buy, and a couple of ATM
withdrawals on a cash day. So fraud is visible in the *sequence* (what follows, timing, a
break from the customer's habits) more than in any single transaction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

CATEGORIES = [
    "grocery",
    "restaurant",
    "coffee",
    "gas",
    "transit",
    "rideshare",
    "streaming",
    "utilities",
    "pharmacy",
    "retail",
    "online_retail",
    "travel",
    "electronics",
    "gift_cards",
    "digital_goods",
    "atm",
    "p2p_transfer",
    "crypto",
]
CHANNELS = ["card_present", "online", "atm", "transfer"]
SCENARIOS = ["card_testing", "account_takeover", "atm_drain"]

# Typical spend per category as (median dollars, lognormal sigma, channel).
CATEGORY_SPEND: dict[str, tuple[float, float, str]] = {
    "grocery": (62.0, 0.5, "card_present"),
    "restaurant": (38.0, 0.5, "card_present"),
    "coffee": (6.0, 0.3, "card_present"),
    "gas": (45.0, 0.3, "card_present"),
    "transit": (3.0, 0.4, "card_present"),
    "rideshare": (19.0, 0.5, "online"),
    "streaming": (15.0, 0.2, "online"),
    "utilities": (110.0, 0.4, "online"),
    "pharmacy": (24.0, 0.6, "card_present"),
    "retail": (55.0, 0.7, "card_present"),
    "online_retail": (42.0, 0.7, "online"),
    "travel": (380.0, 0.6, "online"),
    "electronics": (160.0, 0.7, "card_present"),
    "gift_cards": (50.0, 0.4, "card_present"),
    "digital_goods": (9.0, 0.6, "online"),
    "atm": (80.0, 0.5, "atm"),
    "p2p_transfer": (45.0, 0.7, "transfer"),
    "crypto": (120.0, 0.8, "transfer"),
}

# Relative weights over categories, and the hours of the day a persona is active.
PERSONAS: dict[str, tuple[dict[str, float], tuple[int, int]]] = {
    "commuter": (
        {"coffee": 6, "transit": 6, "grocery": 3, "restaurant": 3, "rideshare": 1, "streaming": 1,
         "utilities": 0.5, "online_retail": 1},
        (6, 21),
    ),
    "family": (
        {"grocery": 7, "gas": 4, "retail": 3, "pharmacy": 2, "restaurant": 2, "utilities": 1,
         "online_retail": 2, "streaming": 1},
        (8, 21),
    ),
    "student": (
        {"coffee": 4, "restaurant": 4, "rideshare": 3, "streaming": 2, "digital_goods": 2,
         "online_retail": 2, "p2p_transfer": 2, "grocery": 2},
        (10, 24),
    ),
    "traveler": (
        {"travel": 2, "restaurant": 5, "rideshare": 4, "coffee": 3, "retail": 2, "electronics": 1,
         "atm": 1, "online_retail": 1},
        (6, 23),
    ),
    "retiree": (
        {"grocery": 6, "pharmacy": 4, "restaurant": 2, "utilities": 1, "gas": 2, "retail": 2,
         "gift_cards": 1},
        (8, 18),
    ),
}  # fmt: skip


@dataclass(frozen=True)
class Transaction:
    user_id: int
    seq_idx: int
    ts: int  # seconds since the start of the simulation
    category: str
    amount: float
    channel: str
    persona: str
    is_fraud: int
    scenario: str  # "" for legitimate transactions


def _amount(rng: random.Random, category: str) -> float:
    median, sigma, _ = CATEGORY_SPEND[category]
    return round(max(0.5, median * math.exp(rng.gauss(0, sigma))), 2)


def _daytime(rng: random.Random, ts: int, active: tuple[int, int]) -> int:
    """Moves `ts` to a random time inside the persona's active hours on the same day."""
    day = ts - ts % 86400
    start, end = active
    return day + rng.randint(start * 3600, end * 3600 - 1)


def _fraud_burst(rng: random.Random, scenario: str, ts: int) -> list[tuple[int, str, float, str]]:
    """(ts, category, amount, channel) for one fraud scenario starting at `ts`."""
    out: list[tuple[int, str, float, str]] = []
    if scenario == "card_testing":
        t = ts
        for _ in range(rng.randint(3, 5)):
            t += rng.randint(20, 240)
            out.append((t, "digital_goods", round(rng.uniform(0.5, 2.99), 2), "online"))
        for _ in range(rng.randint(1, 3)):
            t += rng.randint(60, 900)
            cat = rng.choice(["electronics", "gift_cards"])
            out.append((t, cat, round(rng.uniform(400, 1500), 2), "online"))
    elif scenario == "account_takeover":
        t = ts - ts % 86400 + rng.randint(1 * 3600, 4 * 3600)
        for _ in range(rng.randint(2, 4)):
            t += rng.randint(120, 1800)
            cat = rng.choice(["p2p_transfer", "crypto"])
            out.append((t, cat, round(rng.uniform(250, 1200), 2), "transfer"))
    else:  # atm_drain
        t = ts - ts % 86400 + rng.randint(1 * 3600, 4 * 3600)
        for _ in range(rng.randint(3, 5)):
            t += rng.randint(60, 600)
            out.append((t, "atm", float(rng.choice([200, 300, 400, 500])), "atm"))
    return out


def _lookalike_burst(rng: random.Random, ts: int) -> list[tuple[int, str, float, str]]:
    """A legitimate burst that resembles one of the fraud scenarios."""
    kind = rng.choice(["app_store_binge", "night_crypto", "cash_day"])
    out: list[tuple[int, str, float, str]] = []
    if kind == "app_store_binge":
        t = ts
        for _ in range(rng.randint(3, 5)):
            t += rng.randint(30, 600)
            out.append((t, "digital_goods", round(rng.choice([0.99, 1.99, 2.99, 4.99]), 2), "online"))
    elif kind == "night_crypto":
        t = ts - ts % 86400 + rng.randint(0, 3 * 3600)
        out.append((t, "crypto", round(rng.uniform(100, 500), 2), "transfer"))
    else:
        t = ts
        for _ in range(2):
            t += rng.randint(300, 3600)
            out.append((t, "atm", float(rng.choice([100, 200, 300])), "atm"))
    return out


def generate_customer(rng: random.Random, user_id: int, n_txns: int, scenario: str | None) -> list[Transaction]:
    """One customer's history; `scenario` injects a fraud burst partway through."""
    persona = rng.choice(list(PERSONAS))
    weights, active = PERSONAS[persona]
    cats = list(weights)
    # Each customer leans on a few favourite categories, so habits are learnable per sequence.
    personal = [weights[c] * rng.uniform(0.3, 2.0) for c in cats]
    fraud_at = rng.randint(max(8, n_txns // 3), n_txns - 2) if scenario else -1

    rows: list[tuple[int, str, float, str, int, str]] = []
    ts = rng.randint(0, 7 * 86400)
    i = 0
    while len(rows) < n_txns:
        if i == fraud_at and scenario is not None:
            for t, cat, amt, ch in _fraud_burst(rng, scenario, ts + rng.randint(600, 6 * 3600)):
                rows.append((t, cat, amt, ch, 1, scenario))
            ts = rows[-1][0]
        if rng.random() < 0.02:
            for t, cat, amt, ch in _lookalike_burst(rng, ts + rng.randint(3600, 12 * 3600)):
                rows.append((t, cat, amt, ch, 0, ""))
            ts = rows[-1][0]
        ts += int(rng.expovariate(1 / (14 * 3600)))
        cat = rng.choices(cats, weights=personal)[0]
        rows.append((_daytime(rng, ts, active), cat, _amount(rng, cat), CATEGORY_SPEND[cat][2], 0, ""))
        ts = rows[-1][0]
        i += 1
    rows = sorted(rows[:n_txns], key=lambda r: r[0])
    return [
        Transaction(user_id, k, t, cat, amt, ch, persona, fraud, sc)
        for k, (t, cat, amt, ch, fraud, sc) in enumerate(rows)
    ]


def generate(n_customers: int, fraud_rate: float, seed: int) -> list[Transaction]:
    rng = random.Random(seed)
    out: list[Transaction] = []
    for user_id in range(n_customers):
        scenario = rng.choice(SCENARIOS) if rng.random() < fraud_rate else None
        out.extend(generate_customer(rng, user_id, rng.randint(32, 64), scenario))
    return out


def to_columns(txns: list[Transaction]) -> dict[str, list]:
    """Column-oriented form that `chalkcompute.DatasetClient.upload` accepts."""
    fields = list(Transaction.__dataclass_fields__)
    return {f: [getattr(t, f) for t in txns] for f in fields}
