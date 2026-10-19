# Training a transaction sequence model with PyTorch on Chalk

A small causal transformer that reads a customer's card transactions in order, predicts the
next one, and scores each transaction for fraud. The data is synthetic and the model is tiny,
about 350k parameters. The point is the workflow, not the accuracy:

```
data.py ──▶ Chalk dataset ──▶ training run (GPU, PyTorch) ──▶ Chalk volume ──▶ serving app (CPU)
```

| File | Role |
| --- | --- |
| `data.py` | Generates customers with spending personas, and injects three fraud scenarios into about 12% of them. |
| `model.py` | Tokenisation, the transformer, the training loop, evaluation, and per-transaction explanations. |
| `train.py` | Uploads the data as a Chalk dataset and submits a `chalkcompute.training` run on an L4. |
| `serve.py`, `index.html` | FastAPI app and page that score unseen customers with the trained weights. |
| `deploy.py` | Deploys the serving app as a CPU scaling group. |

## The data

Every customer gets a persona (commuter, family, student, traveler or retiree) that decides
what they buy, how much, and at what time of day. A fraud scenario inserts a short burst of
labelled transactions partway through a customer's history:

- **Card testing:** several sub-$3 online "digital goods" charges minutes apart, then large
  electronics or gift-card purchases.
- **Account takeover:** late-night peer-to-peer and crypto transfers of a few hundred dollars.
- **ATM drain:** back-to-back ATM withdrawals in the small hours.

Legitimate customers produce look-alikes of each pattern: app-store binges, an occasional
late-night crypto buy, and two ATM withdrawals on a cash day. So no single transaction gives
the fraud away. The model has to use what comes before and after it.

## The model

Each transaction becomes five tokens: category, amount bucket, channel, hour of day, and time
since the previous transaction. Their embeddings are summed into one vector per transaction. A
causal `nn.TransformerEncoder` reads the sequence left to right, so position *t* only sees
transactions *0..t*. Two kinds of head sit on top:

- **Next-action heads:** predict transaction *t+1*'s category, amount bucket and channel. They
  are trained self-supervised on every customer, and they give a free "surprise" score: how
  unlikely each transaction was given the customer's own history.
- **Fraud head:** scores whether transaction *t* is fraudulent, trained on the labelled bursts
  with a class-balancing `pos_weight`.

Training reports, on held-out customers, next-category accuracy and fraud average precision
(area under the precision-recall curve). Compare the latter with the fraud base rate, around
1%.

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule an L4.
Credentials come from the SDK’s ambient credentials from `chalk login`, or from `CHALK_API_SERVER`, `CHALK_CLIENT_ID`,
`CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID`.

```bash
./train.py # generate 20k customers, upload them, train on an L4, print per-epoch metrics
./deploy.py # serve the trained weights; prints the page URL
```

`train.py` wraps `train_sequence_model(df, config)` with `chalkcompute.training(dataset=...,
gpu="nvidia-l4")`. Inside the training container, Chalk loads the dataset's latest revision as
a Polars LazyFrame and passes in the config. The function trains, then writes `model.pt` and
`summary.json` to the `transaction-sequence-model` volume. The serving app reads them at
startup with its own workload identity (`chalk_identity=True`).

To iterate without a GPU, run the same code on CPU:

```python
import dataclasses, data, model

rows = [dataclasses.asdict(t) for t in data.generate(4000, 0.12, seed=1)]
cfg = model.TrainConfig(epochs=6, batch_size=64, lr=2e-3, d_model=64, n_layers=2, n_heads=4, fraud_weight=1.0)
trained, summary = model.train_model(rows, cfg, print)
```

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
