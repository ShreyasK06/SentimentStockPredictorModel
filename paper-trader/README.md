# Paper Trading Pipeline

A live paper-trading system built around the direction/sentiment models developed
in the parent research repo. The model has never shown an edge over the always-up
baseline in any research phase (see `../docs/phase*.md`) — this system exists to
(1) provide production-grade infrastructure that is valuable independent of model
performance, and (2) give future model iterations (longer horizons, volatility
prediction, post-event windows) an honest, forward-tested evaluation harness
instead of another backtest.

All trading is through [Alpaca's paper API](https://alpaca.markets/) — **no real
money is ever involved.**

## Architecture

```
                daily.yml (GH Actions cron, ~30 min after market open ET)
                              │
                              ▼
                     python -m paper_trader.pipeline
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                      ▼
  data layer            model integration       strategy & execution
  (price/news           (feature_store,          (decision, ledger,
  fetch + degradation    predictor,               alpaca_client)
  policy, run_log)       sentiment_model)              │
        │                     │                        ▼
        └──────────┬──────────┘              logs/trades/{strategy}.jsonl
                    ▼
        logs/predictions/{strategy}.jsonl
                    │
                    ▼
          evaluation harness (outcomes.py resolves
          matured predictions → metrics.py computes
          hit rate / baselines / Sharpe / calibration
          → report.py writes reports/latest.md)
```

Every module wraps the parent repo's existing `src/` package (data fetchers,
feature engineering, model training) rather than duplicating it — the direction
model is retrained each run on a walk-forward-safe feature matrix (frozen
research history + live-appended days), the sentiment signal is the FinBERT
rolling score already computed by `src/features/sentiment.py`.

**Package naming note**: the plan's tree names this package `src/`, but this
directory's Python package is `paper_trader/`, not `src/`. The parent repo
already has a top-level `src` package that this project imports from
(`from src.data.prices import fetch_ohlcv`, etc.) — two different packages
can't both be importable as top-level `src` in the same interpreter, so the
paper-trader package was named `paper_trader` to avoid the collision. Every
other path in the plan (`logs/`, `reports/`, `config.yaml`, `tests/`) is
unchanged.

## Running locally

```bash
cd paper-trader
pip install -r ../requirements.txt
cp ../.env.example ../.env   # fill in ALPACA_KEY_ID / ALPACA_SECRET_KEY / FINNHUB_API_KEY

python -m paper_trader.pipeline --dry-run              # full pipeline, no orders submitted
python -m paper_trader.pipeline --dry-run --date 2025-12-15   # pin a specific as_of_date
python -m paper_trader.pipeline                          # live paper trading run
python -m pytest tests -v                                 # unit + integration tests
```

Without `ALPACA_KEY_ID`/`ALPACA_SECRET_KEY` configured, the pipeline still runs
end-to-end: it falls back to virtual per-strategy ledgers (the same mechanism
`log_only` strategies use) instead of crashing.

## How the decision rule works

A ticker gets a **paper trade** only when, for the `direction-daily` control
strategy: `confidence >= 0.6` AND the standalone sentiment signal agrees with
the model's direction. Down-signals are skipped by default (`execution.allow_short:
false` in `config.yaml`) rather than shorted. Most days produce zero trades —
this is intended, not a bug; it's the whole point of confidence-gating.

Every universe ticker gets a **logged prediction** every run regardless of
whether it becomes a trade — non-trades are evaluation data too (see
`logs/predictions/direction-daily.jsonl`).

## Multi-strategy support

Three named strategies run side by side from one pipeline run, each with its
own prediction log, trade ledger, and virtual sub-ledger equity (`logs/*/​{strategy}.jsonl`):

| Strategy | Mode | What it is |
|---|---|---|
| `direction-daily` | `trade` | The control — current confidence-gated direction model. Expected ≈ baseline. |
| `volatility` | `log_only` | **Placeholder**: currently reuses the direction model's up/down prediction at a 5-day horizon, log-only. A true high-move classifier (predicting magnitude, not direction) hasn't been built yet — this strategy exists so the plumbing (logging, evaluation, sub-ledger) is ready the day it is. |
| `post-event` | `log_only` | **Placeholder**: same underlying model at a 3-day horizon. A real earnings/news-window filter hasn't been built yet either — same rationale. |

`reports/latest.md` compares all three on the identical date range every run.

## Evaluation harness

- **Outcomes**: once a prediction's horizon has elapsed, `evaluation/outcomes.py`
  resolves it against realized closes → `logs/outcomes/{strategy}.jsonl`.
- **Metrics** (`evaluation/metrics.py`): directional hit rate (all predictions
  and traded-only, separately), two baselines — coin flip (50%) and always-up
  (the realized fraction of up-days over the same range, usually >50% for
  large caps — the honest bar) — each with a binomial significance p-value,
  portfolio cumulative return / max drawdown / annualized Sharpe, and
  calibration by confidence decile.
- **Report** (`evaluation/report.py`): regenerates `reports/latest.md` every
  run with plain-language conclusions, including negative ones (e.g. "hit rate
  51.2% vs. always-up baseline 53.1%, p=0.41 — no evidence of edge"). See
  [`reports/latest.md`](reports/latest.md) once the pipeline has run.
- **Drift alert**: if the rolling 30-day hit rate falls below the always-up
  baseline minus a margin, the report surfaces a prominent warning. No
  auto-retraining — flagged for human review only.

## Known limitations (stated plainly, per the honesty requirement)

1. **News backfill gap**: each run fetches a 14-day news lookback (enough for
   the 3/7-day rolling sentiment features at the tail). On the very first live
   run, any gap between the frozen historical matrix's end date and
   `as_of_date - 14 days` gets neutral (0.0) sentiment rather than true
   backfilled sentiment, since backfilling months of FinBERT scoring on every
   run would be prohibitively slow. This only affects that one-time catch-up
   window, not steady-state daily runs.
2. **Label-resolution embargo**: the frozen historical matrix
   (`data/processed/feature_matrix_extended.parquet`) was built once,
   non-walk-forward. `feature_store._label_safe_cutoff` embargoes any
   historical row within `horizon` trading days of `as_of_date` from training,
   so an `as_of_date` chosen inside the frozen matrix's own range (as in
   `tests/test_pipeline_dry_run.py`) can't leak the future into itself. A real
   live `as_of_date` (always long past the frozen matrix's end) is unaffected.
3. **`volatility` / `post-event` are placeholders** — see the table above.
4. **`max_positions` is enforced per-strategy**, not globally across
   strategies — currently equivalent to global since only one strategy is
   `mode: trade` by default.

## Tests

`tests/` covers the decision rule (threshold/agreement edge cases), position
sizing caps, idempotency, the leakage guard, outcome-resolution math, the
degradation policy, metrics (binomial significance, drift alert), a
mocked-Alpaca integration test (fill polling, timeout/cancel, reconciliation),
and a full offline dry-run of the pipeline on a fixture day. `--dry-run` runs
in CI on every PR that touches `paper-trader/`, `src/`, or `requirements.txt`
(`.github/workflows/pr-checks.yml`).
