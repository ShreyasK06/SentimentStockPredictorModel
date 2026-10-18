# Phase 8 plan: next-day UP/DOWN news-sentiment classifier with selective prediction

Branch: `feat/phase8-news-sentiment` (from `feat/phase7-tft`, commit fb1e6e7e).
Written 2026-10-17, BEFORE any Phase 8 data was downloaded or any model was fit.
Everything in the "Pre-declared protocol" section is frozen; later changes must be
logged in the "Deviations" section at the bottom with a reason, and may never be
made after looking at held-out test numbers.

## Goal

Predict whether a stock closes UP or DOWN on the next trading day from stock news.
Target: ~85% accuracy on *confident calls only* (the model abstains unless confident).
Every confident-call accuracy is reported together with: coverage %, n calls,
the local majority baseline on the same rows, unrestricted (all-rows) accuracy,
and a 95% block-bootstrap CI. CV (walk-forward) and held-out test numbers are shown
side by side. If 85% is not reached at the pre-declared minimum coverage, we say so
and report the best honest number.

Prior expectation (stated up front): next-day direction from news is typically
~52-58% unrestricted. A confident-slice number far above that is treated as a
leakage alarm and investigated before it is reported.

## Facts checked before planning

- FNSPID on Hugging Face (`Zihan1004/FNSPID`): `Stock_news/All_external.csv` is
  5.73 GB (headline-level rows: Date, Article_title, Stock_symbol, Url, Publisher,
  Author, Article, summaries); `nasdaq_exteral_data.csv` is 23 GB (full text, skipped);
  `Stock_price/full_history.zip` 590 MB (skipped, we use yfinance).
  Download speed measured at ~15 MB/s, so All_external.csv streams in ~7 min.
  Disk has ~15 GB free, so we STREAM and filter (never keep the full CSV).
  Licence CC BY-NC 4.0 (non-commercial; noted in README).
- FNSPID timestamps are labelled "UTC" but the first rows look shifted
  (e.g. "46 Stocks Moving In Friday's Mid-Day Session" stamped 08:45 "UTC" = 04:45 ET,
  which cannot be mid-day). The intraday time is therefore NOT trusted
  (see alignment rule below).
- Finnhub free tier returns only ~250 latest articles per request and about one
  year of history; the existing `src/data/news.py` asks for one calendar year per
  call, which is why the archive starts 2025-11. Only `FINNHUB_API_KEY` exists in `.env`.
- Local prices: 28 large caps in `data/raw/prices` (2019-01-02 .. 2025-12-31).
  Older/newer prices and SPY come from yfinance.
- Hardware: 8 CPU cores, no CUDA. Docker and gh absent.
- Test command: `python -m pytest tests -q` (plain `pytest` from the repo root also
  collects `paper-trader/tests` and breaks on a package-name clash). Baseline: 160 passed.

## Pre-declared protocol (frozen)

### Universe and period
- Tickers: the 28 tickers already in `data/raw/prices` (AAPL ABBV AMZN BA BAC CAT COP
  CVX EOG GOOGL GS HD HON JNJ JPM KO MCD META MRK MS MSFT PFE PG SLB UNH WFC WMT XOM),
  fixed now, before seeing any result. The FNSPID filter also keeps the rest of
  the S&P 500 list for possible future work, but Phase 8 numbers use the 28 only.
  Ticker aliases: GOOGL also accepts FNSPID symbol GOOG; META also accepts FB.
- Period: 2012-01-01 .. last FNSPID date (expected ~2023-12/2024-01). If WP1 shows
  a year with < 20% of ticker-days having any news, the development start is moved
  later to the first year with >= 20% (rule fixed now, applied once by WP1).
- Recent window: Finnhub, last ~12 months available on the free tier, fetched in
  7-day chunks (1-day chunks for busy tickers that hit the ~250 cap).

### Timing / alignment (leakage control)
- Because FNSPID clock times are unreliable, FNSPID articles use the CONSERVATIVE
  date rule: an article stamped on calendar date D (as written in the file) is
  treated as available only after the close of D, i.e. its effective trading day
  is E = first NYSE trading day strictly after D. This is safe for any true
  publication time from D-1 ~20:00 ET to D+1 ~09:30 ET.
- Finnhub articles have true UTC epochs and use
  `src/features/trading_day.assign_effective_trading_day` (published before 16:00 ET
  on a trading day -> that day; otherwise next trading day).
- Decision time is the close of E. Label for row (ticker, E):
  `y = 1 if Close[E+1] > Close[E] else 0` (adjusted close, next NYSE session).
  Rows with a zero return are dropped. All features for row E use only news with
  effective day <= E and prices up to and including Close[E].
- Diagnostic only (never used for headline numbers): the same model with the naive
  "trust the clock" alignment, to measure how much the timing assumption matters.

### Rows ("eligible ticker-days")
- One row per (ticker, E) with >= 1 deduplicated article whose effective day is E.
- Coverage % = confident calls / eligible ticker-days in the split being evaluated.
- Dedup: per ticker, exact duplicate (normalized headline) within 3 calendar days is
  dropped; URL duplicates dropped. A headline appearing in two splits is reported.

### Splits
- Development years: start .. 2021. Final held-out TEST: 2022-01-01 .. FNSPID end.
- Walk-forward CV over development, fold years Y in {2017, 2018, 2019, 2020, 2021}:
  train = years < Y-1, validation (calibration + threshold) = year Y-1,
  evaluation = year Y. So CV numbers include threshold selection noise honestly.
- Final model: train = start .. 2020, validation = 2021 (calibration + threshold),
  TEST = 2022 .. end, evaluated ONCE by a script that refuses to run twice
  (writes a lock file; re-runs require deleting the lock, which must be reported).
- Secondary out-of-time check: Finnhub recent window, frozen final model, evaluated
  once, labelled as a different-source window.
- Hyperparameters: small pre-declared grids, picked by mean CV validation-year
  log-loss (not accuracy, not test).

### Selective prediction
- Confidence = calibrated max(p, 1-p). Calibration: isotonic (Platt if validation
  n < 1000) fitted on the validation year only.
- Threshold t chosen on VALIDATION only: maximize validation accuracy among
  thresholds whose validation coverage >= 5% AND validation n calls >= 300.
- Minimum coverage is 5% of eligible ticker-days and n >= 300 calls. This is NOT
  lowered after seeing results. If a split has fewer calls than 300 at t, the
  number is reported with that n and flagged.
- Also reported: accuracy-vs-coverage curve (validation and CV-eval), and the
  pre-declared operating points coverage = 100%, 50%, 20%, 10%, 5%.
- Baselines on the SAME rows: (a) train-majority class applied to the rows,
  (b) local majority = max(up-rate, 1-up-rate) of those rows,
  (c) controls-only model (no news features) at the same coverage.
- CI: 95% moving-block bootstrap over trading dates (block = 20 trading days,
  2000 resamples, all tickers of a date move together).

### Models
- (a) Logistic regression and LightGBM on daily per-ticker FinBERT aggregates
  + causal market/technical controls.
- (b) Frozen-FinBERT embedding head: mean of per-headline FinBERT [CLS]/pooled
  embeddings per ticker-day -> PCA (fit on train) -> logistic / small MLP.
  Full fine-tuning of FinBERT on 8 CPU cores is not feasible within bounds;
  if time allows, a bounded fine-tune of the top layers on a fixed sample is an
  optional extra, clearly labelled.
- (c) Ensemble: average of calibrated probabilities of (a) and (b).
- Ablations: news-only, controls-only, news+controls.

### Sanity checks before any number is reported
- Label-permutation test (train labels shuffled -> ~50%).
- Feature-timestamp audit: assert every feature for row E uses data <= E.
- If any confident-slice accuracy > 70% on CV: stop, inspect the rows
  (tickers, dates, headlines), check alignment and duplicates, before reporting.

## Work packages

| WP | What | Depends on | Owner |
|----|------|-----------|-------|
| WP1a | FNSPID stream-filter ingest, timestamp audit, dedupe, alignment, labels, coverage report | - | Sonnet agent |
| WP1b | New chunked Finnhub fetcher module (`src/data/finnhub_chunked.py`), keep `news.py` untouched, fetch recent window, extend prices to today | - | Sonnet agent (parallel with WP1a) |
| WP4-lib | Selective-prediction + block-bootstrap + threshold library with tests (`src/models/selective.py`) | - | Sonnet agent (parallel) |
| WP2 | FinBERT scoring + embeddings (cached, resumable, batched), daily aggregates, causal controls | WP1a | Sonnet agent |
| WP3+WP4 | Models (a)(b)(c), calibration, walk-forward CV, final test once | WP2, WP4-lib | Sonnet agent |
| WP5 | `scripts/predict_news_direction.py` live inference (Finnhub -> UP/DOWN/ABSTAIN) | WP1b, WP3 artifacts | Sonnet agent |
| WP6 | README rewrite + `docs/phase8_results.md` | WP3/4/5 | Sonnet agent |
| WP7 | Independent leakage/test-discipline/numbers review | all | Sonnet agent |

Supervisor re-runs `python -m pytest tests -q`, reviews every diff, and stages only
Phase 8 files (never `git add -A`; unrelated working-tree changes are left alone).

## Deviations

All logged BEFORE any model was fit (2026-10-18, after data ingest only).

1. **FNSPID file choice.** `All_external.csv` turned out to carry stock symbols only for its first
   ~7M rows (the rest is untagged general news) and, for the 28 tickers, ends 2020-06-11. The
   per-ticker history to 2023-12 lives in `Stock_news/nasdaq_exteral_data.csv` (23 GB, full text).
   Both files were streamed and filtered (never stored in full); URL/headline dedupe runs across both.
   Consequence: 2009-2020 news is a mix of both files, 2021-2023 comes only from the Nasdaq file
   (a source-mix shift between development and test, reported in `reports/phase8/wp1_coverage.md`).
2. **Timestamps.** 99.8% of the kept 28-ticker FNSPID articles are stamped exactly 00:00:00 (date only),
   which confirms the frozen conservative date rule is necessary.
3. **Dev start.** The frozen 20%-coverage rule gives dev_start = 2012 (2012: 22.9% of ticker-days).
4. **Price history** starts 2009-01-01 (needed for 60-day beta / 50-day MA warm-up before 2012).
5. **Primary-model rule** (made precise before the CV run): hyperparameters per family by lowest mean
   CV validation log-loss; the primary family is the one with the highest pooled CV-eval confident-call
   accuracy among a_logit / a_lgbm / b_emb / c_ensemble whose pooled CV coverage is >= 5%.
   All families are evaluated on the test once in the same locked run; the headline is the primary.
6. **Ensemble (c)** = mean of the three validation-calibrated probabilities, then re-calibrated on the
   same validation year (monotone map) before thresholding.
