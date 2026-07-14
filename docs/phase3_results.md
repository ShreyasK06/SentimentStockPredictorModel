# Phase 3 Baseline Results

Train: 2019-03-14 00:00:00 to 2022-12-30 00:00:00 (24675 rows)

Validate: 2023-01-03 00:00:00 to 2023-12-29 00:00:00 (6288 rows)


## Validation-year (2023) results

Logistic regression majority baseline: 0.536

Logistic regression accuracy @0.5: {'accuracy': 0.5353053435114504, 'coverage': 1.0, 'n_predicted': 6288}

LightGBM accuracy @0.5: {'accuracy': 0.5392811704834606, 'coverage': 1.0, 'n_predicted': 6288}


## Confidence-threshold sweep (tuned on validation only)

Logistic regression:

      accuracy  coverage
0.50  0.535305  1.000000
0.55  0.563725  0.389313
0.60  1.000000  0.000159
0.65       NaN  0.000000
0.70       NaN  0.000000

LightGBM:

      accuracy  coverage
0.50  0.539281  1.000000
0.55  0.555847  0.606552
0.60  0.584241  0.306775
0.65  0.628910  0.147424
0.70  0.635220  0.075859

## Ablation (LightGBM, evaluated on validation year)

**CAVEAT (added after investigating the identical technical_only/both accuracy below):**
Phase 1's news pull (Finnhub free tier) only returned recent headlines regardless of the
requested 2019-2025 date range — confirmed via direct API test: historical date ranges
(2019, 2022) return zero articles; only recent ranges return data. As a result, the
`sentiment_mean`/`sentiment_std`/`sentiment_roll3`/`sentiment_roll7`/`headline_count`
columns are constant zero for the entire train (2019-2022) and validate (2023) periods
used below. `technical_only` and `both` are therefore mathematically guaranteed to
produce identical predictions (LightGBM cannot split on a constant column) — this is
**not** evidence that sentiment lacks predictive value; it's an artifact of zero
sentiment coverage in this period. See the "Supplementary: Sentiment-Coverage Sub-Period
Analysis" section below for a (small-sample) attempt to test sentiment where real
coverage actually exists.

                accuracy  coverage
technical_only  0.539281       1.0
sentiment_only  0.535623       1.0
both            0.539281       1.0

## TEST SET RESULTS (2024-2025) - evaluated ONCE at threshold=0.6

Test (untouched until this run): 2024-01-02 00:00:00 to 2025-12-16 00:00:00 (12542 rows)

Test-set majority baseline: 0.561

Logistic regression: {'accuracy': 0.5636363636363636, 'coverage': 0.013155796523680433, 'n_predicted': 165}

LightGBM: {'accuracy': 0.5215123859191656, 'coverage': 0.3057726040503907, 'n_predicted': 3835}


### Per-year breakdown (LightGBM, test set)

      accuracy  majority_baseline     n
2024  0.542242           0.549779  6328
2025  0.502964           0.571612  6214

### Per-regime breakdown (LightGBM, test set)

      accuracy  majority_baseline     n
bull  0.542242           0.549779  6328
flat  0.502964           0.571612  6214

---

# Supplementary: Sentiment-Coverage Sub-Period Analysis

Exploratory, small-sample analysis - NOT part of the official walk-forward test-set evaluation. See module docstring in `scripts/run_sentiment_coverage_analysis.py` for why this exists: Phase 1's news pull only captured recent headlines (Finnhub free-tier limitation), so real sentiment coverage (headline_count > 0) exists only in a narrow window, entirely within the already-evaluated test period.

Coverage window: 2025-08-08 00:00:00 to 2025-12-16 00:00:00 (476 rows, 22 of 28 tickers)

Sub-split (70/30 by row count, chronological, NOT the official train/validate/test split): train 2025-08-08 00:00:00 to 2025-12-05 00:00:00 (333 rows), eval 2025-12-05 00:00:00 to 2025-12-16 00:00:00 (143 rows)

CAVEAT: 333 training rows is a very small sample for LightGBM - treat any signal difference here as a weak, exploratory hint, not a confirmed finding.

Eval-set majority baseline: 0.580

| variant | accuracy@0.5 | n |
|---|---|---|
| technical_only | 0.5455 | 143 |
| both | 0.5734 | 143 |
