# Phase 8: News-Sentiment Next-Day Direction Classifier, measured results

**Bottom line: the ~85% target was not reached, and there is no evidence of any edge.** On the held-out test (2022-01-01..2023-12-16, n=7,578 ticker-days, evaluated once) every model scores 0.517-0.520 unrestricted accuracy against a 0.520 majority-class rate. Confident-call accuracy is 0.494-0.526 at 27-43% coverage; every 95% interval includes or sits at the matched baseline. Walk-forward CV (2017-2021) agrees: pooled confident-call accuracy 0.47-0.52. No number here is anywhere near 85%, and nothing was lowered or re-run to approach it.

## What was built
- **Task:** predict UP/DOWN for the next trading day's close-to-close return from stock news, abstaining unless confident (the user chose "confident-calls only" as the meaning of the 85% target).
- **News:** FNSPID (Hugging Face) for 28 tickers: 162,704 de-duplicated articles, stamped 2009-06-22..2023-12-16. 99.8% carry a date only (no time), so an article dated D is used for the first NYSE session after D (conservative; label = sign(Close[E+1]/Close[E]-1)). Finnhub, fetched in small chunks, is used for live inference (`scripts/predict_news_direction.py`).
- **Features:** FinBERT (`ProsusAI/finbert`) headline scores aggregated per ticker-day (mean/max/min, pos/neg/neu, counts, rolling and decay, surprise), mean FinBERT embeddings, plus causal market controls (SPY, beta, technicals).
- **Models:** `a_logit`, `a_lgbm` (FinBERT aggregates), `b_emb` (embedding head; chosen primary by the pre-declared rule), `c_ensemble`, and two ablations (news-only, controls-only). Probabilities calibrated on validation; the confidence threshold is chosen on validation only, subject to a pre-declared minimum of 5% coverage and n >= 300.

## Splits
| Name | Years | Use |
|---|---|---|
| Development | 2012-2021 (start fixed by a 20% news-coverage rule) | training / selection |
| Walk-forward CV | eval years 2017-2021, validation = prior year | model and threshold selection |
| Final validation | 2021 | threshold for the test run |
| Held-out test | 2022-01-01..2023-12-16 | evaluated once, locked (`reports/phase8/TEST_LOCK.json`) |

## Walk-forward CV, pooled over eval years 2017-2021 (n=18,447)
| Model | Unrestricted acc (maj 0.516) | Coverage | Confident acc (95% CI) | Local maj |
|---|---|---|---|---|
| a_logit | 0.484 | 0.437 | 0.478 (0.455, 0.502) | 0.517 |
| a_lgbm | 0.497 | 0.139 | 0.512 (0.465, 0.561) | 0.506 |
| **b_emb (primary)** | 0.504 | 0.385 | 0.522 (0.495, 0.548) | 0.517 |
| news-only logit | 0.498 | 0.477 | 0.491 (0.472, 0.509) | 0.517 |
| controls-only logit | 0.482 | 0.380 | 0.473 (0.446, 0.500) | 0.516 |
| c_ensemble | 0.495 | 0.203 | 0.503 (0.468, 0.541) | 0.507 |

Single folds sometimes look strong (e.g. a_lgbm 0.62 on about 5% of 2017 and 2019 calls) but do not repeat across folds and vanish when pooled; this is the selection effect expected from picking thresholds on validation.

## Held-out test (2022-01-01..2023-12-16, n=7,578; one run)
| Model | Unrestricted acc (maj 0.520) | Coverage | Calls | Confident acc (95% CI) | Local maj |
|---|---|---|---|---|---|
| a_logit | 0.520 | 0.265 | 2,006 | 0.494 (0.458, 0.533) | 0.508 |
| a_lgbm | 0.517 | 0.333 | 2,521 | 0.524 (0.477, 0.570) | 0.534 |
| **b_emb (primary)** | 0.520 | 0.357 | 2,706 | 0.505 (0.473, 0.537) | 0.505 |
| news-only logit | 0.517 | 0.434 | 3,291 | 0.526 (0.507, 0.544) | 0.524 |
| controls-only logit | 0.520 | 0.327 | 2,476 | 0.495 (0.458, 0.535) | 0.506 |
| c_ensemble | 0.517 | 0.401 | 3,037 | 0.512 (0.476, 0.549) | 0.519 |

For `a_logit` the accuracy-vs-coverage curve on test *decreases* as the model gets more selective (0.512 at 50% coverage, 0.487 at 20%, 0.427 at 5%), i.e. its confidence carries no usable signal out of sample. Its confident calls are 97% UP, so it mostly encodes the up-drift of the sample.

## Diagnostics
- No result exceeded the 70% alarm threshold; the CV alarm list for the primary is empty.
- A label-permutation check and a "naive clock" alignment variant (diagnostic only) show no sign of timing leakage; the naive variant is no better (pooled unrestricted acc 0.497-0.505).
- Features at day E use only data through E's close (unit tests perturb future prices and articles and assert features are unchanged).

## Caveats
- **Source shift:** 2021-2023 news comes only from the Nasdaq file, while earlier years mix two files. Development and test therefore differ in news source.
- **Date-only timestamps** force the conservative next-session rule, which discards same-day reaction information.
- **Single seed**, 28 tickers, daily headlines only; no intraday data.
- **Finnhub** (last year) was fetched for live inference only; it was not part of training or evaluation.
- Reaching 85% would require either leakage or a very small, hand-picked slice; the pre-declared 5% / n >= 300 minimum was kept as set.
