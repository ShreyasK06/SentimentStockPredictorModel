# Phase 7: FinBERT + Temporal Fusion Transformer, measured results

Bottom line: no Phase 7 configuration (TFT stages a-e, or the LightGBM baseline on the same targets) beat the majority-class baseline on the held-out test set (2024-01-01 to 2025-12-31). The primary stages (d, e) are 2.1 and 1.7 points below it (95% block-bootstrap CIs for the mean over horizons include zero for d and e). Stages b, c and lgbm are below the baseline with CIs that exclude zero. Nothing here is a demonstrated edge. All numbers below are measured; no target was set in advance.

Split labels used throughout: **CV fold F1/F2/F3** = walk-forward validation folds (2021 / 2022 / 2023), **CV mean** = mean of F1-F3, **F3 validation** = fold F3 (2023), **held-out test** = 2024-01-01..2025-12-31, **sentiment-window test** = Nov-Dec 2025. "Acc" is directional accuracy; "maj" is the majority-class baseline of the same evaluated rows (the oracle majority of those rows, which is conservative). Unless stated, "mean" means the unweighted mean over the four horizons 1/3/5/7 trading days.

## 1. Setup

**Data.** `data/processed/tft_panel.parquet` (built by `scripts/build_tft_panel.py`): daily per-ticker panel, 28 tickers, 2019-12-31 to 2025-12-31, 42,252 rows. For CPU feasibility the runs use a fixed **16-ticker subset**: the two largest-market-cap tickers of each of 8 sectors, a rule fixed before any model result (`TICKER_SUBSET_RULE` in `src/models/tft.py`): AAPL, MSFT, AMZN, HD, GOOGL, META, JPM, BAC, JNJ, ABBV, XOM, CVX, CAT, BA, WMT, PG.

**Model.** `pytorch_forecasting.TemporalFusionTransformer`, quantile loss at quantiles 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98. Target is the daily log return (`log_ret_1d`); the decoder predicts 7 steps ahead. Horizons 1/3/5/7 are evaluated as follows: the cumulative predicted median up to step h gives the direction, and `p_up = Phi(cum_median_h / sigma_cum_h)`, where per-step sigma is taken from (q90-q10)/2.563 and combined as a sum of variances (independent-increments approximation). The label is the sign of the realised forward h-day return. Directional accuracy uses the predicted-median sign (p_up >= 0.5).

Feature sets: `price` = log_ret_1d, log_volume_chg; `technical` = price + 19 indicators (returns, RSI, MACD, volatility, Bollinger, ATR, stochastic, OBV, 52-week distances, ...); `sentiment` = technical + 8 FinBERT features (sent_mean, sent_pos_mean, sent_neg_mean, sent_neu_mean, article_count, sent_roll3, sent_roll7, has_news). Known-future calendar covariates and the static categorical `ticker` are used in every set.

**Splits (expanding window, chronological).**

| Name | Train through | Early-stopping / validation window | Used for |
|---|---|---|---|
| F1 | 2020-12-31 | 2021 | CV |
| F2 | 2021-12-31 | 2022 | CV |
| F3 | 2022-12-31 | 2023 | CV; **selection fold** for stage d hyperparameters and stage e thresholds |
| Held-out test | model trained <= 2022-12-31 (the F3 model), early-stopped on 2023 | 2024-01-01..2025-12-31 | final evaluation, once per stage |
| Sentiment window | <= 2025-09-30 | 2025-10 early stopping; test 2025-11-01..2025-12-31 | technical vs sentiment ablation |

Early stopping uses each fold's own validation window, so even F1/F2 metrics are mildly optimistic for that reason (the same applies to every stage). The test model is the F3 model; it does not train on 2023.

**Budget.** CPU, 4 threads, seed 0 only (single seed), max 15 epochs with early stopping (epochs used: 5-15 in CV and tuning, 7-9 for the test models), roughly 11.5k training windows per fold at the 16-ticker subset. Recorded run times are at the end of section 8.

**Leakage controls.** Features at day t use only data available at the close of t; news after 16:00 ET is assigned to the next trading day (`src/features/trading_day.py`); `assert_no_target_leakage` guards feature columns; LightGBM training rows satisfy t+h <= train_end so no label window overlaps evaluation; evaluation origins are identical for TFT and LightGBM. An independent code review found no leakage and no test peeking in the selection code.

**Pre-registration and test discipline.** Stage e is the primary result (stage d is primary when no threshold is applied). Stages a, b, c and lgbm are secondary/ablation references; the conclusion in section 7 rests on d and e only, since picking the best of seven test files afterwards would be a multiple-comparison peek. All CV phases for all stages were run and frozen (`frozen_d.json`, `frozen_e.json`) before any test phase. Each test phase (a, b, c, d, e, lgbm) and the sentiment-window phase were then executed exactly once, never with `--force`; none crashed, so none was rerun. Selection (feature set, hyperparameters, thresholds) used CV/F3 only.

## 2. Staged tuning table

Mean over horizons 1/3/5/7. Majority baselines are for the same rows as the accuracy beside them.

| Stage | CV mean acc (F1-F3) | CV mean maj | F3 validation acc | F3 maj | Held-out test acc | Test maj | Test minus maj |
|---|---|---|---|---|---|---|---|
| (a) price-only, untuned | 0.540 | 0.536 | 0.546 | 0.549 | 0.539 | 0.561 | -0.022 |
| (b) + technical | 0.503 | 0.536 | 0.490 | 0.549 | 0.516 | 0.561 | -0.045 |
| (c) + FinBERT sentiment | 0.534 | 0.536 | 0.560 | 0.549 | 0.543 | 0.561 | -0.019 |
| (d) tuned hyperparameters (price features) | 0.537 | 0.536 | 0.570 (selection fold) | 0.549 | 0.541 | 0.561 | -0.021 |
| (e) d + confidence threshold, confident subset only | 0.551 | 0.536 (local) | 0.600 (selection fold) | 0.545 (local) | 0.539 | 0.556 (local) | -0.017 |
| LightGBM, same target/splits | 0.521 | 0.536 | 0.542 | 0.549 | 0.524 | 0.561 | -0.037 |

Stage e rows: accuracy is on the confident subset only, so the baseline beside it is the local majority rate inside that same subset (mean mean-over-horizons coverage on test is 0.372; against the all-rows test majority 0.561 the stage-e difference would be -0.022). Stage e CV mean coverage is 0.50 over F1-F3. The F3 numbers for d and e are in-sample for selection (hyperparameters and thresholds were picked on F3), so F1/F2 and held-out test are the honest checks.

**Selection-fold-free comparison (F1 and F2 only, mean of the two folds, mean over horizons):**

| Stage | F1+F2 mean acc | F1+F2 maj |
|---|---|---|
| a | 0.536 | 0.530 |
| b | 0.509 | 0.530 |
| c | 0.520 | 0.530 |
| d | 0.521 | 0.530 |
| e (confident subset) | 0.526 | 0.531 (local) |
| lgbm | 0.511 | 0.530 |

On the folds not used for tuning, tuned d (0.521) is not better than untuned a (0.536). The F3 gain of d over a (0.570 vs 0.546) reflects selection on that fold, not generalisation.

**Frozen stage d configuration** (chosen on F3 validation only; feature set chosen by mean CV accuracy): feature set `price` (stage a had the best CV mean, 0.540, vs c 0.534 and b 0.503); hidden_size 32, attention_head_size 2, dropout 0.3, hidden_continuous_size 8, learning_rate 0.001, encoder_length 20 (config id 5).

**Frozen stage e thresholds** (per horizon, max accuracy on F3 subject to coverage >= 0.20; a prediction is made only when p_up >= t or p_up <= 1-t): h1 0.53, h3 0.57, h5 0.60, h7 0.55.

### Per-horizon: stage d (no threshold)

| Horizon | CV mean acc | CV maj | F3 acc | F3 maj | Test acc | Test maj | Test minus maj | 95% block-bootstrap CI of test acc | CI of (acc - maj) | n (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1d | 0.517 | 0.521 | 0.530 | 0.528 | 0.520 | 0.536 | -0.017 | [0.508, 0.531] | [-0.027, -0.006] | 7936 |
| 3d | 0.539 | 0.536 | 0.568 | 0.545 | 0.543 | 0.560 | -0.017 | [0.522, 0.562] | [-0.035, 0.004] | 7936 |
| 5d | 0.543 | 0.540 | 0.587 | 0.557 | 0.547 | 0.571 | -0.024 | [0.522, 0.572] | [-0.056, 0.013] | 7936 |
| 7d | 0.550 | 0.548 | 0.596 | 0.564 | 0.553 | 0.579 | -0.026 | [0.521, 0.583] | [-0.060, 0.016] | 7936 |
| Mean | 0.537 | 0.536 | 0.570 | 0.549 | 0.541 | 0.561 | -0.021 | [0.521, 0.559] | [-0.042, 0.005] | |

(F3 n = 3904 per horizon. The row-level difference of 1d is -0.017 in the point estimate; see the CI column.) The train-period majority class (the class a forecaster could have known) is "up" for every horizon (train share up: 0.517/0.534/0.537/0.547), and the test "up" share equals the oracle majority baseline (0.536/0.560/0.571/0.579), so on the test set the train-majority baseline and the oracle majority baseline coincide.

### Per-horizon: stage e (thresholded, confident subset)

| Horizon | Threshold | Split | Acc (confident) | Coverage | Local maj (same rows) | All-rows maj | n predicted |
|---|---|---|---|---|---|---|---|
| 1d | 0.53 | F1 | 0.522 | 0.68 | 0.532 | 0.534 | 2685 |
| 1d | 0.53 | F2 | 0.505 | 0.48 | 0.501 | 0.501 | 1897 |
| 1d | 0.53 | F3 (selection) | 0.556 | 0.38 | 0.531 | 0.528 | 1502 |
| 1d | 0.53 | Test | 0.527 | 0.379 | 0.537 | 0.536 | 3009 |
| 3d | 0.57 | F1 | 0.550 | 0.52 | 0.551 | 0.558 | 2057 |
| 3d | 0.57 | F2 | 0.518 | 0.46 | 0.510 | 0.506 | 1813 |
| 3d | 0.57 | F3 (selection) | 0.601 | 0.26 | 0.541 | 0.545 | 1016 |
| 3d | 0.57 | Test | 0.536 | 0.256 | 0.548 | 0.560 | 2030 |
| 5d | 0.60 | F1 | 0.558 | 0.42 | 0.569 | 0.562 | 1635 |
| 5d | 0.60 | F2 | 0.508 | 0.43 | 0.503 | 0.501 | 1692 |
| 5d | 0.60 | F3 (selection) | 0.624 | 0.21 | 0.538 | 0.557 | 809 |
| 5d | 0.60 | Test | 0.529 | 0.210 | 0.552 | 0.571 | 1664 |
| 7d | 0.55 | F1 | 0.525 | 0.71 | 0.580 | 0.578 | 2813 |
| 7d | 0.55 | F2 | 0.525 | 0.74 | 0.502 | 0.503 | 2908 |
| 7d | 0.55 | F3 (selection) | 0.620 | 0.64 | 0.568 | 0.564 | 2501 |
| 7d | 0.55 | Test | 0.565 | 0.642 | 0.585 | 0.579 | 5093 |

Test, confident subset, with block-bootstrap 95% CIs: 1d acc 0.527 [0.508, 0.549], minus local maj -0.010 [-0.023, 0.002]; 3d 0.536 [0.503, 0.569], -0.012 [-0.046, 0.019]; 5d 0.529 [0.474, 0.586], -0.023 [-0.101, 0.039]; 7d 0.565 [0.526, 0.604], -0.020 [-0.063, 0.024]. Mean over horizons: confident-subset accuracy 0.539 vs local majority 0.556 (-0.017), mean coverage 0.372.

Thresholds were chosen by raw accuracy, not by accuracy minus the local baseline, and high thresholds can pick lopsided subsets (for example the F3 5d subset at coverage 0.21), so every thresholded accuracy is shown next to its local majority baseline. The F3 rows show apparent gains of +2.5 to +8.7 points over the local baseline; these do not appear on F1, F2 or test (F3 is the fold the thresholds were chosen on). Without a threshold, stage e's test row equals stage d (same model): 0.541 vs 0.561.

### Per-horizon: LightGBM baseline

| Horizon | CV mean acc | CV maj | F3 acc | F3 maj | Test acc | Test maj | Test minus maj | 95% CI of test acc | CI of (acc - maj) |
|---|---|---|---|---|---|---|---|---|---|
| 1d | 0.502 | 0.521 | 0.508 | 0.528 | 0.512 | 0.536 | -0.024 | [0.500, 0.524] | [-0.043, -0.006] |
| 3d | 0.516 | 0.536 | 0.532 | 0.545 | 0.525 | 0.560 | -0.035 | [0.507, 0.542] | [-0.054, -0.015] |
| 5d | 0.526 | 0.540 | 0.558 | 0.557 | 0.520 | 0.571 | -0.051 | [0.496, 0.546] | [-0.083, -0.019] |
| 7d | 0.541 | 0.548 | 0.569 | 0.564 | 0.539 | 0.579 | -0.039 | [0.514, 0.569] | [-0.070, -0.006] |
| Mean | 0.521 | 0.536 | 0.542 | 0.549 | 0.524 | 0.561 | -0.037 | [0.506, 0.542] | [-0.058, -0.015] |

LightGBM: 200 trees, num_leaves 15, learning rate 0.03, all sentiment+technical features, fixed parameters (not tuned), same rows and horizons.

### Held-out test CIs for every stage (secondary reference)

Mean over horizons, 95% block-bootstrap CI (20-trading-day blocks of origin dates, 2000 resamples, whole blocks across all tickers). These describe resampling uncertainty of the test estimate, not model-selection uncertainty.

| Stage | Test acc | CI | Test maj | Acc - maj | CI of (acc - maj) |
|---|---|---|---|---|---|
| a | 0.539 | [0.522, 0.557] | 0.561 | -0.022 | [-0.045, 0.002] |
| b | 0.516 | [0.497, 0.534] | 0.561 | -0.045 | [-0.074, -0.016] |
| c | 0.543 | [0.520, 0.564] | 0.561 | -0.019 | [-0.037, -0.001] |
| d | 0.541 | [0.521, 0.559] | 0.561 | -0.021 | [-0.042, 0.005] |
| e (all rows, same model as d) | 0.541 | [0.522, 0.561] | 0.561 | -0.021 | [-0.043, 0.006] |
| lgbm | 0.524 | [0.506, 0.542] | 0.561 | -0.037 | [-0.058, -0.015] |

The naive binomial standard error at n = 7936 is 0.0056 per horizon; the bootstrap half-widths are roughly 3 to 4 times larger because horizons overlap and tickers are correlated, so a binomial CI would badly understate uncertainty. The bootstrap script (`scripts/phase7_bootstrap_ci.py`) regenerates the test predictions by inference only from the already-trained F3 checkpoints (or the deterministic LightGBM refit) and verifies they match `test_<stage>.json` to 1e-9 before writing `bootstrap_ci.json`; it does not modify the test result files.

## 3. Tuning log (stage d)

Selection rule: maximise mean directional accuracy over horizons on F3 validation (2023); ties go to lower pinball loss. Bounded random search: the default config plus 5 seeded random draws from a 3x2x2x2 grid. Feature set was chosen first by mean CV accuracy over F1-F3: price 0.540, sentiment 0.534, technical 0.503. F3 maj = 0.549 for all rows below.

| Config | hidden | dropout | lr | encoder | F3 val mean acc | F3 pinball | Epochs | Chosen |
|---|---|---|---|---|---|---|---|---|
| 0 (untuned default) | 16 | 0.1 | 1e-3 | 30 | 0.546 | 0.00318 | 7 | |
| 1 | 8 | 0.3 | 1e-3 | 20 | 0.554 | 0.00315 | 15 | |
| 2 | 8 | 0.3 | 3e-3 | 20 | 0.567 | 0.00316 | 11 | |
| 3 | 8 | 0.1 | 3e-3 | 20 | 0.553 | 0.00317 | 7 | |
| 4 | 16 | 0.1 | 1e-3 | 60 | 0.549 | 0.00315 | 9 | |
| 5 | 32 | 0.3 | 1e-3 | 20 | 0.570 | 0.00316 | 8 | yes |

The six configs span only 0.546-0.570 on F3, a 2.4 point range, comparable to the run-to-run noise of about 3 points seen in section 4. The selected config's F3 value (0.570), and d's F3 numbers throughout this document, are in-sample for selection; selecting the maximum of six noisy numbers also inflates it. On held-out test the tuned model (0.541) is indistinguishable from the untuned one (0.539).

## 4. Sentiment ablation

**Facts about the sentiment data.** All eight sentiment features are exactly zero for every row from 2019-12-31 to 2025-07-31. News (Finnhub) exists only from 2025-08-08. In the full 28-ticker panel 0% of rows have news before August 2025 and 27% of rows in Aug-Dec 2025; **in the 16-ticker subset used here, no row has news before November 2025** (has_news share by month: Aug-Oct 2025 = 0, Nov = 0.17, Dec = 0.72). Consequently every CV fold (2021-2023) and the held-out test through July 2025 see constant-zero sentiment inputs.

**Stage b vs c in CV and test.** The sentiment features are constant in those windows, so b (technical) vs c (technical + sentiment) differ only by extra constant-zero inputs to the variable-selection network, which changes initialisation and the optimisation path. The gap below is therefore a measurement of training noise, not a sentiment effect:

| Comparison | CV mean acc | F3 val acc | Test acc |
|---|---|---|---|
| b (technical) | 0.503 | 0.490 | 0.516 |
| c (technical + sentiment, all-zero in these windows) | 0.534 | 0.560 | 0.543 |
| c minus b | +0.031 | +0.070 | +0.027 |

A difference of about 3 points in CV mean (7 on F3) between two models that see identical information indicates the size of single-seed training noise. It must not be read as evidence that sentiment helps. Stage c's CV mean (0.534) is also below stage a's (0.540), and `frozen_d.json` has feature set `price` because stage a had the best CV mean. Differences between a, b and c are within training noise, so the ordering a > c > b should not be interpreted as a ranking of feature sets, and the feature-set choice feeding stage d was effectively a noise-driven pick.

**Sentiment-window experiment (Nov-Dec 2025, evaluated once).** Technical vs sentiment feature sets, both with the frozen stage-d hyperparameters, trained through 2025-09-30, early-stopped on 2025-10, tested on 2025-11-01..2025-12-31 (656 rows, 560 usable per horizon, 16 tickers). Test rows with news: 46.3%. Training rows with news: **0.0%** (none in the 16-ticker subset before November 2025). The sentiment model therefore trained on all-zero sentiment inputs and was tested on non-zero ones; it had no opportunity to learn any sentiment relationship. The technical-vs-sentiment difference below is therefore training noise, not a sentiment effect, and the experiment has essentially no power.

Why the news is missing (root cause): `src/data/news.py::fetch_company_news` requests one calendar-year chunk per call (`_year_chunks`), and Finnhub returns only the most recent roughly 250 articles per request, so each ticker retains only its latest weeks of 2025 news (about 225-248 archived articles per ticker). First date with any news per ticker in the 16-ticker subset: PG 2025-11-05, CAT 2025-11-06, ABBV 2025-11-10, HD 2025-11-24, XOM 2025-11-28, CVX 2025-12-01, BAC 2025-12-03, BA 2025-12-04, JNJ 2025-12-04, WMT 2025-12-08, JPM 2025-12-17, AAPL 2025-12-19, META 2025-12-19, AMZN 2025-12-22, GOOGL 2025-12-22, MSFT 2025-12-22. Only two tickers outside the subset (EOG, SLB) have news before October.

**No experiment in Phase 7 could test whether FinBERT sentiment helps; the sentiment ablation is inconclusive by construction, not negative.**

| Feature set | Mean acc (sentiment-window test) | Maj (same rows) | Acc minus maj | 1d | 3d | 5d | 7d | Pinball (all steps) |
|---|---|---|---|---|---|---|---|---|
| technical | 0.525 | 0.534 | -0.009 | 0.511 | 0.537 | 0.523 | 0.530 | 0.00310 |
| sentiment | 0.534 | 0.534 | 0.000 | 0.521 | 0.527 | 0.534 | 0.554 | 0.00310 |

Majority baselines per horizon: 0.514, 0.523, 0.539, 0.559. The 0.009 difference in mean accuracy between the feature sets (n = 560 per horizon, overlapping horizons) is far inside noise: a naive binomial 95% half-width at n = 560 is about 4 points per horizon (an understatement), and the b-vs-c comparison shows about 3 points of training noise from a single seed. Seeds were not varied, so no seed spread is available. **FinBERT sentiment showed no measurable benefit in any setting, but no setting contained usable sentiment training data, so this says nothing about whether sentiment could help.**

## 5. Forecast quality

Pinball loss is the mean over all 7 quantiles and decoder steps (log-return units). Coverage is the share of realised step returns inside the model's [q10, q90] band (expected 80%), for each horizon's decoder step.

| Stage | F1 pinball | F2 pinball | F3 pinball | Test pinball | F3 q10-q90 coverage (h1/h3/h5/h7) | Test q10-q90 coverage (h1/h3/h5/h7) |
|---|---|---|---|---|---|---|
| a | 0.00343 | 0.00469 | 0.00318 | 0.00331 | 0.834/0.839/0.844/0.843 | 0.817/0.826/0.827/0.825 |
| b | 0.00351 | 0.00469 | 0.00317 | 0.00329 | 0.825/0.825/0.834/0.845 | 0.812/0.815/0.823/0.833 |
| c | 0.00340 | 0.00470 | 0.00316 | 0.00330 | 0.861/0.847/0.846/0.850 | 0.840/0.833/0.831/0.832 |
| d (= e) | 0.00351 | 0.00463 | 0.00316 | 0.00330 | 0.857/0.845/0.838/0.834 | 0.845/0.832/0.824/0.823 |

All stages are over-covered by about 1 to 6 points (intervals slightly too wide) and pinball losses differ by less than 3% between stages. LightGBM produces no quantile forecasts. Interval calibration is reasonable; it does not translate into directional accuracy above the majority class.

## 6. Comparison with prior phases

Phases 3-5 used a different label (10-day +/-2% first-touch) and different universe and windows, so their numbers are **not directly comparable** with Phase 7's directional labels and are listed for context only. Phase 3 (`docs/phase3_results.md`; test majority baseline 0.561) found all four algorithms below their own local majority baseline on the 2019-2025 walk-forward test; Phase 5 (`docs/phase5_cross_sectional_results.md`) found no edge in any configuration. Phase 7 reaches the same qualitative result on a different label: the same-target LightGBM baseline and all TFT variants are below the majority class on held-out test.

## 7. Conclusion

No configuration beats the majority-class baseline on held-out test (2024-2025). Primary result, stage d: test accuracy 0.541 vs majority 0.561 (-0.021, 95% block-bootstrap CI [-0.042, +0.005]). Stage e on the confident subset: 0.539 vs local majority 0.556 (-0.017), with per-horizon differences from -0.010 to -0.023 and every per-horizon CI including zero. The apparent F3 gains of stages d and e (0.570 and 0.600) are selection effects on the fold used for tuning and do not appear on F1, F2 or the test; tuning (d) did not beat the untuned price-only model (a) on F1/F2 or test, and the confidence threshold (e) did not lift accuracy above its local baseline on test. The FinBERT sentiment question is unanswered (inconclusive by construction): sentiment is zero in every training and validation window. Stages b, c and lgbm are below the baseline with (acc - maj) CIs that exclude zero on the negative side.

Limitations: single seed; 16 of 28 tickers; sentiment history only from August 2025 (none for the subset before November 2025); early stopping on each fold's validation window; the test baseline (0.561) is higher than the CV baseline (0.536) because 2024-2025 had more up days, and the models predict "up" for 75% to 91% of test rows (stages a, c, d), so their accuracy is mostly the up-day rate with some noise.

Next steps: fix the news history. `fetch_company_news` should use weekly or daily chunks instead of calendar-year chunks, within the free Finnhub tier's roughly one-year history window (or use a paid tier for a longer history), then rebuild the panel and rerun stages c and the sentiment window. Until then no Phase 7 result speaks to sentiment. `news.py` was not changed in this work package.

## 8. Reproduce

```
python scripts/build_tft_panel.py
python scripts/run_phase7_tft.py --stage a --phase cv
python scripts/run_phase7_tft.py --stage b --phase cv
python scripts/run_phase7_tft.py --stage c --phase cv
python scripts/run_phase7_tft.py --stage lgbm --phase cv
python scripts/run_phase7_tft.py --stage d --phase cv   # tunes 6 configs on F3, freezes, runs F1/F2
python scripts/run_phase7_tft.py --stage e --phase cv   # freezes thresholds on F3
# only after every cv phase is complete and frozen_d.json / frozen_e.json exist; each once, never --force:
for s in a b c d e lgbm; do python scripts/run_phase7_tft.py --stage $s --phase test; done
python scripts/run_phase7_tft.py --phase sentiment_window
python scripts/phase7_bootstrap_ci.py
python -m pytest tests -q
```

Result files: `reports/phase7/{cv_a,cv_b,cv_c,cv_d,cv_e,cv_lgbm,test_a,test_b,test_c,test_d,test_e,test_lgbm,sentiment_window,tuning_d,frozen_d,frozen_e,bootstrap_ci}.json`. Models and cached predictions are in `models/phase7/full/` (gitignored).

Recorded compute (sum of `wall_time_s` fields; stages a and b ran concurrently, c and lgbm followed a; d and e ran alone): cv a 669 s, cv b 964 s, cv c 723 s, cv d 1207 s (includes tuning), cv e under 1 s (uses cached d models), cv lgbm 6 s; test phases 13-16 s each for TFT stages (inference on cached F3 checkpoints) and 4 s for lgbm; sentiment window 995 s (4 model fits). Total about 77 minutes of recorded compute.
