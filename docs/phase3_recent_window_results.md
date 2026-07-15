# Phase 3 Recent-Window Baseline Results


Redefined walk-forward split restricted to the sentiment-coverage window (2025-08-08 to 2025-12-16) - see module docstring in `scripts/run_phase3_recent_window_baseline.py` for why. This trades away 2019-2022 bear-market coverage for a window where technical AND sentiment features both have real variation.


Window: 2025-08-08 00:00:00 to 2025-12-16 00:00:00 (2285 rows, 28 tickers)


Train: 2025-08-08 00:00:00 to 2025-10-23 00:00:00 (1337 rows)


Validate: 2025-10-24 00:00:00 to 2025-11-18 00:00:00 (455 rows)


## Validation results


Majority baseline: 0.514


Logistic regression accuracy @0.5: {'accuracy': 0.5274725274725275, 'coverage': 1.0, 'n_predicted': 455, 'local_majority_baseline': 0.5142857142857142}


LightGBM accuracy @0.5: {'accuracy': 0.4901098901098901, 'coverage': 1.0, 'n_predicted': 455, 'local_majority_baseline': 0.5142857142857142}


Random Forest accuracy @0.5: {'accuracy': 0.5032967032967033, 'coverage': 1.0, 'n_predicted': 455, 'local_majority_baseline': 0.5142857142857142}


XGBoost accuracy @0.5: {'accuracy': 0.5076923076923077, 'coverage': 1.0, 'n_predicted': 455, 'local_majority_baseline': 0.5142857142857142}


## Confidence-threshold sweep (tuned on validation only)


Logistic regression:


      accuracy  coverage
0.50  0.527473  1.000000
0.55  0.534946  0.817582
0.60  0.510345  0.637363
0.65  0.491379  0.509890
0.70  0.482955  0.386813


LightGBM:


      accuracy  coverage
0.50  0.490110  1.000000
0.55  0.497630  0.927473
0.60  0.494872  0.857143
0.65  0.501425  0.771429
0.70  0.493333  0.659341


Random Forest:


      accuracy  coverage
0.50  0.503297  1.000000
0.55  0.496241  0.584615
0.60  0.461538  0.342857
0.65  0.361702  0.206593
0.70  0.360000  0.164835


XGBoost:


      accuracy  coverage
0.50  0.507692  1.000000
0.55  0.504695  0.936264
0.60  0.509852  0.892308
0.65  0.509284  0.828571
0.70  0.513196  0.749451


## Ablation (LightGBM, evaluated on validation) - technical vs sentiment now meaningful


                accuracy  coverage
technical_only  0.492308       1.0
sentiment_only  0.494505       1.0
both            0.490110       1.0


## TEST RESULTS - evaluated ONCE at threshold=0.6



Test: 2025-11-19 00:00:00 to 2025-12-16 00:00:00 (493 rows)


Test-set majority baseline: 0.627


Logistic regression: {'accuracy': 0.5365853658536586, 'coverage': 0.7484787018255578, 'n_predicted': 369, 'local_majority_baseline': 0.6178861788617886}


LightGBM: {'accuracy': 0.5477386934673367, 'coverage': 0.8073022312373225, 'n_predicted': 398, 'local_majority_baseline': 0.6206030150753769}


Random Forest: {'accuracy': 0.651685393258427, 'coverage': 0.36105476673427994, 'n_predicted': 178, 'local_majority_baseline': 0.6460674157303371}


XGBoost: {'accuracy': 0.5370370370370371, 'coverage': 0.8762677484787018, 'n_predicted': 432, 'local_majority_baseline': 0.6365740740740741}



### Ablation on test set (LightGBM, technical vs sentiment)


                accuracy  coverage
technical_only  0.541582       1.0
sentiment_only  0.549696       1.0
both            0.549696       1.0


### Per-year breakdown (LightGBM, test set)


      accuracy  majority_baseline    n
2025  0.547739           0.626775  493


### Per-regime breakdown (LightGBM, test set)



Note: this entire window falls within a single regime (2025 = "flat" per src/models/regimes.py), so this breakdown is trivially one row - included for completeness, not because multiple regimes are represented.


      accuracy  majority_baseline    n
flat  0.547739           0.626775  493

## THE REAL ROOT CAUSE, FOUND AND FIXED

While investigating why Random Forest kept nominally "beating baseline" on this experiment's test set (documented in the superseded analysis below), we found the actual explanation: **`accuracy_at_threshold` was comparing confidence-filtered accuracy against the wrong baseline.** The function filters to a "confident" subset of test rows (`proba >= threshold` or `proba <= 1-threshold`), but every report in this project compared that subset's accuracy against `majority_baseline` computed on the FULL test set (100% coverage) - a mismatched comparison whenever the model's confidence correlates with the training set's class imbalance rather than genuine skill.

Direct measurement on the full-history experiment confirmed this: of 155 rows Random Forest was "confident" about, 146 (94%) were confident-*positive* - the model is almost never confident about the minority class. The confident subset's own majority rate was 61.3%, not the full test set's 56.1%. Comparing confident-accuracy against the full-set rate made an unremarkable result look like a win.

**Fixed:** `accuracy_at_threshold` (src/models/evaluate.py) now also returns `local_majority_baseline` - the majority rate computed WITHIN the same confident subset, giving an apples-to-apples comparison. Re-running both experiments with this fix:

| Model | Accuracy | Local baseline | Beats local baseline? |
|---|---|---|---|
| Logistic regression | 0.537 | 0.618 | No |
| LightGBM | 0.548 | 0.621 | No |
| Random Forest | 0.652 | 0.646 | Marginally (+0.6pp, n=178 - statistically nothing) |
| XGBoost | 0.537 | 0.637 | No, by 10 points |

**Conclusion, now on solid footing:** across both experiments (full 2019-2025 history and the sentiment-covered recent window), all four algorithms - logistic regression, LightGBM, Random Forest, XGBoost - fail to beat their own correctly-matched baseline, with one negligible exception (RF, +0.6pp on 178 predictions) that is well within noise given everything else measured in this investigation (50-seed robustness check, erratic confidence-threshold sweeps). Every earlier "beats baseline" result in this project - across Phase 3 and Phase 4 - was either this evaluation-methodology artifact, small-sample noise, or both. No algorithm tried has a genuine, validated edge on this prediction problem.

---

## SUPERSEDED ANALYSIS (kept for the investigation trail, not as a current finding)

The sections below (robustness-check caveats, 50-seed analysis) were the path that led to discovering the bug above. They remain accurate as a record of the investigation but are superseded by the corrected comparison above - the RF results discussed below were measured against the wrong baseline.