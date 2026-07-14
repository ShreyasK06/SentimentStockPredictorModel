# Phase 3 Recent-Window Baseline Results


Redefined walk-forward split restricted to the sentiment-coverage window (2025-08-08 to 2025-12-16) - see module docstring in `scripts/run_phase3_recent_window_baseline.py` for why. This trades away 2019-2022 bear-market coverage for a window where technical AND sentiment features both have real variation.


Window: 2025-08-08 00:00:00 to 2025-12-16 00:00:00 (2285 rows, 28 tickers)


Train: 2025-08-08 00:00:00 to 2025-10-23 00:00:00 (1337 rows)


Validate: 2025-10-24 00:00:00 to 2025-11-18 00:00:00 (455 rows)


## Validation results


Majority baseline: 0.514


Logistic regression accuracy @0.5: {'accuracy': 0.512087912087912, 'coverage': 1.0, 'n_predicted': 455}


LightGBM accuracy @0.5: {'accuracy': 0.5098901098901099, 'coverage': 1.0, 'n_predicted': 455}


## Confidence-threshold sweep (tuned on validation only)


Logistic regression:


      accuracy  coverage
0.50  0.512088  1.000000
0.55  0.504587  0.718681
0.60  0.500000  0.439560
0.65  0.473214  0.246154
0.70  0.380282  0.156044


LightGBM:


      accuracy  coverage
0.50  0.509890  1.000000
0.55  0.513382  0.903297
0.60  0.516043  0.821978
0.65  0.501484  0.740659
0.70  0.500000  0.668132


## Ablation (LightGBM, evaluated on validation) - technical vs sentiment now meaningful


                accuracy  coverage
technical_only  0.483516       1.0
sentiment_only  0.494505       1.0
both            0.509890       1.0


## TEST RESULTS - evaluated ONCE at threshold=0.6



Test: 2025-11-19 00:00:00 to 2025-12-16 00:00:00 (493 rows)


Test-set majority baseline: 0.627


Logistic regression: {'accuracy': 0.5771812080536913, 'coverage': 0.6044624746450304, 'n_predicted': 298}


LightGBM: {'accuracy': 0.5366430260047281, 'coverage': 0.8580121703853956, 'n_predicted': 423}



### Ablation on test set (LightGBM, technical vs sentiment)


                accuracy  coverage
technical_only  0.584178       1.0
sentiment_only  0.549696       1.0
both            0.531440       1.0


### Per-year breakdown (LightGBM, test set)


      accuracy  majority_baseline    n
2025  0.536643           0.626775  493


### Per-regime breakdown (LightGBM, test set)



Note: this entire window falls within a single regime (2025 = "flat" per src/models/regimes.py), so this breakdown is trivially one row - included for completeness, not because multiple regimes are represented.


      accuracy  majority_baseline    n
flat  0.536643           0.626775  493