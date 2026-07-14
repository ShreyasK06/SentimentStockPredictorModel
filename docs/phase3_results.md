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