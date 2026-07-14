# Phase 3 Baseline Results

Train: 2019-12-31 00:00:00 to 2022-12-30 00:00:00 (19855 rows)

Validate: 2023-01-03 00:00:00 to 2023-12-29 00:00:00 (6288 rows)


## Validation-year (2023) results

Logistic regression majority baseline: 0.536

Logistic regression accuracy @0.5: {'accuracy': 0.5295801526717557, 'coverage': 1.0, 'n_predicted': 6288}

LightGBM accuracy @0.5: {'accuracy': 0.5357824427480916, 'coverage': 1.0, 'n_predicted': 6288}

Random Forest accuracy @0.5: {'accuracy': 0.5508905852417303, 'coverage': 1.0, 'n_predicted': 6288}

XGBoost accuracy @0.5: {'accuracy': 0.5340330788804071, 'coverage': 1.0, 'n_predicted': 6288}


## Confidence-threshold sweep (tuned on validation only)

Logistic regression:

      accuracy  coverage
0.50  0.529580  1.000000
0.55  0.550937  0.195134
0.60  0.672131  0.009701
0.65  1.000000  0.000318
0.70  1.000000  0.000159

LightGBM:

      accuracy  coverage
0.50  0.535782  1.000000
0.55  0.557122  0.654262
0.60  0.557596  0.381043
0.65  0.591571  0.207538
0.70  0.596244  0.101622

Random Forest:

      accuracy  coverage
0.50  0.550891  1.000000
0.55  0.600887  0.215172
0.60  0.640777  0.016380
0.65  0.550000  0.003181
0.70  0.800000  0.000795

XGBoost:

      accuracy  coverage
0.50  0.534033  1.000000
0.55  0.545270  0.785146
0.60  0.553586  0.583174
0.65  0.563813  0.408715
0.70  0.561652  0.281170

## Ablation (LightGBM, evaluated on validation year)

                accuracy  coverage
technical_only  0.535782       1.0
sentiment_only  0.535623       1.0
both            0.535782       1.0

## TEST SET RESULTS (2024-2025) - evaluated ONCE at threshold=0.6

Test (untouched until this run): 2024-01-02 00:00:00 to 2025-12-16 00:00:00 (12542 rows)

Test-set majority baseline: 0.561

Logistic regression: {'accuracy': 0.5434782608695652, 'coverage': 0.014670706426407271, 'n_predicted': 184}

LightGBM: {'accuracy': 0.5127367733507512, 'coverage': 0.36620953595917716, 'n_predicted': 4593}

Random Forest: {'accuracy': 0.4853801169590643, 'coverage': 0.04090256737362462, 'n_predicted': 513}

XGBoost: {'accuracy': 0.5083512931034483, 'coverage': 0.591931111465476, 'n_predicted': 7424}


### Per-year breakdown (LightGBM, test set)

      accuracy  majority_baseline     n
2024  0.533333           0.549779  6328
2025  0.490489           0.571612  6214

### Per-regime breakdown (LightGBM, test set)

      accuracy  majority_baseline     n
bull  0.533333           0.549779  6328
flat  0.490489           0.571612  6214