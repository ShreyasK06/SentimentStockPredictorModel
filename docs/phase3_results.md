# Phase 3 Baseline Results

Train: 2019-12-31 00:00:00 to 2022-12-30 00:00:00 (19855 rows)

Validate: 2023-01-03 00:00:00 to 2023-12-29 00:00:00 (6288 rows)


## Validation-year (2023) results

Logistic regression majority baseline: 0.536

Logistic regression accuracy @0.5: {'accuracy': 0.5268765903307888, 'coverage': 1.0, 'n_predicted': 6288, 'local_majority_baseline': 0.5356234096692112}

LightGBM accuracy @0.5: {'accuracy': 0.5318066157760815, 'coverage': 1.0, 'n_predicted': 6288, 'local_majority_baseline': 0.5356234096692112}

Random Forest accuracy @0.5: {'accuracy': 0.5464376590330788, 'coverage': 1.0, 'n_predicted': 6288, 'local_majority_baseline': 0.5356234096692112}

XGBoost accuracy @0.5: {'accuracy': 0.5271946564885496, 'coverage': 1.0, 'n_predicted': 6288, 'local_majority_baseline': 0.5356234096692112}


## Confidence-threshold sweep (tuned on validation only)

Logistic regression:

      accuracy  coverage
0.50  0.526877  1.000000
0.55  0.540478  0.206266
0.60  0.657534  0.011609
0.65  1.000000  0.000477
0.70  1.000000  0.000159

LightGBM:

      accuracy  coverage
0.50  0.531807  1.000000
0.55  0.544658  0.653467
0.60  0.552962  0.367844
0.65  0.575366  0.184637
0.70  0.582878  0.087309

Random Forest:

      accuracy  coverage
0.50  0.546438  1.000000
0.55  0.594893  0.230439
0.60  0.558140  0.013677
0.65  0.416667  0.001908
0.70  0.666667  0.000477

XGBoost:

      accuracy  coverage
0.50  0.527195  1.000000
0.55  0.531799  0.787691
0.60  0.534258  0.584924
0.65  0.539604  0.417621
0.70  0.553035  0.280375

## Ablation (LightGBM, evaluated on validation year)

                accuracy  coverage
technical_only  0.531807       1.0
sentiment_only  0.535623       1.0
both            0.531807       1.0

## TEST SET RESULTS (2024-2025) - evaluated ONCE at threshold=0.6

Test (untouched until this run): 2024-01-02 00:00:00 to 2025-12-16 00:00:00 (12542 rows)

Test-set majority baseline: 0.561

Logistic regression: {'accuracy': 0.49375, 'coverage': 0.012757136022962846, 'n_predicted': 160, 'local_majority_baseline': 0.5125}

LightGBM: {'accuracy': 0.5283884996375936, 'coverage': 0.33001116249402007, 'n_predicted': 4139, 'local_majority_baseline': 0.5622130949504711}

Random Forest: {'accuracy': 0.5935483870967742, 'coverage': 0.012358475522245256, 'n_predicted': 155, 'local_majority_baseline': 0.6129032258064516}

XGBoost: {'accuracy': 0.4921616568387746, 'coverage': 0.5543772922978791, 'n_predicted': 6953, 'local_majority_baseline': 0.5558751618006615}


### Per-year breakdown (LightGBM, test set)

      accuracy  majority_baseline     n
2024  0.527058           0.549779  6328
2025  0.529897           0.571612  6214

### Per-regime breakdown (LightGBM, test set)

      accuracy  majority_baseline     n
bull  0.527058           0.549779  6328
flat  0.529897           0.571612  6214

## FINAL CORRECTED INTERPRETATION

The `local_majority_baseline` field above (added after discovering a methodology bug - see docs/phase3_recent_window_results.md for the full investigation) is the correct, apples-to-apples comparison: the majority-class rate WITHIN the same confident subset each model actually predicted on, not the full test set's rate. Comparing each model's `accuracy` against its own `local_majority_baseline`:

| Model | Accuracy | Local baseline | Beats local baseline? |
|---|---|---|---|
| Logistic regression | 0.494 | 0.513 | No |
| LightGBM | 0.528 | 0.562 | No |
| Random Forest | 0.594 | 0.613 | No |
| XGBoost | 0.492 | 0.556 | No |

**All four algorithms underperform their own correctly-matched baseline on the full 2019-2025 walk-forward test set.** The earlier apparent "Random Forest beats baseline (0.594 vs 0.561)" result was comparing against the wrong (full-set) baseline; against the correct one, RF is actually 1.9 points worse. No algorithm tried in this project has a genuine, validated edge on this experiment.