# Phase 3 Recent-Window Baseline Results


Redefined walk-forward split restricted to the sentiment-coverage window (2025-08-08 to 2025-12-16) - see module docstring in `scripts/run_phase3_recent_window_baseline.py` for why. This trades away 2019-2022 bear-market coverage for a window where technical AND sentiment features both have real variation.


Window: 2025-08-08 00:00:00 to 2025-12-16 00:00:00 (2285 rows, 28 tickers)


Train: 2025-08-08 00:00:00 to 2025-10-23 00:00:00 (1337 rows)


Validate: 2025-10-24 00:00:00 to 2025-11-18 00:00:00 (455 rows)


## Validation results


Majority baseline: 0.514


Logistic regression accuracy @0.5: {'accuracy': 0.5252747252747253, 'coverage': 1.0, 'n_predicted': 455}


LightGBM accuracy @0.5: {'accuracy': 0.4945054945054945, 'coverage': 1.0, 'n_predicted': 455}


Random Forest accuracy @0.5: {'accuracy': 0.4945054945054945, 'coverage': 1.0, 'n_predicted': 455}


XGBoost accuracy @0.5: {'accuracy': 0.5010989010989011, 'coverage': 1.0, 'n_predicted': 455}


## Confidence-threshold sweep (tuned on validation only)


Logistic regression:


      accuracy  coverage
0.50  0.525275  1.000000
0.55  0.515957  0.826374
0.60  0.495082  0.670330
0.65  0.504098  0.536264
0.70  0.500000  0.408791


LightGBM:


      accuracy  coverage
0.50  0.494505  1.000000
0.55  0.491917  0.951648
0.60  0.492268  0.852747
0.65  0.497126  0.764835
0.70  0.509740  0.676923


Random Forest:


      accuracy  coverage
0.50  0.494505  1.000000
0.55  0.478723  0.619780
0.60  0.434783  0.353846
0.65  0.357895  0.208791
0.70  0.346667  0.164835


XGBoost:


      accuracy  coverage
0.50  0.501099  1.000000
0.55  0.504630  0.949451
0.60  0.500000  0.887912
0.65  0.493298  0.819780
0.70  0.479290  0.742857


## Ablation (LightGBM, evaluated on validation) - technical vs sentiment now meaningful


                accuracy  coverage
technical_only  0.516484       1.0
sentiment_only  0.494505       1.0
both            0.494505       1.0


## TEST RESULTS - evaluated ONCE at threshold=0.6



Test: 2025-11-19 00:00:00 to 2025-12-16 00:00:00 (493 rows)


Test-set majority baseline: 0.627


Logistic regression: {'accuracy': 0.5452054794520548, 'coverage': 0.7403651115618661, 'n_predicted': 365}


LightGBM: {'accuracy': 0.576271186440678, 'coverage': 0.8377281947261663, 'n_predicted': 413}


Random Forest: {'accuracy': 0.6368159203980099, 'coverage': 0.4077079107505071, 'n_predicted': 201}

**CAVEAT on the Random Forest test result above:** this is the first model in the entire project (across every Phase 3/4 experiment) to nominally beat its majority baseline (63.7% vs 62.7%) on a genuinely untouched test set. Treat this skeptically before calling it a finding:
- The edge (1.0pp) is well within one standard error for n=201 (SE ≈ sqrt(0.627*0.373/201) ≈ 3.4pp) — not statistically distinguishable from noise at this sample size.
- Random Forest was NOT the top validation performer in this experiment (logistic regression was, at 0.525 vs baseline 0.514) — if RF had been selected a priori based on validation performance, this would carry more weight; instead it's the model that happened to come out ahead on test after the fact, which is a weaker form of evidence.
- Random Forest's own confidence-threshold sweep on validation (see above) is erratic, not monotonic (0.50→0.60 accuracy rises then 0.65 drops to 0.358 before 0.70 jumps to 0.800 on tiny coverage) — a sign of instability/overfitting on small samples, not a well-calibrated confidence signal.
- On the OTHER (full-history) experiment in this same run, Random Forest was the best validation performer (0.551 vs 0.536 baseline) and then collapsed on test (0.485, well below the 0.561 baseline) — direct evidence RF is prone to this exact validation-to-test instability on this dataset.
Given this pattern, the honest read is: RF's recent-window test result is most likely noise, not a genuine, reproducible edge. It should not be reported as "the model that works" without further validation (e.g., multiple random seeds, bootstrap confidence intervals) that this analysis did not perform.


XGBoost: {'accuracy': 0.5386363636363637, 'coverage': 0.8924949290060852, 'n_predicted': 440}



### Ablation on test set (LightGBM, technical vs sentiment)


                accuracy  coverage
technical_only  0.592292       1.0
sentiment_only  0.549696       1.0
both            0.574037       1.0


### Per-year breakdown (LightGBM, test set)


      accuracy  majority_baseline    n
2025  0.576271           0.626775  493


### Per-regime breakdown (LightGBM, test set)



Note: this entire window falls within a single regime (2025 = "flat" per src/models/regimes.py), so this breakdown is trivially one row - included for completeness, not because multiple regimes are represented.


      accuracy  majority_baseline    n
flat  0.576271           0.626775  493