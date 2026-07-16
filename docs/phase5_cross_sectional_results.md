# Phase 5 Cross-Sectional Feature Results


LightGBM only (most consistently well-behaved algorithm across prior phases). Same full-history walk-forward split throughout.



## Configuration: original_label_original_features


Label column: `label`, feature count: 26

Train rows: 19855, Validate rows: 6288

Validation majority baseline: 0.536

Validation accuracy @0.6: {'accuracy': 0.5529615218331172, 'coverage': 0.3678435114503817, 'n_predicted': 2313, 'local_majority_baseline': 0.5499351491569391}


Test rows: 12542 - evaluated ONCE at threshold=0.6

Test result: {'accuracy': 0.5283884996375936, 'coverage': 0.33001116249402007, 'n_predicted': 4139, 'local_majority_baseline': 0.5622130949504711}

Beats local_majority_baseline? No



## Configuration: original_label_plus_xs_features


Label column: `label`, feature count: 34

Train rows: 19833, Validate rows: 6275

Validation majority baseline: 0.536

Validation accuracy @0.6: {'accuracy': 0.5728534258456202, 'coverage': 0.36749003984063744, 'n_predicted': 2306, 'local_majority_baseline': 0.5520381613183001}


Test rows: 12527 - evaluated ONCE at threshold=0.6

Test result: {'accuracy': 0.5114100408359357, 'coverage': 0.33232218408238207, 'n_predicted': 4163, 'local_majority_baseline': 0.5491232284410281}

Beats local_majority_baseline? No



## Configuration: new_20d4pct_label_plus_xs_features


Label column: `label_20d_4pct`, feature count: 34

Train rows: 18083, Validate rows: 5308

Validation majority baseline: 0.560

Validation accuracy @0.6: {'accuracy': 0.5771365149833518, 'coverage': 0.5092313489073097, 'n_predicted': 2703, 'local_majority_baseline': 0.5667776544580096}


Test rows: 10512 - evaluated ONCE at threshold=0.6

Test result: {'accuracy': 0.5197672085089303, 'coverage': 0.4740296803652968, 'n_predicted': 4983, 'local_majority_baseline': 0.5918121613485852}

Beats local_majority_baseline? No
