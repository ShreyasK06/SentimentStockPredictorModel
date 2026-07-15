# Phase 4 Small LSTM Results


Lookback window: 10 days. Single-layer LSTM, hidden_size=16, per the spec's 'keep it small' guidance.



## Experiment: full_history


Train sequences: 19603

Validate sequences: 6036


Validation majority baseline: 0.533

LSTM validation accuracy @0.5: {'accuracy': 0.5210404241219351, 'coverage': 1.0, 'n_predicted': 6036, 'local_majority_baseline': 0.5326375082836315}

LSTM validation accuracy @0.6 (matches the threshold used for test below): {'accuracy': 0.5341463414634147, 'coverage': 0.06792577866136515, 'n_predicted': 410, 'local_majority_baseline': 0.5658536585365853}


Test (2024-01-16 00:00:00 to 2025-12-16 00:00:00, 12290 sequences) - evaluated ONCE at threshold=0.6

LSTM test result: {'accuracy': 0.5641025641025641, 'coverage': 0.06346623270951994, 'n_predicted': 780, 'local_majority_baseline': 0.5653846153846154}

Beats local_majority_baseline? No


### Per-year breakdown

      accuracy  majority_baseline     n
2024  0.549669           0.552666  6076
2025  0.584098           0.571612  6214


### Per-regime breakdown

      accuracy  majority_baseline     n
bull  0.549669           0.552666  6076
flat  0.584098           0.571612  6214


## Experiment: recent_window


Train sequences: 1085

Validate sequences: 203


Validation majority baseline: 0.557

LSTM validation accuracy @0.5: {'accuracy': 0.3842364532019704, 'coverage': 1.0, 'n_predicted': 203, 'local_majority_baseline': 0.5566502463054187}

LSTM validation accuracy @0.6 (matches the threshold used for test below): {'accuracy': 0.30303030303030304, 'coverage': 0.3251231527093596, 'n_predicted': 66, 'local_majority_baseline': 0.6212121212121212}


Test (2025-12-03 00:00:00 to 2025-12-16 00:00:00, 241 sequences) - evaluated ONCE at threshold=0.6

LSTM test result: {'accuracy': 0.36363636363636365, 'coverage': 0.13692946058091288, 'n_predicted': 33, 'local_majority_baseline': 0.6060606060606061}

Beats local_majority_baseline? No


### Per-year breakdown

      accuracy  majority_baseline    n
2025  0.363636           0.547718  241


### Per-regime breakdown

      accuracy  majority_baseline    n
flat  0.363636           0.547718  241