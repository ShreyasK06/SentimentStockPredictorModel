import numpy as np
import pandas as pd


def generate_labels(close: pd.Series, horizon: int = 10, up_threshold: float = 0.02, down_threshold: float = 0.02) -> pd.Series:
    values = close.values
    n = len(values)
    labels = np.full(n, np.nan)

    for t in range(n):
        entry = values[t]
        future = values[t + 1: t + 1 + horizon]
        if len(future) < horizon:
            continue  # insufficient future data - drop (tail row)

        up_target = entry * (1 + up_threshold)
        down_target = entry * (1 - down_threshold)
        up_hits = future >= up_target
        down_hits = future <= down_target
        hit_up = int(np.argmax(up_hits)) if up_hits.any() else None
        hit_down = int(np.argmax(down_hits)) if down_hits.any() else None

        if hit_up is None and hit_down is None:
            continue  # neither threshold hit within horizon - ambiguous, drop
        elif hit_down is None:
            labels[t] = 1.0
        elif hit_up is None:
            labels[t] = 0.0
        else:
            labels[t] = 1.0 if hit_up <= hit_down else 0.0

    return pd.Series(labels, index=close.index, name="label")
