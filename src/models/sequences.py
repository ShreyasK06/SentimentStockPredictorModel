import numpy as np
import pandas as pd


def build_sequences(df: pd.DataFrame, feature_columns: list, lookback: int = 10):
    X_seq_list = []
    y_list = []
    index_list = []
    ticker_list = []

    for ticker_name, group in df.groupby("ticker"):
        group = group.sort_index()
        n = len(group)
        if n < lookback:
            continue
        features = group[feature_columns].values
        labels = group["label"].values
        dates = group.index

        for end in range(lookback - 1, n):
            start = end - lookback + 1
            X_seq_list.append(features[start:end + 1])
            y_list.append(labels[end])
            index_list.append(dates[end])
            ticker_list.append(ticker_name)

    if not X_seq_list:
        return (
            np.empty((0, lookback, len(feature_columns))),
            np.empty((0,)),
            pd.DatetimeIndex([]),
            pd.Series([], dtype=object),
        )

    X_seq = np.stack(X_seq_list)
    y = np.array(y_list)
    index = pd.DatetimeIndex(index_list)
    ticker = pd.Series(ticker_list)

    return X_seq, y, index, ticker
