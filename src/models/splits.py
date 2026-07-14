import pandas as pd


def time_split(df: pd.DataFrame, train_end: str, validate_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into train, validate, and test sets with non-overlapping temporal boundaries.

    Boundaries: train has index <= train_end; validate has train_end < index <= validate_end; test has index > validate_end.
    """
    train_end_ts = pd.Timestamp(train_end)
    validate_end_ts = pd.Timestamp(validate_end)

    if train_end_ts >= validate_end_ts:
        raise ValueError(
            f"train_end must be strictly before validate_end, "
            f"but got train_end={train_end_ts} >= validate_end={validate_end_ts}"
        )

    train = df[df.index <= train_end_ts]
    validate = df[(df.index > train_end_ts) & (df.index <= validate_end_ts)]
    test = df[df.index > validate_end_ts]

    return train, validate, test
