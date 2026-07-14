import pandas as pd


def time_split(df: pd.DataFrame, train_end: str, validate_end: str):
    train_end_ts = pd.Timestamp(train_end)
    validate_end_ts = pd.Timestamp(validate_end)

    train = df[df.index <= train_end_ts]
    validate = df[(df.index > train_end_ts) & (df.index <= validate_end_ts)]
    test = df[df.index > validate_end_ts]

    return train, validate, test
