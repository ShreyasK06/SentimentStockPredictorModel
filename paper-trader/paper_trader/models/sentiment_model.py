from paper_trader import config
from paper_trader.models.feature_store import build_universe_feature_state


def sentiment_from_state(state: dict, sentiment_column: str = None) -> dict:
    """
    Standalone sentiment up/down signal used as the agreement filter, derived
    purely from the aggregated FinBERT rolling sentiment score (independent of
    the price-based direction model) - not a second trained classifier.
    Returns {ticker: {direction, score, as_of_date}}.
    """
    sentiment_column = sentiment_column or config.MODEL_CFG["sentiment_column"]
    predict_rows = state["predict_rows"]
    as_of_date = state.get("as_of_date")

    results = {}
    for _, row in predict_rows.iterrows():
        score = float(row[sentiment_column])
        direction = "up" if score > 0 else ("down" if score < 0 else "flat")
        results[row["ticker"]] = {"direction": direction, "score": score, "as_of_date": as_of_date}
    return results


def sentiment_universe(tickers: list, as_of_date: str, sentiment_column: str = None) -> dict:
    """Builds its own feature state - convenience entrypoint for tests/standalone use."""
    state = build_universe_feature_state(tickers, as_of_date, feature_set="base")
    state["as_of_date"] = as_of_date
    return sentiment_from_state(state, sentiment_column=sentiment_column)


def sentiment(ticker: str, as_of_date: str) -> dict:
    """sentiment(ticker, as_of_date) -> {direction, score} - the Phase 2 stable interface."""
    return sentiment_universe([ticker], as_of_date).get(ticker)
