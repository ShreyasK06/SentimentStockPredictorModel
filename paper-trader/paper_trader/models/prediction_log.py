import json

from paper_trader import config


def log_predictions(strategy_name: str, run_date: str, predictions: dict, sentiments: dict) -> int:
    """
    Append one JSONL record per universe ticker to logs/predictions/{strategy}.jsonl,
    for every ticker with a prediction this run - not just the ones that become
    trades. Non-trades are evaluation data too. Returns the number of records written.
    """
    path = config.predictions_log_path(strategy_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a") as f:
        for ticker, pred in predictions.items():
            sent = sentiments.get(ticker, {})
            record = {
                "date": run_date,
                "strategy": strategy_name,
                "ticker": ticker,
                "direction": pred["direction"],
                "confidence": pred["confidence"],
                "proba_up": pred["proba_up"],
                "sentiment_direction": sent.get("direction"),
                "sentiment_score": sent.get("score"),
                "horizon": pred["horizon"],
                "model_version": pred["model_version"],
                "data_window_hash": pred["data_window_hash"],
            }
            f.write(json.dumps(record) + "\n")
            n += 1
    return n


def load_predictions(strategy_name: str) -> list:
    path = config.predictions_log_path(strategy_name)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
