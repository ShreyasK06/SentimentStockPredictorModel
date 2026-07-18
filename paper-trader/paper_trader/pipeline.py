import argparse
import time
import traceback
from datetime import datetime, timezone

from paper_trader import config
from paper_trader.data.degradation import evaluate_data_coverage
from paper_trader.data.run_log import write_run_log, last_run_date

from paper_trader.models.feature_store import build_universe_feature_state
from paper_trader.models.predictor import predict_from_state
from paper_trader.models.sentiment_model import sentiment_from_state
from paper_trader.models.prediction_log import log_predictions, load_predictions

from paper_trader.strategy.decision import should_trade, compute_position_size, select_new_positions, compute_exit_date

from paper_trader.execution.ledger import (
    already_traded_today, open_positions, positions_due_for_exit, append_trade,
    compute_virtual_equity, load_trades,
)
from paper_trader.execution.alpaca_client import AlpacaClient, reconcile_positions

from paper_trader.evaluation.outcomes import resolve_outcomes, _get_close_price
from paper_trader.evaluation.report import generate_report

from src.data.calendar import get_nyse_trading_days


def _is_trading_day(as_of_date: str) -> bool:
    return len(get_nyse_trading_days(as_of_date, as_of_date)) > 0


def _check_gap(run_date: str) -> dict:
    """Gap detection: if the last recorded run is >1 trading day before run_date, flag it."""
    prior = last_run_date()
    if prior is None or prior >= run_date:
        return {"gap_detected": False}
    trading_days = get_nyse_trading_days(prior, run_date)
    gap_trading_days = max(len(trading_days) - 2, 0)  # endpoints included; 0 = healthy daily cadence
    return {"gap_detected": gap_trading_days > 0, "prior_run_date": prior, "gap_trading_days": gap_trading_days}


def _earliest_prediction_date() -> str:
    dates = [p["date"] for name in config.strategy_names() for p in load_predictions(name)]
    return min(dates) if dates else datetime.now(timezone.utc).date().isoformat()


def run_strategy(strategy_cfg: dict, as_of_date: str, dry_run: bool, alpaca, shared_state: dict) -> dict:
    name = strategy_cfg["name"]
    model_cfg = dict(config.MODEL_CFG)
    model_cfg["horizon_days"] = strategy_cfg.get("horizon_days", model_cfg["horizon_days"])
    confidence_threshold = strategy_cfg.get("confidence_threshold", model_cfg["confidence_threshold"])

    predictions = predict_from_state(shared_state, model_cfg=model_cfg)
    sentiments = sentiment_from_state(shared_state)
    n_logged = log_predictions(name, as_of_date, predictions, sentiments)

    summary = {"strategy": name, "mode": strategy_cfg["mode"], "n_predictions": n_logged, "n_entries": 0, "n_exits": 0, "n_unfilled": 0}

    # --- exits first: mechanical, time-based, no discretionary logic ---
    for pos in positions_due_for_exit(name, as_of_date):
        exit_price = _get_close_price(pos["ticker"], as_of_date)
        if exit_price is None:
            continue

        if strategy_cfg["mode"] == "trade" and not dry_run and alpaca is not None:
            side = "sell" if pos["side"] == "buy" else "buy"
            order = alpaca.submit_market_order(pos["ticker"], int(pos["qty"]), side)
            fill = alpaca.poll_fill(order["order_id"], timeout_seconds=config.EXECUTION_CFG["order_fill_timeout_seconds"])
            fill_price = fill["fill_price"] or exit_price
        else:
            fill_price = exit_price

        exit_notional = round(fill_price * pos["qty"], 2)
        append_trade(name, {
            "event": "exit", "ticker": pos["ticker"], "entry_date": pos["entry_date"], "exit_date": as_of_date,
            "side": pos["side"], "qty": pos["qty"], "fill_price": fill_price, "notional": exit_notional,
            "entry_fill_price": pos["entry_price"],
            "return_pct": (fill_price - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] else None,
            "prediction_id": pos.get("prediction_id"), "dry_run": dry_run,
        })
        summary["n_exits"] += 1

    # --- entries: only trade-mode strategies touch capital ---
    if strategy_cfg["mode"] == "trade":
        current_open = open_positions(name)

        candidates = []
        for ticker, pred in predictions.items():
            if ticker in current_open or already_traded_today(name, ticker, as_of_date):
                continue
            decision = should_trade(pred, sentiments.get(ticker), confidence_threshold, allow_short=config.EXECUTION_CFG["allow_short"])
            if decision["should_trade"]:
                candidates.append({"ticker": ticker, "pred": pred, "decision": decision})
        candidates.sort(key=lambda c: c["pred"]["confidence"], reverse=True)
        selected = select_new_positions(candidates, config.EXECUTION_CFG["max_positions"], len(current_open))

        if not dry_run and alpaca is not None:
            account = alpaca.get_account()
            equity, cash = account["equity"], account["cash"]
        else:
            current_prices = {t: p for t, p in ((t, _get_close_price(t, as_of_date)) for t in current_open) if p is not None}
            virtual = compute_virtual_equity(name, config.EXECUTION_CFG["starting_virtual_equity"], current_prices)
            equity, cash = virtual["equity"], virtual["cash"]

        for cand in selected:
            ticker = cand["ticker"]
            price = _get_close_price(ticker, as_of_date)
            if price is None:
                continue
            sizing = compute_position_size(equity, config.EXECUTION_CFG["position_size_fraction"], cash, price)
            if sizing["shares"] <= 0:
                continue

            if not dry_run and alpaca is not None:
                order = alpaca.submit_market_order(ticker, sizing["shares"], cand["decision"]["side"])
                fill = alpaca.poll_fill(order["order_id"], timeout_seconds=config.EXECUTION_CFG["order_fill_timeout_seconds"])
                if fill["order_status"] != "filled":
                    append_trade(name, {
                        "event": "unfilled", "ticker": ticker, "entry_date": as_of_date,
                        "side": cand["decision"]["side"], "qty": sizing["shares"],
                        "order_status": fill["order_status"], "dry_run": dry_run,
                    })
                    summary["n_unfilled"] += 1
                    continue
                fill_price = fill["fill_price"]
            else:
                fill_price = price

            exit_date = compute_exit_date(as_of_date, model_cfg["horizon_days"])
            notional = round(fill_price * sizing["shares"], 2)
            cash -= notional
            append_trade(name, {
                "event": "entry", "ticker": ticker, "entry_date": as_of_date, "side": cand["decision"]["side"],
                "qty": sizing["shares"], "fill_price": fill_price, "notional": notional,
                "prediction_id": f"{as_of_date}:{ticker}:{name}", "scheduled_exit_date": exit_date, "dry_run": dry_run,
            })
            summary["n_entries"] += 1

    trades = load_trades(name)
    traded_keys = {(t["entry_date"], t["ticker"]) for t in trades if t.get("event") == "entry"}
    summary["n_outcomes_resolved"] = len(resolve_outcomes(name, as_of_date, traded_keys=traded_keys))

    return summary


def run_pipeline(as_of_date: str = None, dry_run: bool = False) -> dict:
    as_of_date = as_of_date or datetime.now(timezone.utc).date().isoformat()
    start_time = time.monotonic()
    timings = {}
    run_record = {"date": as_of_date, "dry_run": dry_run, "status": "error"}

    try:
        if not _is_trading_day(as_of_date):
            run_record["status"] = "skipped_not_trading_day"
            write_run_log(as_of_date, {**run_record, "timings": timings})
            return run_record

        run_record["gap_detection"] = _check_gap(as_of_date)

        tickers = config.get_universe()

        t0 = time.monotonic()
        state = build_universe_feature_state(tickers, as_of_date, feature_set=config.MODEL_CFG["feature_set"])
        state["as_of_date"] = as_of_date
        timings["feature_build_seconds"] = round(time.monotonic() - t0, 2)

        coverage = evaluate_data_coverage(tickers, state["price_errors"], state["news_coverage"], model_supports_price_only=True)
        run_record["data_coverage"] = coverage
        run_record["status"] = coverage["status"]

        if not coverage["proceed"]:
            run_record["timings"] = timings
            write_run_log(as_of_date, run_record)
            return run_record

        alpaca = None
        if not dry_run:
            try:
                alpaca = AlpacaClient()
            except ValueError:
                # No Alpaca credentials configured - degrade to virtual-ledger
                # sizing for every strategy instead of crashing the run.
                alpaca = None
            if alpaca is not None and config.MARKET_HOURS_CFG["guard"] and not alpaca.is_market_open_today(as_of_date):
                run_record["status"] = "skipped_market_closed"
                run_record["timings"] = timings
                write_run_log(as_of_date, run_record)
                return run_record
        run_record["alpaca_available"] = alpaca is not None

        if alpaca is not None:
            local_positions = {}
            for strategy_name in config.trade_mode_strategy_names():
                for ticker, pos in open_positions(strategy_name).items():
                    signed_qty = pos["qty"] if pos["side"] == "buy" else -pos["qty"]
                    local_positions[ticker] = {"qty": signed_qty}
            run_record["reconciliation"] = reconcile_positions(local_positions, alpaca.get_open_positions())

        t0 = time.monotonic()
        run_record["strategies"] = [run_strategy(cfg, as_of_date, dry_run, alpaca, state) for cfg in config.STRATEGIES_CFG]
        timings["strategies_seconds"] = round(time.monotonic() - t0, 2)

        t0 = time.monotonic()
        generate_report(_earliest_prediction_date(), as_of_date)
        timings["report_seconds"] = round(time.monotonic() - t0, 2)

        timings["total_seconds"] = round(time.monotonic() - start_time, 2)
        run_record["timings"] = timings
        write_run_log(as_of_date, run_record)
        return run_record

    except Exception as exc:
        timings["total_seconds"] = round(time.monotonic() - start_time, 2)
        run_record["status"] = "error"
        run_record["error"] = f"{type(exc).__name__}: {exc}"
        run_record["traceback"] = traceback.format_exc()
        run_record["timings"] = timings
        write_run_log(as_of_date, run_record)
        raise


def main():
    parser = argparse.ArgumentParser(description="Paper trading daily pipeline")
    parser.add_argument("--date", default=None, help="as_of_date (YYYY-MM-DD); defaults to today (UTC)")
    parser.add_argument("--dry-run", action="store_true", help="Run the full pipeline without submitting any orders")
    args = parser.parse_args()

    result = run_pipeline(as_of_date=args.date, dry_run=args.dry_run)
    print(f"Run status: {result.get('status')}")


if __name__ == "__main__":
    main()
