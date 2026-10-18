import io

import numpy as np
import pandas as pd

from src.data.calendar import get_nyse_trading_days
from src.data.fnspid import (
    build_labels,
    build_symbol_map,
    conservative_effective_day,
    dedupe_articles,
    filter_chunk,
    make_article_id,
    naive_effective_day,
    normalize_headline,
    stamped_date,
)

CSV = (
    "Date,Article_title,Stock_symbol,Url,Publisher,Author,Article\n"
    '2020-05-22 08:45:06 UTC,"Mid-Day, ""quoted"" headline",GOOG,http://a/1,Lisa,,"multi\nline body"\n'
    "2020-05-22 00:00:00 UTC,Other co headline,ZZZZ,http://a/2,Lisa,,\n"
    "2020-05-23 10:00:00 UTC,Facebook thing,FB,http://a/3,Bz,,\n"
    "2020-05-23 10:00:00 UTC,Apple thing,AAPL,http://a/4,Bz,,\n"
)


def _chunk():
    return pd.read_csv(io.StringIO(CSV), usecols=["Date", "Article_title", "Stock_symbol", "Url", "Publisher"], dtype=str)


def test_filter_chunk_keeps_symbols_and_aliases():
    out = filter_chunk(_chunk(), build_symbol_map(["AAPL", "GOOGL", "META"]))
    assert list(out["ticker"]) == ["GOOGL", "META", "AAPL"]
    assert list(out["fnspid_symbol"]) == ["GOOG", "FB", "AAPL"]
    assert out.loc[0, "headline"] == 'Mid-Day, "quoted" headline'
    assert out.loc[0, "raw_date"] == "2020-05-22 08:45:06 UTC"
    assert set(out.columns) == {"raw_date", "headline", "url", "publisher", "fnspid_symbol", "ticker"}


def test_conservative_effective_day_is_strictly_after_stamped_date():
    days = get_nyse_trading_days("2020-05-01", "2020-06-30")
    stamped = pd.Series(pd.to_datetime(["2020-05-21", "2020-05-22", "2020-05-23", "2020-05-25", "2020-05-26"]))
    eff = conservative_effective_day(stamped, days)
    # Thu->Fri, Fri->Tue (Mon 25th is Memorial Day), Sat->Tue, Mon holiday->Tue, Tue->Wed
    assert list(eff) == list(pd.to_datetime(["2020-05-22", "2020-05-26", "2020-05-26", "2020-05-26", "2020-05-27"]))


def test_stamped_date_ignores_clock_and_tz():
    s = pd.Series(["2020-05-22 23:59:59 UTC", "2020-05-22 00:00:00 UTC"])
    assert list(stamped_date(s)) == [pd.Timestamp("2020-05-22")] * 2


def test_naive_effective_day_uses_clock():
    days = get_nyse_trading_days("2020-05-01", "2020-06-30")
    raw = pd.Series(["2020-05-22 08:45:06 UTC", "2020-05-22 21:00:00 UTC"])
    got = naive_effective_day(raw, days)
    # 08:45 UTC = 04:45 ET Fri -> Fri; 21:00 UTC = 17:00 ET -> next session Tue
    assert list(got) == [pd.Timestamp("2020-05-22"), pd.Timestamp("2020-05-26")]


def test_normalize_headline():
    assert normalize_headline("  Stocks, UP!!  today ") == "stocks up today"
    assert normalize_headline(None) == ""


def test_article_id_stable_and_ticker_specific():
    a = make_article_id("AAPL", "http://x", "h", "2020-01-01")
    assert a == make_article_id("AAPL", "http://x", "h", "2020-01-01")
    assert a != make_article_id("MSFT", "http://x", "h", "2020-01-01")
    assert make_article_id("AAPL", None, "h", "d") != make_article_id("AAPL", None, "h2", "d")


def test_dedupe_url_and_headline_window():
    df = pd.DataFrame({
        "ticker": ["A"] * 6 + ["B"],
        "url": ["u1", "u1", "u2", "u3", "u4", None, "u5"],
        "headline": ["Hello World", "Other", "hello, world!", "Hello World", "Hello World", "x", "Hello World"],
        "stamped_date": pd.to_datetime(
            ["2020-01-01", "2020-01-01", "2020-01-03", "2020-01-04", "2020-01-10", "2020-01-10", "2020-01-02"]),
    })
    out, counts = dedupe_articles(df)
    assert counts["url_duplicates_removed"] == 1  # second u1
    # u2 (Jan 3, within 3d of Jan 1) and u3 (Jan 4, 3d after Jan 1) dropped; Jan 10 kept
    assert counts["headline_duplicates_removed"] == 2
    assert counts["output"] == 4
    assert len(out[out.ticker == "B"]) == 1  # per-ticker


def test_label_for_E_uses_only_close_E_and_next_close():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    prices = pd.DataFrame({"ticker": "A", "date": dates, "Close": [10.0, 11.0, 11.0, 9.0]})
    lab = build_labels(prices).set_index("date")
    assert lab.loc["2020-01-02", "y"] == 1.0 and np.isclose(lab.loc["2020-01-02", "ret_next"], 0.1)
    assert np.isnan(lab.loc["2020-01-03", "y"])  # zero return dropped
    assert lab.loc["2020-01-06", "y"] == 0.0 and lab.loc["2020-01-06", "next_close"] == 9.0
    assert np.isnan(lab.loc["2020-01-07", "next_close"]) and np.isnan(lab.loc["2020-01-07", "y"])
    # changing the last close must not change labels that do not use it
    p2 = prices.copy()
    p2.loc[3, "Close"] = 100.0
    lab2 = build_labels(p2).set_index("date")
    assert lab2.loc["2020-01-02", "y"] == lab.loc["2020-01-02", "y"]
    assert lab2.loc["2020-01-03", "ret_next"] == lab.loc["2020-01-03", "ret_next"]
    assert np.isclose(lab.loc["2020-01-06", "ret_1d"], 0.0)
