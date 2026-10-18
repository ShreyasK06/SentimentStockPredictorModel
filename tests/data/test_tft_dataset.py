import numpy as np
import pandas as pd
import pytest

from src.data.tft_dataset import (
    CALENDAR_FEATURES,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    OBSERVED_FEATURES,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    TARGET_COLUMNS,
    TECHNICAL_FEATURES,
    add_time_idx,
    assert_no_target_leakage,
    build_calendar_features,
    build_forward_targets,
    build_sentiment_features,
    build_ticker_panel,
    finalize_panel,
    sentiment_coverage_report,
)
from src.features.trading_day import assign_effective_trading_day

N_DAYS = 400
DAYS = pd.bdate_range("2020-01-01", periods=N_DAYS)


def _prices(seed: int, days: pd.DatetimeIndex = DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000, 5_000, len(days)).astype(float),
        },
        index=pd.DatetimeIndex(days, name="date"),
    )


def _news(rows: list[tuple[str, float]]) -> pd.DataFrame:
    # (UTC timestamp string, sentiment_score)
    return pd.DataFrame(
        {
            "headline": ["h"] * len(rows),
            "positive": [max(s, 0.0) for _, s in rows],
            "negative": [max(-s, 0.0) for _, s in rows],
            "neutral": [0.5] * len(rows),
            "sentiment_score": [s for _, s in rows],
            "published_at": pd.to_datetime([t for t, _ in rows]),
        }
    )


def test_effective_trading_day_boundaries():
    days = pd.bdate_range("2025-03-03", "2025-03-14")
    published = pd.Series(
        pd.to_datetime(
            [
                "2025-03-05 20:59",  # 15:59 ET (EST) Wed -> same day
                "2025-03-05 21:01",  # 16:01 ET Wed -> Thu
                "2025-03-08 15:00",  # Saturday -> Monday 3/10
            ]
        )
    )
    eff = assign_effective_trading_day(published, days)
    assert eff.iloc[0] == pd.Timestamp("2025-03-05")
    assert eff.iloc[1] == pd.Timestamp("2025-03-06")
    assert eff.iloc[2] == pd.Timestamp("2025-03-10")


def test_sentiment_features_assignment_and_rolling_are_causal():
    days = pd.bdate_range("2025-03-03", "2025-03-14")
    news = _news([("2025-03-05 20:59", 0.6), ("2025-03-05 21:01", -0.4)])
    s = build_sentiment_features(news, days)
    assert s.loc["2025-03-05", "article_count"] == 1
    assert s.loc["2025-03-05", "sent_mean"] == pytest.approx(0.6)
    assert s.loc["2025-03-06", "sent_mean"] == pytest.approx(-0.4)
    assert s.loc["2025-03-04", "has_news"] == 0
    assert s.loc["2025-03-04", "sent_roll3"] == 0
    assert s.loc["2025-03-06", "sent_roll3"] == pytest.approx((0 + 0.6 - 0.4) / 3)

    later = _news([("2025-03-05 20:59", 0.6), ("2025-03-05 21:01", -0.4), ("2025-03-12 15:00", 0.9)])
    s2 = build_sentiment_features(later, days)
    pd.testing.assert_frame_equal(s.loc[:"2025-03-11"], s2.loc[:"2025-03-11"])


def test_sentiment_features_without_news_are_neutral():
    s = build_sentiment_features(None, DAYS)
    assert list(s.columns) == SENTIMENT_FEATURES
    assert (s == 0).all().all()


def test_forward_targets():
    idx = pd.bdate_range("2024-01-01", periods=20)
    close = pd.Series(np.linspace(100, 120, 20), index=idx)
    close.iloc[10] = 90.0
    t = build_forward_targets(close)
    for h in (1, 3, 5, 7):
        expected = np.log(close.shift(-h) / close)
        pd.testing.assert_series_equal(t[f"fwd_ret_{h}d"], expected, check_names=False)
        assert t[f"fwd_ret_{h}d"].iloc[-h:].isna().all()
        assert t[f"fwd_ret_{h}d"].iloc[:-h].notna().all()
        direction = t[f"fwd_dir_{h}d"]
        assert direction.iloc[-h:].isna().all()
        valid = t[f"fwd_ret_{h}d"].notna()
        assert (direction[valid] == (t[f"fwd_ret_{h}d"][valid] > 0).astype(float)).all()
    assert t["fwd_dir_1d"].iloc[9] == 0.0
    assert t["fwd_dir_1d"].iloc[10] == 1.0


def test_calendar_features_depend_only_on_date():
    full = build_calendar_features(DAYS)
    subset_days = DAYS[50:200]
    sub = build_calendar_features(subset_days)
    index_dependent = {"is_month_end", "is_quarter_end", "is_pre_holiday", "days_to_next_trading_day"}
    cols = [c for c in CALENDAR_FEATURES if c not in index_dependent]
    pd.testing.assert_frame_equal(full.loc[subset_days, cols], sub[cols])
    assert list(full.columns) == CALENDAR_FEATURES
    assert full.loc["2020-01-31", "is_month_end"] == 1
    assert full.loc["2020-03-31", "is_quarter_end"] == 1
    assert full.loc["2020-02-28", "is_quarter_end"] == 0
    assert full.loc["2020-01-03", "days_to_next_trading_day"] == 3  # Friday


def test_pre_holiday_flag():
    days = pd.DatetimeIndex(["2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"])
    c = build_calendar_features(days)
    assert c.loc["2024-07-03", "is_pre_holiday"] == 1  # Jul 4 closed
    assert c.loc["2024-07-02", "is_pre_holiday"] == 0


def _build_two(perturb_after: pd.Timestamp | None = None) -> pd.DataFrame:
    pa, pb = _prices(1), _prices(2)
    if perturb_after is not None:
        pa = pa.copy()
        mask = pa.index > perturb_after
        pa.loc[mask, ["Open", "High", "Low", "Close"]] *= 1.5
        pa.loc[mask, "Volume"] *= 3
    news = _news([("2021-03-01 15:00", 0.5), ("2021-03-02 22:00", -0.2)])
    frames = [
        build_ticker_panel(pa, news, DAYS, ticker="AAA"),
        build_ticker_panel(pb, None, DAYS, ticker="BBB"),
    ]
    return finalize_panel(pd.concat(frames, ignore_index=True))


def test_no_leakage_from_future_prices():
    base = _build_two()
    cutoff = base["date"].iloc[len(base) // 4]
    perturbed = _build_two(perturb_after=cutoff)
    feats = ["date", "ticker", "time_idx", "log_close"] + OBSERVED_FEATURES + CALENDAR_FEATURES
    a = base[(base["ticker"] == "AAA") & (base["date"] <= cutoff)][feats].reset_index(drop=True)
    b = perturbed[(perturbed["ticker"] == "AAA") & (perturbed["date"] <= cutoff)][feats].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # Targets do look ahead, so they must differ at the cutoff.
    ta = base[(base["ticker"] == "AAA") & (base["date"] == cutoff)]["fwd_ret_1d"].iloc[0]
    tb = perturbed[(perturbed["ticker"] == "AAA") & (perturbed["date"] == cutoff)]["fwd_ret_1d"].iloc[0]
    assert ta != pytest.approx(tb)


def test_time_idx_contiguous_and_shared():
    panel = _build_two()
    assert panel["time_idx"].min() == 0
    for _, g in panel.groupby("ticker"):
        assert (g["time_idx"].diff().dropna() == 1).all()
    assert (panel.groupby("date")["time_idx"].nunique() == 1).all()
    assert panel.groupby("ticker").size().nunique() == 1
    assert not panel[PRICE_FEATURES + TECHNICAL_FEATURES].isna().any().any()
    assert panel["date"].min() == DAYS[251]  # 252-day window warm-up


def test_missing_ticker_days_are_filled_consistently():
    holes = DAYS.delete([100, 101, 300])
    frames = [
        build_ticker_panel(_prices(1), None, DAYS, ticker="AAA"),
        build_ticker_panel(_prices(2, holes), None, DAYS, ticker="BBB"),
    ]
    panel = finalize_panel(pd.concat(frames, ignore_index=True))
    assert panel.groupby("ticker").size().nunique() == 1


def test_add_time_idx_global_mapping():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"]),
            "ticker": ["A", "A", "B", "B"],
        }
    )
    assert add_time_idx(df)["time_idx"].tolist() == [0, 1, 1, 2]


def test_feature_lists_do_not_intersect_targets():
    all_features = OBSERVED_FEATURES + KNOWN_REALS + KNOWN_CATEGORICALS
    assert not set(all_features) & set(TARGET_COLUMNS)
    assert_no_target_leakage(all_features)
    with pytest.raises(ValueError):
        assert_no_target_leakage(all_features + ["fwd_ret_3d"])


def test_sentiment_coverage_report():
    cov = sentiment_coverage_report(_build_two())
    assert cov.loc["AAA", "n_days_with_news"] == 2
    assert cov.loc["BBB", "n_days_with_news"] == 0
    assert pd.isna(cov.loc["BBB", "first_news_date"])
    assert cov.loc["AAA", "pct_days_with_news"] == pytest.approx(100 * 2 / cov.loc["AAA", "n_days"])
