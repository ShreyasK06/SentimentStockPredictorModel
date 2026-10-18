import pandas as pd
import pytest

from src.data import finnhub_chunked as fc
from src.data.calendar import get_nyse_trading_days


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class FakeSession:
    """Responds via fn(params) -> payload | (payload, status)."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        r = self.fn(params)
        return FakeResp(*r) if isinstance(r, tuple) else FakeResp(r)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)


def art(i, ts=1704153600, url=None):
    return {"id": i, "headline": f"h{i}", "source": "Reuters", "url": url or f"http://x/{i}",
            "datetime": ts, "summary": "s"}


def test_date_chunks_boundaries():
    assert fc.date_chunks("2024-01-01", "2024-01-15", 7) == [
        ("2024-01-01", "2024-01-07"), ("2024-01-08", "2024-01-14"), ("2024-01-15", "2024-01-15")]
    assert fc.date_chunks("2024-02-28", "2024-03-01", 1) == [
        ("2024-02-28", "2024-02-28"), ("2024-02-29", "2024-02-29"), ("2024-03-01", "2024-03-01")]
    assert fc.date_chunks("2024-01-05", "2024-01-01", 7) == []


def test_adaptive_resplit_on_cap():
    def fn(p):
        if p["from"] != p["to"]:
            return [art(i) for i in range(5)]  # hits cap_threshold=5
        return [art(int(p["from"][-2:]) * 100)]
    s = FakeSession(fn)
    rep = fc.FetchReport()
    out = fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-07", "k", session=s,
                                        cap_threshold=5, report=rep)
    assert len(s.calls) == 1 + 7
    assert len(out) == 7
    assert rep.capped_chunks == []


def test_cap_at_one_day_is_recorded():
    s = FakeSession(lambda p: [art(i) for i in range(5)])
    rep = fc.FetchReport()
    fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-01", "k", session=s,
                                  cap_threshold=5, report=rep)
    assert rep.capped_chunks == [("AAPL", "2024-01-01", "2024-01-01", 5)]


def test_429_retry_then_success():
    state = {"n": 0}

    def fn(p):
        state["n"] += 1
        return ([], 429) if state["n"] < 3 else [art(1)]
    s = FakeSession(fn)
    rep = fc.FetchReport()
    out = fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-07", "k", session=s, report=rep)
    assert len(out) == 1 and len(s.calls) == 3 and rep.failed_chunks == []


def test_persistent_error_recorded_and_skipped():
    s = FakeSession(lambda p: ([], 500))
    rep = fc.FetchReport()
    out = fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-07", "k", session=s,
                                        max_retries=2, report=rep)
    assert out == [] and len(rep.failed_chunks) == 1


def test_cache_resume_makes_no_requests(tmp_path):
    s1 = FakeSession(lambda p: [art(1), art(2)])
    first = fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-14", "k", session=s1,
                                          cache_dir=tmp_path)
    s2 = FakeSession(lambda p: pytest.fail("should not call"))
    second = fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-14", "k", session=s2,
                                           cache_dir=tmp_path)
    assert s2.calls == [] and second == first


def test_dedupe_by_id_then_url():
    payloads = iter([[art(1), art(2)], [art(1), art(3, url="http://x/2")]])
    s = FakeSession(lambda p: next(payloads))
    out = fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-14", "k", session=s)
    assert [a["id"] for a in out] == [1, 2]


def test_missing_key_raises_clear_error():
    with pytest.raises(fc.FinnhubUnavailable):
        fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-07", None)
    with pytest.raises(fc.FinnhubUnavailable):
        fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-07", "  ")


def test_auth_failure_raises_unavailable():
    s = FakeSession(lambda p: ([], 401))
    with pytest.raises(fc.FinnhubUnavailable):
        fc.fetch_company_news_chunked("AAPL", "2024-01-01", "2024-01-07", "bad", session=s)


def _epoch(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def test_effective_day_around_close_and_weekend():
    days = get_nyse_trading_days("2024-01-01", "2024-02-29")
    raw = [
        art(1, _epoch("2024-01-16 20:00")),  # Tue 15:00 ET -> same day
        art(2, _epoch("2024-01-16 21:30")),  # Tue 16:30 ET -> Wed
        art(3, _epoch("2024-01-19 22:00")),  # Fri 17:00 ET -> Mon 22
        art(4, _epoch("2024-01-20 15:00")),  # Sat -> Mon 22
        art(5, _epoch("2024-01-15 15:00")),  # MLK holiday -> Tue 16
    ]
    df = fc.to_articles_frame(raw, "AAPL", days)
    got = [d.strftime("%Y-%m-%d") for d in df["effective_day"]]
    assert got == ["2024-01-16", "2024-01-17", "2024-01-22", "2024-01-22", "2024-01-16"]


def test_articles_frame_schema():
    days = get_nyse_trading_days("2024-01-01", "2024-02-29")
    df = fc.to_articles_frame([art(7, _epoch("2024-01-16 20:00"))], "AAPL", days)
    assert list(df.columns) == fc.ARTICLE_COLUMNS
    r = df.iloc[0]
    assert r["article_id"] == "7" and r["publisher"] == "Reuters" and r["source"] == "finnhub"
    assert r["alignment"] == "epoch" and str(df["published_at"].dt.tz) == "UTC"
    assert df["effective_day"].dtype == "datetime64[ns]"
    assert (df["effective_day"] == df["naive_effective_day"]).all()
    assert fc.to_articles_frame([], "AAPL", days).empty
