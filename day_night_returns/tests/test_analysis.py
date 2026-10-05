import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as an  # noqa: E402
import data  # noqa: E402


def make_prices(n=600, night=0.0005, day=-0.0002, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    close = [100.0]
    opens = [100.0]
    for _ in range(n - 1):
        o = close[-1] * np.exp(night + rng.normal(0, 0.005))
        c = o * np.exp(day + rng.normal(0, 0.01))
        opens.append(o)
        close.append(c)
    return pd.DataFrame({"Open": opens, "Close": close}, index=idx)


def test_legs_multiply_to_close_to_close():
    r = an.compute_returns(make_prices())
    assert an.sanity_check(r).empty
    np.testing.assert_allclose(r["night_log"] + r["day_log"], np.log1p(r["close_to_close"]))


def test_known_values():
    p = pd.DataFrame({"Open": [10, 11, 12], "Close": [10, 12, 9]},
                     index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]))
    r = an.compute_returns(p)
    assert r["night"].tolist() == pytest.approx([0.1, 0.0])
    assert r["day"].tolist() == pytest.approx([12 / 11 - 1, -0.25])
    assert r["weekend"].tolist() == [True, False]


def test_significance_detects_drift():
    r = an.compute_returns(make_prices(n=3000, night=0.001, day=-0.001))
    t = an.significance(r)
    assert t.mean_diff_daily > 0 and t.paired_p < 0.01
    s = an.summary_table(r)
    assert s.loc["Ann. mean (log)", "Night"] > 0 > s.loc["Ann. mean (log)", "Day"]


def test_earnings_mapping():
    days = pd.bdate_range("2024-01-01", "2024-01-31")
    hits = an.earnings_night_dates(
        pd.to_datetime(["2024-01-10 07:00", "2024-01-17 16:30", "2024-01-24", "2024-01-27 16:00"], format="mixed"),
        days,
    )
    assert list(hits.strftime("%m-%d")) == ["01-10", "01-18", "01-24", "01-25", "01-29"]


def test_exclude_period_costs_benchmark():
    r = an.compute_returns(make_prices())
    dropped = an.exclude_dates(r, r.index[:5])
    assert len(dropped) == len(r) - 5
    assert set(an.period_table(r, "2006-01-01").columns) == {"Before 2006-01-01", "From 2006-01-01"}
    free, costly = an.with_trading_costs(r, 0), an.with_trading_costs(r, 5)
    assert costly.iloc[-1, 0] < free.iloc[-1, 0]
    np.testing.assert_allclose(free.iloc[-1, 0], r["night_log"].sum())
    bt = an.benchmark_table(r, an.compute_returns(make_prices(seed=1)), ("A", "B"))
    assert bt.loc["Night corr. with benchmark", "B"] == 1.0
    ws = an.weekend_split(r)
    assert ws.loc["Days", "Weeknight"] + ws.loc["Days", "Weekend / holiday"] == len(r)


def test_read_csv_case_insensitive():
    csv = "date,open,high,low,close\n2024-01-02,1,2,0.5,1.5\n2024-01-03,1.6,2,1,1.8\n"
    df = data.read_price_csv(csv.encode())
    assert list(df.columns) == ["Open", "Close"] and len(df) == 2
