import numpy as np
import pandas as pd

import analysis as an


def test_identity_and_split():
    p = an.synthetic_prices(500)
    r = an.split_returns(p)
    assert an.sanity_check(r)["max_identity_error"] < 1e-12
    assert np.allclose(r["night_log"] + r["day_log"], r["total_log"])
    assert np.isclose(r["night"].iloc[0], p["Open"].iloc[1] / p["Close"].iloc[0] - 1)


def test_unadjusted_split_flagged():
    p = an.synthetic_prices(100)
    p.iloc[50:] = p.iloc[50:] / 2  # fake 2:1 split not adjusted back
    assert len(an.sanity_check(an.split_returns(p))["suspect_days"]) == 1


def test_detects_overnight_drift():
    r = an.split_returns(an.synthetic_prices(3000, night_drift=0.002))
    t = an.compare_test(r)
    assert t["diff_ann"] > 0 and t["p_paired"] < 0.05


def test_earnings_drops_date_and_next_day():
    r = an.split_returns(an.synthetic_prices(50))
    e = r.index[10]
    out = an.exclude_earnings(r, [e])
    assert len(out) == len(r) - 2 and e not in out.index and r.index[11] not in out.index


def test_weekend_split():
    r = an.split_returns(an.synthetic_prices(300))
    wk, long = an.weekend_split(r)
    assert len(wk) + len(long) == len(r)
    assert (long.index.weekday == 0).all()


def test_costs_reduce_returns():
    r = an.split_returns(an.synthetic_prices(300))
    net = an.apply_costs(r, 2)
    assert np.isclose(r["night_log"].mean() - net["night_log"].mean(), 4e-4)


def test_robustness_table_runs():
    r = an.split_returns(an.synthetic_prices(1000))
    t = an.robustness_table(r, [r.index[5]], r.index[500])
    assert len(t) == 6
