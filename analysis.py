"""Overnight vs. intraday return analysis. Pure pandas/scipy, no UI or network code."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
# An overnight move bigger than this on adjusted data usually means a bad split/dividend adjustment.
SUSPECT_MOVE = 0.40


def split_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Split each day's close-to-close return into night and day parts.

    night(t) = Open(t) / Close(t-1) - 1
    day(t)   = Close(t) / Open(t) - 1

    `prices` needs Open and Close columns on a DatetimeIndex, both adjusted
    for splits and dividends and from the same source.
    """
    df = prices[["Open", "Close"]].astype(float).dropna()
    df = df[(df > 0).all(axis=1)].sort_index()
    prev_close = df["Close"].shift(1)
    out = pd.DataFrame(index=df.index)
    out["night"] = df["Open"] / prev_close - 1
    out["day"] = df["Close"] / df["Open"] - 1
    out["total"] = df["Close"] / prev_close - 1
    out["gap_days"] = df.index.to_series().diff().dt.days
    out = out.dropna()
    for c in ("night", "day", "total"):
        out[f"{c}_log"] = np.log1p(out[c])
    return out


def sanity_check(r: pd.DataFrame) -> dict:
    """(1+night)(1+day)-1 must equal close-to-close; also flag implausible overnight gaps."""
    diff = ((1 + r["night"]) * (1 + r["day"]) - 1 - r["total"]).abs().max()
    suspects = r[(r["night"].abs() > SUSPECT_MOVE) | (r["day"].abs() > SUSPECT_MOVE)]
    return {"max_identity_error": float(diff), "suspect_days": suspects[["night", "day"]]}


def exclude_earnings(r: pd.DataFrame, earnings_dates) -> pd.DataFrame:
    """Drop each earnings date and the next trading day.

    Reports before the open land in that day's night return; reports after the
    close land in the next day's night return. Dropping both days covers both.
    """
    drop = set()
    idx = r.index
    for d in pd.to_datetime(list(earnings_dates)):
        d = d.tz_localize(None) if d.tzinfo else d
        pos = idx.searchsorted(d.normalize())
        for p in (pos, pos + 1):
            if p < len(idx):
                drop.add(idx[p])
    return r.drop(index=list(drop & set(idx)))


def weekend_split(r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weeknights, long gaps). A gap of 3+ calendar days is a weekend or holiday."""
    long = r["gap_days"] >= 3
    return r[~long], r[long]


def apply_costs(r: pd.DataFrame, bps_per_side: float) -> pd.DataFrame:
    """Each strategy trades twice a day (in and out), so it pays 2 sides daily."""
    c = 2 * bps_per_side / 1e4
    out = r.copy()
    out["night_log"] = out["night_log"] - c
    out["day_log"] = out["day_log"] - c
    return out


def leg_stats(log_ret: pd.Series) -> dict:
    n = len(log_ret)
    if n < 2:
        return dict(n=n, ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, pct_pos=np.nan, total=np.nan)
    mean, sd = log_ret.mean(), log_ret.std(ddof=1)
    ann_ret, ann_vol = mean * TRADING_DAYS, sd * np.sqrt(TRADING_DAYS)
    return dict(
        n=n,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=ann_ret / ann_vol if ann_vol > 0 else np.nan,
        pct_pos=float((log_ret > 0).mean()),
        total=float(np.expm1(log_ret.sum())),
    )


def summary_table(r: pd.DataFrame) -> pd.DataFrame:
    rows = {"Night": leg_stats(r["night_log"]), "Day": leg_stats(r["day_log"])}
    return pd.DataFrame(rows).T


def compare_test(r: pd.DataFrame) -> dict:
    """Do average night and day returns differ?

    The two series cover the same days, so the paired test is the right one;
    Welch's unpaired test is included for reference.
    """
    if len(r) < 3:
        return dict(diff_ann=np.nan, t_paired=np.nan, p_paired=np.nan, t_welch=np.nan, p_welch=np.nan)
    tp = stats.ttest_rel(r["night_log"], r["day_log"])
    tw = stats.ttest_ind(r["night_log"], r["day_log"], equal_var=False)
    return dict(
        diff_ann=(r["night_log"].mean() - r["day_log"].mean()) * TRADING_DAYS,
        t_paired=float(tp.statistic), p_paired=float(tp.pvalue),
        t_welch=float(tw.statistic), p_welch=float(tw.pvalue),
    )


def cumulative(r: pd.DataFrame) -> pd.DataFrame:
    """Growth of $1 held only overnight, only intraday, or buy-and-hold."""
    return np.exp(r[["night_log", "day_log", "total_log"]].cumsum()).rename(
        columns={"night_log": "Night only", "day_log": "Day only", "total_log": "Buy & hold"}
    )


def robustness_table(r: pd.DataFrame, earnings_dates=None, split_date=None) -> pd.DataFrame:
    """Night vs day across sample variations."""
    scenarios = {"All days": r}
    if earnings_dates is not None and len(earnings_dates):
        scenarios["Excluding earnings days"] = exclude_earnings(r, earnings_dates)
    week, long = weekend_split(r)
    scenarios["Weeknights only"] = week
    scenarios["Weekends/holidays only"] = long
    if split_date is not None:
        sd = pd.Timestamp(split_date)
        scenarios[f"Before {sd.date()}"] = r[r.index < sd]
        scenarios[f"From {sd.date()}"] = r[r.index >= sd]
    rows = []
    for name, sub in scenarios.items():
        t = compare_test(sub)
        rows.append({
            "Sample": name,
            "Days": len(sub),
            "Night ann. return": sub["night_log"].mean() * TRADING_DAYS if len(sub) else np.nan,
            "Day ann. return": sub["day_log"].mean() * TRADING_DAYS if len(sub) else np.nan,
            "Night − Day": t["diff_ann"],
            "p (paired t)": t["p_paired"],
        })
    return pd.DataFrame(rows).set_index("Sample")


def synthetic_prices(n_days: int = 2500, seed: int = 0, night_drift: float = 0.0008) -> pd.DataFrame:
    """Demo data with a built-in overnight drift, for offline use and tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    night = rng.normal(night_drift, 0.006, n_days)
    day = rng.normal(0.0000, 0.010, n_days)
    close = 100 * np.exp(np.cumsum(night + day))
    open_ = close * np.exp(-day)
    return pd.DataFrame({"Open": open_, "Close": close}, index=idx)
