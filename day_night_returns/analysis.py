"""Core analysis: split each day's return into a night (close->open) and a
day (open->close) leg, then compare the two series.

All functions take a DataFrame indexed by trading date with ``Open`` and
``Close`` columns that are adjusted for splits and dividends.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with simple and log night/day returns for each day t.

    night = Open(t) / Close(t-1) - 1
    day   = Close(t) / Open(t) - 1
    """
    df = prices[["Open", "Close"]].astype(float).dropna().sort_index()
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    prev_close = df["Close"].shift(1)

    out = pd.DataFrame(index=df.index)
    out["night"] = df["Open"] / prev_close - 1
    out["day"] = df["Close"] / df["Open"] - 1
    out["close_to_close"] = df["Close"] / prev_close - 1
    out["night_log"] = np.log(df["Open"] / prev_close)
    out["day_log"] = np.log(df["Close"] / df["Open"])

    # Calendar days covered by the night leg: 1 on a normal weeknight,
    # 3 for Friday->Monday, more around holidays.
    dates = pd.Series(pd.to_datetime(df.index), index=df.index)
    out["gap_days"] = dates.diff().dt.days
    out["weekend"] = out["gap_days"] > 1
    return out.iloc[1:]


def sanity_check(returns: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Rows where (1+night)*(1+day)-1 differs from the close-to-close return.

    An empty result means the data is internally consistent.
    """
    recon = (1 + returns["night"]) * (1 + returns["day"]) - 1
    bad = (recon - returns["close_to_close"]).abs() > tol
    return returns.loc[bad, ["night", "day", "close_to_close"]]


def suspicious_gaps(returns: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """Night returns large enough to suggest an unadjusted split or dividend."""
    return returns.loc[returns["night"].abs() > threshold, ["night", "gap_days"]]


def cumulative(returns: pd.DataFrame) -> pd.DataFrame:
    """Running total of log returns (sums cleanly over time)."""
    return pd.DataFrame(
        {
            "Night (close→open)": returns["night_log"].cumsum(),
            "Day (open→close)": returns["day_log"].cumsum(),
            "Buy & hold": (returns["night_log"] + returns["day_log"]).cumsum(),
        }
    )


@dataclass
class Summary:
    n: int
    ann_mean: float
    ann_vol: float
    sharpe: float
    pct_positive: float
    total_log: float


def summarize(series: pd.Series) -> Summary:
    """Annualized stats for a series of daily log returns (risk-free = 0)."""
    s = series.dropna()
    mean = s.mean() * TRADING_DAYS
    vol = s.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return Summary(
        n=len(s),
        ann_mean=mean,
        ann_vol=vol,
        sharpe=mean / vol if vol > 0 else np.nan,
        pct_positive=(s > 0).mean() if len(s) else np.nan,
        total_log=s.sum(),
    )


def summary_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = {
        "Night": summarize(returns["night_log"]),
        "Day": summarize(returns["day_log"]),
        "Buy & hold": summarize(returns["night_log"] + returns["day_log"]),
    }
    return pd.DataFrame(
        {
            name: {
                "Days": s.n,
                "Ann. mean (log)": s.ann_mean,
                "Ann. volatility": s.ann_vol,
                "Sharpe (rf=0)": s.sharpe,
                "% positive": s.pct_positive,
                "Total log return": s.total_log,
                "Growth of $1": np.exp(s.total_log),
            }
            for name, s in rows.items()
        }
    )


@dataclass
class TestResult:
    mean_diff_daily: float
    paired_t: float
    paired_p: float
    welch_t: float
    welch_p: float


def significance(returns: pd.DataFrame) -> TestResult:
    """Do average night returns differ from average day returns?

    The paired test (on night - day for the same date) is the primary one,
    since both legs come from the same day. Welch's two-sample test is shown
    for comparison.
    """
    r = returns[["night_log", "day_log"]].dropna()
    if len(r) < 3:
        return TestResult(np.nan, np.nan, np.nan, np.nan, np.nan)
    paired = stats.ttest_rel(r["night_log"], r["day_log"])
    welch = stats.ttest_ind(r["night_log"], r["day_log"], equal_var=False)
    return TestResult(
        mean_diff_daily=(r["night_log"] - r["day_log"]).mean(),
        paired_t=float(paired.statistic),
        paired_p=float(paired.pvalue),
        welch_t=float(welch.statistic),
        welch_p=float(welch.pvalue),
    )


# ---------------------------------------------------------------- robustness


def earnings_night_dates(
    earnings: pd.Series | pd.DatetimeIndex, trading_days: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Map earnings announcement timestamps to the trading day whose night
    return absorbs the surprise.

    - A morning (before-noon) timestamp lands in that day's night return.
    - An afternoon/evening timestamp lands in the *next* trading day's.
    - A date with no time (midnight) is ambiguous, so both are flagged.
    """
    trading_days = pd.DatetimeIndex(trading_days).normalize()
    ts = pd.DatetimeIndex(pd.to_datetime(earnings))
    if ts.tz is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    hit: set[pd.Timestamp] = set()
    for t in ts:
        day = t.normalize()
        pos = trading_days.searchsorted(day)  # first trading day >= day
        same = trading_days[pos] if pos < len(trading_days) else None
        if same is not None and same == day:
            nxt = trading_days[pos + 1] if pos + 1 < len(trading_days) else None
        else:
            nxt = same  # announced on a non-trading day: next session absorbs it
            same = None
        has_time = t != day
        if has_time and t.hour < 12:
            targets = [same or nxt]
        elif has_time:
            targets = [nxt]
        else:
            targets = [same, nxt]
        hit.update(d for d in targets if d is not None)
    return pd.DatetimeIndex(sorted(hit))


def exclude_dates(returns: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(returns.index).normalize()
    return returns.loc[~idx.isin(pd.DatetimeIndex(dates).normalize())]


def weekend_split(returns: pd.DataFrame) -> pd.DataFrame:
    """Night-return stats for weeknights vs weekend/holiday gaps."""
    groups = {
        "Weeknight": returns.loc[~returns["weekend"], "night_log"],
        "Weekend / holiday": returns.loc[returns["weekend"], "night_log"],
        "Day (all)": returns["day_log"],
    }
    rows = {}
    for name, s in groups.items():
        sm = summarize(s)
        rows[name] = {
            "Days": sm.n,
            "Mean per occurrence (bp)": s.mean() * 1e4 if len(s) else np.nan,
            "% positive": sm.pct_positive,
            "Total log return": sm.total_log,
        }
    return pd.DataFrame(rows)


def period_split(returns: pd.DataFrame, split: str | pd.Timestamp) -> dict[str, pd.DataFrame]:
    split = pd.Timestamp(split)
    idx = pd.DatetimeIndex(returns.index)
    out = {}
    before, after = returns.loc[idx < split], returns.loc[idx >= split]
    if len(before):
        out[f"Before {split.date()}"] = before
    if len(after):
        out[f"From {split.date()}"] = after
    return out


def period_table(returns: pd.DataFrame, split: str | pd.Timestamp) -> pd.DataFrame:
    rows = {}
    for name, part in period_split(returns, split).items():
        n, d = summarize(part["night_log"]), summarize(part["day_log"])
        t = significance(part)
        rows[name] = {
            "Days": n.n,
            "Night ann. mean": n.ann_mean,
            "Day ann. mean": d.ann_mean,
            "Night Sharpe": n.sharpe,
            "Day Sharpe": d.sharpe,
            "Paired t p-value": t.paired_p,
        }
    return pd.DataFrame(rows)


def with_trading_costs(returns: pd.DataFrame, cost_bps_per_side: float) -> pd.DataFrame:
    """Cumulative log return of trading each leg daily, net of costs.

    Holding only the night (buy at close, sell at open) or only the day
    (buy at open, sell at close) means a round trip every day, i.e. paying
    ``cost_bps_per_side`` twice.
    """
    round_trip = 2 * cost_bps_per_side / 1e4
    net_night = np.log1p(returns["night"] - round_trip)
    net_day = np.log1p(returns["day"] - round_trip)
    return pd.DataFrame(
        {
            "Night, net of costs": net_night.cumsum(),
            "Day, net of costs": net_day.cumsum(),
            "Buy & hold": (returns["night_log"] + returns["day_log"]).cumsum(),
        }
    )


def breakeven_cost_bps(returns: pd.DataFrame) -> dict[str, float]:
    """Per-side cost (bp) at which each leg's average simple return hits zero."""
    return {
        "Night": returns["night"].mean() / 2 * 1e4,
        "Day": returns["day"].mean() / 2 * 1e4,
    }


def benchmark_table(stock: pd.DataFrame, bench: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    """Side-by-side night/day stats for a stock and a benchmark over common dates."""
    common = stock.index.intersection(bench.index)
    s, b = stock.loc[common], bench.loc[common]
    rows = {}
    for label, r in zip(names, (s, b)):
        n, d = summarize(r["night_log"]), summarize(r["day_log"])
        rows[label] = {
            "Night ann. mean": n.ann_mean,
            "Day ann. mean": d.ann_mean,
            "Night Sharpe": n.sharpe,
            "Day Sharpe": d.sharpe,
            "Paired t p-value": significance(r).paired_p,
        }
    df = pd.DataFrame(rows)
    df.loc["Night corr. with benchmark"] = [s["night_log"].corr(b["night_log"]), 1.0]
    df.loc["Day corr. with benchmark"] = [s["day_log"].corr(b["day_log"]), 1.0]
    return df
