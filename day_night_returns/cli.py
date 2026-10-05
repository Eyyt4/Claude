"""Command-line version: print the night/day report and optionally save a chart.

    python cli.py AAPL --start 2005-01-01 --exclude-earnings --plot aapl.png
    python cli.py --csv prices.csv --benchmark-csv spy.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

import analysis as an
import data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ticker", nargs="?", default="AAPL")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--csv", help="Use a local CSV (Date, Open, Close) instead of Yahoo")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--benchmark-csv")
    p.add_argument("--no-benchmark", action="store_true")
    p.add_argument("--exclude-earnings", action="store_true")
    p.add_argument("--exclude-dates", default="", help="Comma-separated dates to drop")
    p.add_argument("--split", default="2009-01-01")
    p.add_argument("--cost-bps", type=float, default=2.0, help="Trading cost per side, bp")
    p.add_argument("--plot", help="Save the cumulative chart to this PNG path")
    a = p.parse_args()

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    prices = data.read_price_csv(a.csv) if a.csv else data.fetch_prices(a.ticker, a.start)
    prices = prices[prices.index >= pd.Timestamp(a.start)]
    r = an.compute_returns(prices)

    bad = an.sanity_check(r)
    print(f"{a.ticker}: {len(r):,} days, {r.index[0].date()} → {r.index[-1].date()}")
    print("Sanity check:", "OK" if bad.empty else f"{len(bad)} inconsistent rows")
    gaps = an.suspicious_gaps(r)
    if not gaps.empty:
        print(f"Warning: {len(gaps)} night moves over 25% — check for unadjusted splits/dividends")

    excluded = data.parse_date_list(a.exclude_dates)
    if a.exclude_earnings:
        earn = data.fetch_earnings_dates(a.ticker) if not a.csv else pd.DatetimeIndex([])
        excluded = an.earnings_night_dates(excluded.union(earn), r.index)
    if len(excluded):
        r = an.exclude_dates(r, excluded)
        print(f"Excluded {len(excluded)} dates")

    print("\n== Summary ==")
    print(an.summary_table(r))
    t = an.significance(r)
    print(f"\nNight − day: {t.mean_diff_daily * 1e4:.2f} bp/day | "
          f"paired t={t.paired_t:.2f}, p={t.paired_p:.4g} | Welch t={t.welch_t:.2f}, p={t.welch_p:.4g}")

    print("\n== Weeknights vs weekends ==")
    print(an.weekend_split(r))
    print(f"\n== Split at {a.split} ==")
    print(an.period_table(r, a.split))

    be = an.breakeven_cost_bps(r)
    net = an.with_trading_costs(r, a.cost_bps).iloc[-1]
    print(f"\n== Trading costs ({a.cost_bps:g} bp/side) ==")
    print(f"Break-even per side: night {be['Night']:.2f} bp, day {be['Day']:.2f} bp")
    print(net.to_string())

    if not a.no_benchmark:
        try:
            bp = data.read_price_csv(a.benchmark_csv) if a.benchmark_csv else data.fetch_prices(a.benchmark, a.start)
            br = an.compute_returns(bp)
            if len(excluded):
                br = an.exclude_dates(br, excluded)
            print(f"\n== Benchmark: {a.benchmark} ==")
            print(an.benchmark_table(r, br, (a.ticker, a.benchmark)))
        except Exception as e:
            print(f"\nBenchmark skipped: {e}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = an.cumulative(r).plot(figsize=(11, 5), color=["#3b5bdb", "#e8590c", "#868e96"])
        ax.set_title(f"{a.ticker}: cumulative night vs day log returns")
        ax.set_ylabel("Cumulative log return")
        ax.axhline(0, color="#adb5bd", lw=1)
        plt.tight_layout()
        plt.savefig(a.plot, dpi=120)
        print(f"\nChart saved to {a.plot}")


if __name__ == "__main__":
    main()
