"""Streamlit app: compare a stock's night (close→open) and day (open→close) returns.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import data

st.set_page_config(page_title="Night vs Day Returns", layout="wide")

NIGHT_COLOR, DAY_COLOR, HOLD_COLOR = "#3b5bdb", "#e8590c", "#868e96"


@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str) -> pd.DataFrame:
    return data.fetch_prices(ticker, start)


@st.cache_data(show_spinner=False)
def load_earnings(ticker: str) -> pd.DatetimeIndex:
    return data.fetch_earnings_dates(ticker)


def line_chart(df: pd.DataFrame, title: str, ytitle: str = "Cumulative log return") -> go.Figure:
    colors = [NIGHT_COLOR, DAY_COLOR, HOLD_COLOR]
    fig = go.Figure()
    for col, color in zip(df.columns, colors):
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col], name=col, mode="lines",
                line=dict(color=color, width=2 if color != HOLD_COLOR else 1.2,
                          dash="dot" if color == HOLD_COLOR else "solid"),
            )
        )
    fig.update_layout(
        title=title, yaxis_title=ytitle, hovermode="x unified", height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.add_hline(y=0, line_width=1, line_color="#adb5bd")
    return fig


def fmt_summary(df: pd.DataFrame) -> pd.DataFrame:
    pct = ["Ann. mean (log)", "Ann. volatility", "% positive", "Total log return"]
    out = df.copy().astype(object)
    for r in df.index:
        for c in df.columns:
            v = df.loc[r, c]
            if r in pct:
                out.loc[r, c] = f"{v:.2%}"
            elif r == "Days":
                out.loc[r, c] = f"{int(v):,}"
            elif r == "Growth of $1":
                out.loc[r, c] = f"${v:,.2f}"
            else:
                out.loc[r, c] = f"{v:.2f}"
    return out


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Data")
source = st.sidebar.radio("Source", ["Yahoo Finance", "Upload CSV"], horizontal=True)
ticker = st.sidebar.text_input("Ticker", "AAPL").strip().upper()
start = st.sidebar.date_input("Start date", dt.date(2000, 1, 1), min_value=dt.date(1970, 1, 1))
upload = bench_upload = None
if source == "Upload CSV":
    upload = st.sidebar.file_uploader("Stock CSV (Date, Open, Close — adjusted)", type="csv")

st.sidebar.header("Robustness checks")
excl_earn = st.sidebar.checkbox("Exclude earnings nights", value=False)
extra_dates = st.sidebar.text_area(
    "Extra dates to exclude (YYYY-MM-DD)", "",
    help="Night returns on these trading days are dropped. Useful for earnings "
         "dates Yahoo doesn't have, or for an uploaded CSV.",
)
split_date = st.sidebar.date_input("Period split date", dt.date(2009, 1, 1))
bench_ticker = st.sidebar.text_input("Benchmark ticker", "SPY").strip().upper()
if source == "Upload CSV":
    bench_upload = st.sidebar.file_uploader("Benchmark CSV (optional)", type="csv")
cost_bps = st.sidebar.slider("Trading cost per side (bp)", 0.0, 20.0, 2.0, 0.5,
                             help="Half-spread plus fees, paid at both entry and exit.")

st.title("Night vs Day Returns")
st.caption(
    "Each day's return is split into a **night** leg, Open(t) ÷ Close(t−1) − 1, and a "
    "**day** leg, Close(t) ÷ Open(t) − 1. Charts and stats use log returns so they add up over time."
)

# ---------------------------------------------------------------- load
try:
    if source == "Yahoo Finance":
        with st.spinner(f"Downloading {ticker}…"):
            prices = load_prices(ticker, str(start))
    else:
        if upload is None:
            st.info("Upload a CSV with Date, Open and Close columns (split- and dividend-adjusted).")
            st.stop()
        prices = data.read_price_csv(upload.getvalue())
        prices = prices[prices.index >= pd.Timestamp(start)]
        ticker = ticker or "Uploaded"
except Exception as e:  # network errors, bad ticker, bad CSV
    st.error(f"Couldn't load prices: {e}")
    st.stop()

returns = an.compute_returns(prices)
if len(returns) < 30:
    st.warning("Fewer than 30 trading days of data — results will be meaningless.")
    st.stop()

years = (returns.index[-1] - returns.index[0]).days / 365.25
if years < 10:
    st.warning(f"Only {years:.1f} years of data. Daily returns are noisy; 10+ years is better.")

bad = an.sanity_check(returns)
gaps = an.suspicious_gaps(returns)
with st.expander(
    f"Data checks — {'✅ consistent' if bad.empty else f'⚠️ {len(bad)} inconsistent rows'}"
    f"{'' if gaps.empty else f', ⚠️ {len(gaps)} night moves over 25%'}",
    expanded=not bad.empty or not gaps.empty,
):
    st.write(
        f"{len(returns):,} trading days from {returns.index[0].date()} to {returns.index[-1].date()}. "
        "(1 + night) × (1 + day) − 1 is checked against the close-to-close return for every day."
    )
    if not bad.empty:
        st.dataframe(bad)
    if not gaps.empty:
        st.write("Very large overnight moves can mean unadjusted splits or dividends. "
                 "Check these are real (e.g. earnings or news):")
        st.dataframe(gaps.style.format({"night": "{:.2%}"}))

# exclusions
excluded = pd.DatetimeIndex([])
if excl_earn:
    if source == "Yahoo Finance":
        earn = load_earnings(ticker)
        if len(earn) == 0:
            st.sidebar.warning("Yahoo returned no earnings dates; add them manually below.")
        excluded = excluded.union(an.earnings_night_dates(earn, returns.index))
    else:
        st.sidebar.info("Earnings dates aren't fetched for uploads; list them in the box below.")
try:
    manual = data.parse_date_list(extra_dates)
    excluded = excluded.union(an.earnings_night_dates(manual, returns.index) if excl_earn else manual)
except Exception:
    st.sidebar.error("Couldn't parse the extra dates.")
analysed = an.exclude_dates(returns, excluded) if len(excluded) else returns
if len(excluded):
    st.info(f"Excluding {len(returns) - len(analysed)} days (earnings/listed dates). "
            "Yahoo's earnings history often covers only recent years.")

# ---------------------------------------------------------------- overview
summary = an.summary_table(analysed)
test = an.significance(analysed)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Night ann. return", f"{summary.loc['Ann. mean (log)', 'Night']:.2%}")
c2.metric("Day ann. return", f"{summary.loc['Ann. mean (log)', 'Day']:.2%}")
c3.metric("Night − day (per day)", f"{test.mean_diff_daily * 1e4:.2f} bp")
c4.metric("Paired t-test p-value", f"{test.paired_p:.4f}",
          "significant at 5%" if test.paired_p < 0.05 else "not significant",
          delta_color="off")

tabs = st.tabs(["Cumulative", "Statistics", "Weekends", "Time periods", "Benchmark", "Trading costs", "Data"])

with tabs[0]:
    st.plotly_chart(line_chart(an.cumulative(analysed), f"{ticker}: cumulative night vs day log returns"),
                    width="stretch")
    st.caption("Running sum of log returns. exp(value) is the growth of $1 holding only that leg.")

with tabs[1]:
    st.subheader("Summary statistics")
    st.dataframe(fmt_summary(summary), width="stretch")
    st.caption("Annualized with 252 trading days. Sharpe uses a 0% risk-free rate.")
    st.subheader("Is the difference real?")
    st.markdown(
        f"""
| Test | t-statistic | p-value |
|---|---|---|
| Paired t-test (night − day, same date) | {test.paired_t:.2f} | {test.paired_p:.4g} |
| Welch two-sample t-test | {test.welch_t:.2f} | {test.welch_p:.4g} |

Night returns average **{test.mean_diff_daily * 1e4:.2f} bp/day** {'more' if test.mean_diff_daily > 0 else 'less'} than day returns.
A p-value under 0.05 suggests the gap is unlikely to be pure noise. The paired test is the better fit
because both legs come from the same trading day.
"""
    )

with tabs[2]:
    st.subheader("Weeknights vs weekend/holiday gaps")
    st.write("Friday close → Monday open spans ~3 calendar days, so those nights are split out.")
    ws = an.weekend_split(analysed)
    st.dataframe(ws.style.format("{:.2f}", subset=pd.IndexSlice[["Mean per occurrence (bp)"], :])
                 .format("{:.2%}", subset=pd.IndexSlice[["% positive", "Total log return"], :])
                 .format("{:,.0f}", subset=pd.IndexSlice[["Days"], :]),
                 width="stretch")
    wk = analysed.copy()
    wk["night_log"] = wk["night_log"].where(~wk["weekend"], 0.0)
    wkend = analysed.copy()
    wkend["night_log"] = wkend["night_log"].where(wkend["weekend"], 0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk.index, y=wk["night_log"].cumsum(), name="Weeknights",
                             line=dict(color=NIGHT_COLOR)))
    fig.add_trace(go.Scatter(x=wkend.index, y=wkend["night_log"].cumsum(), name="Weekends/holidays",
                             line=dict(color="#7048e8", dash="dash")))
    fig.add_trace(go.Scatter(x=analysed.index, y=analysed["day_log"].cumsum(), name="Day",
                             line=dict(color=DAY_COLOR)))
    fig.update_layout(height=380, hovermode="x unified", yaxis_title="Cumulative log return",
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

with tabs[3]:
    st.subheader(f"Split at {split_date}")
    pt = an.period_table(analysed, split_date)
    if pt.shape[1] < 2:
        st.warning("The split date is outside the data range, so there's only one period.")
    fmt = pt.copy().astype(object)
    for r in pt.index:
        fmt.loc[r] = [f"{v:,.0f}" if r == "Days" else f"{v:.2%}" if "mean" in r
                      else f"{v:.4g}" if "p-value" in r else f"{v:.2f}" for v in pt.loc[r]]
    st.dataframe(fmt, width="stretch")
    cols = st.columns(max(len(an.period_split(analysed, split_date)), 1))
    for col, (name, part) in zip(cols, an.period_split(analysed, split_date).items()):
        col.plotly_chart(line_chart(an.cumulative(part), name), width="stretch")

with tabs[4]:
    st.subheader(f"{ticker} vs {bench_ticker}")
    try:
        if source == "Upload CSV" and bench_upload is None:
            raise ValueError("upload a benchmark CSV in the sidebar to compare.")
        if source == "Upload CSV":
            bprices = data.read_price_csv(bench_upload.getvalue())
        else:
            bprices = load_prices(bench_ticker, str(start))
        bret = an.compute_returns(bprices)
        bret = an.exclude_dates(bret, excluded) if len(excluded) else bret
        bt = an.benchmark_table(analysed, bret, (ticker, bench_ticker))
        fmt = bt.copy().astype(object)
        for r in bt.index:
            fmt.loc[r] = [f"{v:.2%}" if "mean" in r else f"{v:.4g}" if "p-value" in r
                          else f"{v:.2f}" for v in bt.loc[r]]
        st.dataframe(fmt, width="stretch")
        common = analysed.index.intersection(bret.index)
        fig = go.Figure()
        for r, name, dash in ((analysed, ticker, "solid"), (bret, bench_ticker, "dot")):
            r = r.loc[common]
            fig.add_trace(go.Scatter(x=common, y=r["night_log"].cumsum(), name=f"{name} night",
                                     line=dict(color=NIGHT_COLOR, dash=dash)))
            fig.add_trace(go.Scatter(x=common, y=r["day_log"].cumsum(), name=f"{name} day",
                                     line=dict(color=DAY_COLOR, dash=dash)))
        fig.update_layout(height=420, hovermode="x unified", yaxis_title="Cumulative log return",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("High night/day correlation means the stock's pattern largely follows the market.")
    except Exception as e:
        st.error(f"Couldn't load benchmark: {e}")

with tabs[5]:
    st.subheader(f"Trading every day at {cost_bps:g} bp per side")
    st.write("Holding only the night (buy at close, sell at open) — or only the day — means a "
             "round trip every trading day, paying the spread and fees twice.")
    st.plotly_chart(line_chart(an.with_trading_costs(analysed, cost_bps), "Net of costs"),
                    width="stretch")
    be = an.breakeven_cost_bps(analysed)
    b1, b2 = st.columns(2)
    b1.metric("Break-even cost, night strategy", f"{be['Night']:.2f} bp/side")
    b2.metric("Break-even cost, day strategy", f"{be['Day']:.2f} bp/side")
    st.caption("Above the break-even cost, the strategy's average daily return is negative. "
               "Taxes, slippage and auction prices aren't modelled.")

with tabs[6]:
    show = analysed[["night", "day", "close_to_close", "gap_days"]].copy()
    st.dataframe(show.style.format({"night": "{:.3%}", "day": "{:.3%}", "close_to_close": "{:.3%}"}),
                 width="stretch", height=400)
    st.download_button("Download CSV", show.to_csv().encode(), f"{ticker}_night_day.csv", "text/csv")
