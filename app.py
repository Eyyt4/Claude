"""Streamlit app: compare stock returns earned overnight vs. during the trading day."""
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import data

st.set_page_config(page_title="Day vs Night Returns", layout="wide")
st.title("Day vs. night stock returns")
st.caption("Night = prior close → open. Day = open → close. Adjusted prices, log returns.")

NIGHT, DAY, HOLD = "#4c78a8", "#f58518", "#8c8c8c"
PCT = "{:.2%}"

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    source = st.radio("Data source", ["Yahoo Finance", "Upload CSV", "Demo (synthetic)"])
    if source == "Yahoo Finance":
        ticker = st.text_input("Ticker", "AAPL").strip().upper()
        bench = st.text_input("Benchmark", "SPY").strip().upper()
        start = st.date_input("Start", date(2010, 1, 1), min_value=date(1990, 1, 1))
    elif source == "Upload CSV":
        ticker = st.text_input("Label", "CUSTOM").strip().upper()
        up = st.file_uploader("Date, Open, Close (adjusted)", type="csv")
        bench_up = st.file_uploader("Benchmark CSV (optional)", type="csv")
        bench = "Benchmark"
    else:
        ticker, bench = "DEMO", "DEMO-B"
    st.divider()
    split_date = st.date_input("Split period at", date(2009, 6, 30))
    use_earn = st.checkbox("Exclude earnings days", value=source == "Yahoo Finance",
                           disabled=source != "Yahoo Finance")
    bps = st.number_input("Trading cost (bps per side)", 0.0, 100.0, 1.0, 0.5,
                          help="Charged twice a day: once to get in, once to get out.")


@st.cache_data(show_spinner="Downloading prices…", ttl=3600)
def _yahoo(t, s):
    return data.load_yahoo(t, s)


@st.cache_data(show_spinner=False, ttl=86400)
def _earn(t):
    return data.load_earnings_dates(t)


def get_prices():
    if source == "Yahoo Finance":
        return _yahoo(ticker, start), _yahoo(bench, start) if bench else None
    if source == "Upload CSV":
        if up is None:
            st.info("Upload a CSV to begin.")
            st.stop()
        return data.load_csv(up), (data.load_csv(bench_up) if bench_up else None)
    return an.synthetic_prices(seed=1), an.synthetic_prices(seed=2)


try:
    px, bpx = get_prices()
except Exception as e:  # network, bad ticker, bad CSV
    st.error(str(e))
    st.stop()

r = an.split_returns(px)
rb = an.split_returns(bpx) if bpx is not None and len(bpx) > 2 else None
earn = _earn(ticker) if use_earn else []
if use_earn and not earn:
    st.warning("No earnings dates found for this ticker; showing results without that filter.")

chk = an.sanity_check(r)
if chk["max_identity_error"] > 1e-9 or len(chk["suspect_days"]):
    st.warning(
        f"Data check: {len(chk['suspect_days'])} day(s) with a >{an.SUSPECT_MOVE:.0%} night or day move. "
        "This often means prices are not consistently adjusted for splits/dividends."
    )
    with st.expander("Suspect days"):
        st.dataframe(chk["suspect_days"].style.format(PCT))

st.caption(f"{ticker}: {r.index[0].date()} → {r.index[-1].date()} · {len(r):,} trading days")


# ---------------------------------------------------------------- helpers
def growth_chart(frames: dict[str, pd.Series], log_y=True):
    fig = go.Figure()
    for name, s in frames.items():
        fig.add_scatter(x=s.index, y=s, name=name, mode="lines")
    fig.update_yaxes(type="log" if log_y else "linear", title="Growth of $1")
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.05))
    return fig


def fmt_summary(t):
    return t.style.format({"n": "{:,.0f}", "ann_return": PCT, "ann_vol": PCT, "sharpe": "{:.2f}",
                           "pct_pos": "{:.1%}", "total": "{:,.1%}"}, na_rep="–").set_properties(**{})


def verdict(t):
    if pd.isna(t["p_paired"]):
        return
    d = "night" if t["diff_ann"] > 0 else "day"
    sig = t["p_paired"] < 0.05
    (st.success if sig else st.info)(
        f"Average {d} returns are higher by {abs(t['diff_ann']):.1%} a year "
        f"(paired t = {t['t_paired']:.2f}, p = {t['p_paired']:.3f}). "
        + ("The gap is statistically significant at 5%." if sig else "Not significant at 5%: could be noise.")
    )


def show_test(t):
    c = st.columns(4)
    c[0].metric("Night − Day (ann.)", f"{t['diff_ann']:.2%}")
    c[1].metric("Paired t / p", f"{t['t_paired']:.2f} / {t['p_paired']:.3f}")
    c[2].metric("Welch t / p", f"{t['t_welch']:.2f} / {t['p_welch']:.3f}")
    c[3].metric("Days", f"{len(r):,}")
    verdict(t)


# ---------------------------------------------------------------- tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Robustness", "Benchmark", "Trading costs", "Data"])

with tab1:
    cum = an.cumulative(r)
    st.plotly_chart(growth_chart({c: cum[c] for c in cum}).update_traces(
        selector=dict(name="Night only"), line_color=NIGHT).update_traces(
        selector=dict(name="Day only"), line_color=DAY).update_traces(
        selector=dict(name="Buy & hold"), line_color=HOLD, line_dash="dot"), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(fmt_summary(an.summary_table(r)), use_container_width=True)
    st.subheader("Is the gap real?")
    show_test(an.compare_test(r))

with tab2:
    st.markdown("Does the pattern survive different cuts of the sample?")
    rt = an.robustness_table(r, earn if use_earn else None, split_date)
    st.dataframe(rt.style.format({"Days": "{:,.0f}", "Night ann. return": PCT, "Day ann. return": PCT,
                                  "Night − Day": PCT, "p (paired t)": "{:.3f}"}, na_rep="–"),
                 use_container_width=True)
    st.caption("Weekends/holidays = night returns spanning 3+ calendar days (e.g. Friday close → Monday open). "
               "Earnings filter drops each report date and the following trading day.")

with tab3:
    if rb is None:
        st.info("No benchmark data loaded.")
    else:
        bc = an.cumulative(rb)
        st.plotly_chart(growth_chart({
            f"{ticker} night": cum["Night only"], f"{ticker} day": cum["Day only"],
            f"{bench} night": bc["Night only"], f"{bench} day": bc["Day only"],
        }), use_container_width=True)
        both = pd.concat({ticker: an.summary_table(r), bench: an.summary_table(rb)}, names=["Asset", "Leg"])
        st.dataframe(fmt_summary(both), use_container_width=True)
        a, b = an.compare_test(r), an.compare_test(rb)
        st.markdown(f"**{ticker}** night − day: {a['diff_ann']:.2%} (p = {a['p_paired']:.3f}) · "
                    f"**{bench}**: {b['diff_ann']:.2%} (p = {b['p_paired']:.3f}). "
                    "If both look alike, the pattern is the market's, not the stock's.")

with tab4:
    rc = an.apply_costs(r, bps)
    cc = an.cumulative(rc)
    st.plotly_chart(growth_chart({
        "Night only (net)": cc["Night only"], "Day only (net)": cc["Day only"],
        "Night only (gross)": cum["Night only"], "Day only (gross)": cum["Day only"],
    }).update_traces(selector=dict(name="Night only (net)"), line_color=NIGHT)
      .update_traces(selector=dict(name="Day only (net)"), line_color=DAY)
      .update_traces(selector=dict(name="Night only (gross)"), line_color=NIGHT, line_dash="dot", opacity=.5)
      .update_traces(selector=dict(name="Day only (gross)"), line_color=DAY, line_dash="dot", opacity=.5),
      use_container_width=True)
    st.dataframe(fmt_summary(an.summary_table(rc)), use_container_width=True)
    st.caption(f"Net of {bps:g} bps per side = {2 * bps:g} bps per day. "
               "Each strategy trades in and out daily, which often erases the edge.")

with tab5:
    st.dataframe(r.style.format({c: "{:.3%}" for c in ["night", "day", "total", "night_log", "day_log", "total_log"]}),
                 use_container_width=True, height=400)
    st.download_button("Download returns CSV", r.to_csv().encode(), f"{ticker}_day_night.csv", "text/csv")
