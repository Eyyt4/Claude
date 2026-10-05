# Night vs Day Returns

Compares a stock's **night** returns (previous close → open) with its **day** returns (open → close), in the style of Knuteson's overnight-return studies.

For each trading day *t*:

- Night return = Open(t) ÷ Close(t−1) − 1
- Day return = Close(t) ÷ Open(t) − 1

The app checks every day that (1 + night) × (1 + day) − 1 equals the close-to-close return. Charts and summary stats use log returns, so the values add up over time.

## Run it

```bash
cd day_night_returns
pip install -r requirements.txt
streamlit run app.py          # interactive app
python cli.py AAPL --exclude-earnings --plot aapl.png   # text report
pytest -q tests
```

Prices come from Yahoo Finance via `yfinance` with `auto_adjust=True`, which adjusts Open and Close for splits and dividends together. You can also upload a CSV with `Date`, `Open` and `Close` columns, such as a Yahoo or Stooq export. The prices in it must already be adjusted.

## What the app shows

| Tab | Contents |
|---|---|
| Cumulative | Running sum of night, day and buy-and-hold log returns |
| Statistics | Annualized mean, volatility, Sharpe (rf = 0), % positive days, growth of $1. Paired and Welch t-tests on night vs day |
| Weekends | Weeknight nights vs weekend/holiday gaps (any night spanning more than 1 calendar day) |
| Time periods | The same stats before and after a split date you choose (default 2009-01-01) |
| Benchmark | Side by side with SPY (or any ticker), plus night/day correlations |
| Trading costs | Cumulative returns for trading each leg every day, after a per-side cost, and the break-even cost |
| Data | Daily return table with CSV download |

**Excluding earnings.** Earnings dates come from Yahoo. A morning announcement drops that day's night return. An after-close announcement drops the next trading day's night return. A date with no time drops both. Yahoo's earnings history often covers only recent years, so you can paste more dates in the sidebar.

The **Data checks** panel flags inconsistent rows and any night move over 25%. A move that large often means a split or dividend wasn't adjusted.

## Files

- `analysis.py` – return calculations, stats, tests and robustness checks (no I/O)
- `data.py` – Yahoo and CSV loaders
- `app.py` – Streamlit UI
- `cli.py` – command-line report and matplotlib chart
- `tests/` – unit tests on synthetic data
