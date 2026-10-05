# Day vs. night returns

Streamlit app comparing a stock's overnight (prior close → open) and intraday (open → close) returns.

```
pip install -r requirements.txt
streamlit run app.py
pytest
```

Tabs: cumulative chart and stats (annualized return, volatility, Sharpe, % positive days, paired and Welch t-tests);
robustness (ex-earnings, weeknights vs weekends/holidays, before/after a split date); benchmark vs SPY; trading costs.

Data: Yahoo Finance via `yfinance` (`auto_adjust=True`, so Open and Close are adjusted consistently), or a CSV
with `Date, Open, Close` columns that you have already adjusted. A demo mode with synthetic data works offline.
