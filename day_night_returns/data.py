"""Price and earnings-date loaders."""
from __future__ import annotations

import io

import pandas as pd


def fetch_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Daily Open/Close from Yahoo Finance, both adjusted for splits and
    dividends (``auto_adjust=True`` adjusts every price column consistently)."""
    import yfinance as yf

    df = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, actions=False
    )
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker!r}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df[["Open", "Close"]].dropna()


def fetch_earnings_dates(ticker: str, limit: int = 100) -> pd.DatetimeIndex:
    """Historical earnings announcement timestamps from Yahoo (may be partial)."""
    import yfinance as yf

    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception:
        return pd.DatetimeIndex([])
    if df is None or df.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(df.index)


def read_price_csv(data: bytes | str) -> pd.DataFrame:
    """Load a CSV with a Date column plus Open and Close (case-insensitive).

    Works with Yahoo and Stooq exports. The prices must already be adjusted
    for splits and dividends.
    """
    buf = io.BytesIO(data) if isinstance(data, bytes) else data
    df = pd.read_csv(buf)
    cols = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in ("date", "open", "close") if c not in cols]
    if missing:
        raise ValueError(f"CSV is missing column(s): {', '.join(missing)}")
    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(df[cols["open"]], errors="coerce").values,
            "Close": pd.to_numeric(df[cols["close"]], errors="coerce").values,
        },
        index=pd.to_datetime(df[cols["date"]]).dt.tz_localize(None),
    )
    out.index.name = "Date"
    return out.dropna().sort_index()


def parse_date_list(text: str) -> pd.DatetimeIndex:
    """Parse dates separated by commas, spaces or newlines."""
    parts = [p for p in text.replace(",", " ").split() if p]
    return pd.DatetimeIndex(pd.to_datetime(parts)) if parts else pd.DatetimeIndex([])
