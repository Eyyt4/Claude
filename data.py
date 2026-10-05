"""Data loading: Yahoo Finance (via yfinance) or a user-supplied CSV."""
from __future__ import annotations

import io

import pandas as pd


def load_yahoo(ticker: str, start, end=None) -> pd.DataFrame:
    """Daily Open/Close with auto_adjust=True, so both columns are split and dividend adjusted."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker!r}. Check the ticker or your network access.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "Close"]].dropna()


def load_earnings_dates(ticker: str) -> list[pd.Timestamp]:
    """Best-effort historical earnings dates. Returns [] if unavailable (e.g. ETFs)."""
    try:
        import yfinance as yf

        e = yf.Ticker(ticker).get_earnings_dates(limit=100)
        if e is None or e.empty:
            return []
        return sorted(pd.to_datetime(e.index).tz_localize(None).normalize().unique())
    except Exception:
        return []


def load_csv(file) -> pd.DataFrame:
    """CSV with Date, Open and Close columns (case-insensitive). Prices must already be adjusted."""
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    df = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw))
    df.columns = [c.strip().title() for c in df.columns]
    missing = {"Date", "Open", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()[["Open", "Close"]].dropna()
