import pandas as pd
import yfinance as yf


def normalize_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    df = df.reset_index()
    df = df.rename(columns={
        "Date": "time", "Datetime": "time", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })
    if "close" not in df.columns:
        return pd.DataFrame()
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def load_history(symbol: str, timeframe: str) -> pd.DataFrame:
    intervals = {"5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}
    periods = {"5m": "60d", "15m": "60d", "1h": "730d", "1d": "5y"}
    raw = yf.Ticker(symbol).history(
        period=periods[timeframe], interval=intervals[timeframe],
        auto_adjust=False, actions=False, timeout=20
    )
    return normalize_history(raw)
