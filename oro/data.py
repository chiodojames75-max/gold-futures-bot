"""
Loading and preparing price bars.

Accepts the common CSV exports:
  - Databento ohlcv-1m / ohlcv-1h  (ts_event in UTC nanoseconds or ISO)
  - TradingView "Export chart data" (time as unix seconds)
  - Any CSV with a datetime column + open/high/low/close[/volume]

Everything is converted to America/New_York and resampled to the strategy bar size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
_TIME_COLS = ("ts_event", "time", "timestamp", "datetime", "date")


def front_month(raw: pd.DataFrame) -> pd.DataFrame:
    """Databento parent-symbol downloads (e.g. GC) contain every expiry plus calendar spreads.
    Keep outright contracts only and, for each UTC day, the one with the most volume
    (the front month). Roll happens on the day volume moves to the next contract."""
    if "symbol" not in raw.columns:
        return raw
    out = raw[~raw["symbol"].astype(str).str.contains("-")].copy()   # drop spreads like GCZ6-GCG7
    day = pd.to_datetime(out["ts_event"], utc=True).dt.date
    vol = out.groupby([day, out["symbol"]])["volume"].sum()
    best = vol.groupby(level=0).idxmax().map(lambda x: x[1])
    keep = out["symbol"].to_numpy() == best.reindex(day).to_numpy()
    return out[keep]


def load_csv(path, tz_if_naive: str = "UTC") -> pd.DataFrame:
    paths = path if isinstance(path, (list, tuple)) else [path]
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw = front_month(raw)
    tcol = next((c for c in _TIME_COLS if c in raw.columns), None)
    if tcol is None:
        raise ValueError(f"No time column found. Columns: {list(raw.columns)}")

    t = raw[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ns" if t.iloc[0] > 1e15 else ("ms" if t.iloc[0] > 1e12 else "s")
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        has_offset = t.astype(str).str.contains(r"[+-]\d\d:\d\d$|Z$", regex=True).all()
        if has_offset:
            idx = pd.to_datetime(t, utc=True)
        else:
            idx = pd.to_datetime(t).dt.tz_localize(tz_if_naive)
    df = raw[["open", "high", "low", "close"] + (["volume"] if "volume" in raw.columns else [])].copy()
    df.index = pd.DatetimeIndex(idx).tz_convert(ET)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df.astype(float)


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Bars are labelled by their OPEN time (a 9:30 bar covers 9:30-9:35)."""
    rule = f"{minutes}min"
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    trade_date: Topstep trading day. A new day starts at 6:00 PM ET (5:00 PM CT),
                so Tuesday 7 PM ET belongs to Wednesday.
    tod:        minutes since midnight ET (for session windows).
    """
    df = df.copy()
    shifted = df.index + pd.Timedelta(hours=6)
    df["trade_date"] = shifted.date
    df["tod"] = df.index.hour * 60 + df.index.minute
    df["sm"] = (df["tod"] - 18 * 60) % 1440  # minutes since 6 PM ET session open
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].where(df["volume"] > 0, 1.0)  # fall back to equal weights when volume missing
    pv = (typical * vol).groupby(df["trade_date"]).cumsum()
    vv = vol.groupby(df["trade_date"]).cumsum()
    df["vwap"] = pv / vv
    # Daily context, using ONLY completed prior sessions
    g = df.groupby("trade_date")
    daily = pd.DataFrame({"h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last()})
    daily["datr"] = (daily.h - daily.l).rolling(14, min_periods=5).mean().shift(1)
    daily["pdh"], daily["pdl"], daily["pdc"] = daily.h.shift(1), daily.l.shift(1), daily.c.shift(1)
    for col in ("datr", "pdh", "pdl", "pdc"):
        df[col] = df["trade_date"].map(daily[col])
    # 12-bar (1 hour) rolling channel of PRIOR bars
    df["hh12"] = df["high"].rolling(12).max().shift(1)
    df["ll12"] = df["low"].rolling(12).min().shift(1)
    return df


def prepare(df_1m_or_any: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    bars = resample(df_1m_or_any, bar_minutes)
    bars = add_session_columns(bars)
    return add_indicators(bars)


def synthetic_gold(days: int = 250, seed: int = 7, start_price: float = 4000.0) -> pd.DataFrame:
    """
    Random-walk 1-minute gold-like data with a realistic intraday volatility shape.
    It has NO edge by construction. Used only to prove the engine works and has no
    look-ahead bias: every strategy should lose roughly its costs on this data.
    """
    rng = np.random.default_rng(seed)
    idx = []
    sessions = pd.bdate_range("2025-01-02", periods=days, tz=ET)
    for d in sessions:
        # session: 18:00 previous day -> 17:00 ET (one-hour CME halt 17:00-18:00)
        start = (d - pd.Timedelta(days=1)).replace(hour=18, minute=0)
        idx.append(pd.date_range(start, periods=23 * 60, freq="1min"))
    index = pd.DatetimeIndex(np.concatenate([i.values for i in idx])).tz_localize("UTC").tz_convert(ET)
    tod = index.hour * 60 + index.minute
    # higher vol around London (3 ET) and COMEX/NY open (8:20-11 ET)
    vol = np.full(len(index), 0.35)
    vol[(tod >= 180) & (tod < 300)] = 0.6
    vol[(tod >= 500) & (tod < 660)] = 1.0
    rets = rng.standard_normal(len(index)) * vol
    close = start_price + np.cumsum(rets)
    open_ = np.r_[start_price, close[:-1]]
    noise_h = np.abs(rng.standard_normal(len(index))) * vol * 0.6
    noise_l = np.abs(rng.standard_normal(len(index))) * vol * 0.6
    high = np.maximum(open_, close) + noise_h
    low = np.minimum(open_, close) - noise_l
    vol_ct = rng.integers(20, 400, len(index)).astype(float)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol_ct}, index=index)
    for c in ("open", "high", "low", "close"):  # round to the 0.10 tick
        df[c] = (df[c] / 0.1).round() * 0.1
    return df
