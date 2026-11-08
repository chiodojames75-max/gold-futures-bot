"""Vector: the technical agent. James's 3-month trend method applied to gold.
Trend-on  = close above the 200-day average AND the 200-day is rising (vs 20 sessions ago).
ROC reset = the 63-session (~3-month) rate of change was above +10% within the last 60 sessions and is now within ±3% of zero.
Backtest (Oct 2021–Oct 2026, Night Shift v2 nights): trend-on nights avg +$36, win 57%, t 2.83 (520 nights);
trend-off nights avg −$7, win 52%, t −0.55 (197 nights). The ROC-reset entries (8 since 2023) averaged +13% over the next 3 months
vs +5% for all days. Used as a desk bias only: Topstep requires flat by 4:10 PM ET, so a 3-month hold is not possible in the Combine."""
import sys, json, os
import numpy as np, pandas as pd

def daily_closes(csv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    csv = csv or next(p for p in (os.path.join(root, "data", "GC_5m_front.csv"), os.path.join(root, "..", "data", "GC_5m_front.csv")) if os.path.exists(p))
    from oro.data import load_csv, add_session_columns
    df = add_session_columns(load_csv(csv))
    dc = df.groupby("trade_date").close.last(); dc.index = pd.to_datetime(pd.Index(dc.index).astype(str))
    hi = df.groupby("trade_date").high.max(); lo = df.groupby("trade_date").low.min(); hi.index = lo.index = dc.index
    extra = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "desk", "gold_closes.csv")
    if os.path.exists(extra):   # daily settlements the desk appends from public quotes (date,close[,high,low])
        x = pd.read_csv(extra, parse_dates=["date"]).set_index("date").sort_index(); x = x[x.index > dc.index[-1]]
        dc = pd.concat([dc, x["close"]]); hi = pd.concat([hi, x.get("high", x["close"])]); lo = pd.concat([lo, x.get("low", x["close"])])
    return dc, hi, lo

def read(dc, hi, lo):
    sma = dc.rolling(200).mean(); rising = sma.iloc[-1] > sma.iloc[-21]; above = dc.iloc[-1] > sma.iloc[-1]
    roc = dc / dc.shift(63) - 1; was_high = roc.iloc[-60:].max() > 0.10; reset = abs(roc.iloc[-1]) < 0.03
    rng = (hi - lo).rolling(20).mean().iloc[-1]
    return {"date": str(dc.index[-1].date()), "close": round(float(dc.iloc[-1]), 1), "sma200": round(float(sma.iloc[-1]), 1),
            "sma200_rising": bool(rising), "above_sma200": bool(above), "roc63_pct": round(float(roc.iloc[-1]) * 100, 1),
            "roc63_max_60d_pct": round(float(roc.iloc[-60:].max()) * 100, 1), "roc_reset": bool(was_high and reset),
            "trend_on": bool(above and rising), "avg_daily_range": round(float(rng), 1),
            "tonight_stop_dist": round(float(rng) * 0.5, 1), "tonight_target_dist": round(float(rng), 1),
            "levels": {"prior_high": round(float(hi.iloc[-1]), 1), "prior_low": round(float(lo.iloc[-1]), 1),
                       "high_20d": round(float(hi.iloc[-20:].max()), 1), "low_20d": round(float(lo.iloc[-20:].min()), 1)}}

if __name__ == "__main__":
    print(json.dumps(read(*daily_closes()), indent=1))
