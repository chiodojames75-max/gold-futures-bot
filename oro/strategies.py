"""
Candidate rule sets. None of these is a known edge. They are hypotheses to test.

Contract for every strategy:
  - precompute(df) adds columns using ONLY data available at each bar's close.
  - signal(r, day) is called with bar i (r) at its CLOSE and may return a Signal.
    The engine fills it at the OPEN of bar i+1. The strategy never sees bar i+1.
  - Parameters are fixed while trading. Changing them is an offline, human-approved step.

Times are US/Eastern minutes-since-midnight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _add(s: str, minutes: int) -> str:
    t = hm(s) + minutes
    return f"{t // 60:02d}:{t % 60:02d}"


@dataclass
class Signal:
    side: int            # +1 long, -1 short
    stop: float          # protective stop price
    target: float        # profit target price
    reason: str = ""


@dataclass
class Strategy:
    name: str = "base"
    params: dict = field(default_factory=dict)
    exit_by: str = "16:05"   # strategy's own time stop (ET); engine also enforces Topstep flatten

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def signal(self, r, day: dict) -> Optional[Signal]:
        raise NotImplementedError


def _sm(tod: int) -> int:
    """ET clock minute -> minutes since the 6:00 PM ET session open (chronological within a trade day)."""
    return (tod - 18 * 60) % 1440


def _window_range(df: pd.DataFrame, start: int, end: int, prefix: str) -> pd.DataFrame:
    """High/low of bars with start <= clock < end, made visible only on bars chronologically
    AFTER the window closes in the same trade day. Works for windows that cross midnight."""
    s, e = _sm(start), _sm(end)
    in_win = (df["sm"] >= s) & (df["sm"] < e)
    after = df["sm"] >= e
    hi = df["high"].where(in_win).groupby(df["trade_date"]).transform("max")
    lo = df["low"].where(in_win).groupby(df["trade_date"]).transform("min")
    df[f"{prefix}_high"] = hi.where(after)
    df[f"{prefix}_low"] = lo.where(after)
    return df


class ComexORB(Strategy):
    """
    Hypothesis: the COMEX gold open (8:20 ET) sets a range; a 5-min close outside it
    during the morning carries through.
      Range:  08:20-08:50 ET
      Entry:  first 5-min close beyond the range, 08:50-10:30 ET, one trade per day
      Stop:   middle of the range
      Target: entry +/- R * reward_r
      Filter: skip if range is wider than max_range_atr x ATR (too much risk, usually news)
      Exit:   11:30 ET if neither stop nor target hit
    """

    def __init__(self, reward_r: float = 2.0, max_range_atr: float = 4.0, open_at: str = "08:20"):
        super().__init__("comex_orb", {"reward_r": reward_r, "max_range_atr": max_range_atr}, exit_by=_add(open_at, 190))
        self.o = hm(open_at)

    def precompute(self, df):
        return _window_range(df.copy(), self.o, self.o + 30, "or")

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or not (self.o + 30 <= r.tod < self.o + 130):
            return None
        hi, lo, atr = r.or_high, r.or_low, r.atr
        if np.isnan(hi) or np.isnan(atr) or (hi - lo) > self.params["max_range_atr"] * atr:
            return None
        mid = (hi + lo) / 2
        R = self.params["reward_r"]
        if r.close > hi:
            return Signal(+1, mid, r.close + R * (r.close - mid), "close above opening range")
        if r.close < lo:
            return Signal(-1, mid, r.close - R * (mid - r.close), "close below opening range")
        return None


class VwapPullback(Strategy):
    """
    Hypothesis: in a trending NY session, pullbacks to the 20 EMA that hold get continuation.
      Trend:  close on the trend side of session VWAP and EMA20 vs EMA50 agrees
      Entry:  bar's low touches EMA20 and closes back above it (mirror for shorts), 09:30-14:00 ET
      Stop:   bar extreme -/+ 0.25 ATR
      Target: reward_r x risk
      Max 2 trades/day, exit 15:30 ET
    """

    def __init__(self, reward_r: float = 2.0, max_trades: int = 2):
        super().__init__("vwap_pullback", {"reward_r": reward_r, "max_trades": max_trades}, exit_by="15:30")

    def signal(self, r, day):
        if day.get("trades", 0) >= self.params["max_trades"] or not (hm("09:30") <= r.tod < hm("14:00")):
            return None
        if np.isnan(r.atr) or np.isnan(r.vwap):
            return None
        R = self.params["reward_r"]
        if r.close > r.vwap and r.ema20 > r.ema50 and r.low <= r.ema20 < r.close:
            stop = r.low - 0.25 * r.atr
            return Signal(+1, stop, r.close + R * (r.close - stop), "long pullback to EMA20 in uptrend")
        if r.close < r.vwap and r.ema20 < r.ema50 and r.high >= r.ema20 > r.close:
            stop = r.high + 0.25 * r.atr
            return Signal(-1, stop, r.close - R * (stop - r.close), "short pullback to EMA20 in downtrend")
        return None


class LondonBreakout(Strategy):
    """
    Hypothesis: London's open breaks the quiet Asian range and the move persists.
      Range:  19:00-02:00 ET (Asia)
      Entry:  first 5-min close beyond the range, 02:00-05:00 ET, one trade per day
      Stop:   middle of the Asian range;  Target: reward_r x risk;  Exit 08:00 ET
    """

    def __init__(self, reward_r: float = 1.5, max_range_atr: float = 6.0):
        super().__init__("london_breakout", {"reward_r": reward_r, "max_range_atr": max_range_atr}, exit_by="08:00")

    def precompute(self, df):
        return _window_range(df.copy(), hm("19:00"), hm("02:00"), "asia")

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or not (hm("02:00") <= r.tod < hm("05:00")):
            return None
        hi, lo, atr = r.asia_high, r.asia_low, r.atr
        if np.isnan(hi) or np.isnan(atr) or (hi - lo) > self.params["max_range_atr"] * atr:
            return None
        mid = (hi + lo) / 2
        R = self.params["reward_r"]
        if r.close > hi:
            return Signal(+1, mid, r.close + R * (r.close - mid), "London break above Asia range")
        if r.close < lo:
            return Signal(-1, mid, r.close - R * (mid - r.close), "London break below Asia range")
        return None


def _ok(*v):
    return all(x == x and x is not None for x in v)   # no NaNs


class OrbFade(Strategy):
    """Failed opening-range breakout (range = first 30 min after the session open):
    a close outside that closes back inside -> trade toward the other side."""

    def __init__(self, stop_frac: float = 0.5, tgt_frac: float = 1.0, open_at: str = "08:20"):
        super().__init__("orb_fade", {"stop_frac": stop_frac, "tgt_frac": tgt_frac}, exit_by=_add(open_at, 220))
        self.o = hm(open_at)

    def precompute(self, df):
        return _window_range(df.copy(), self.o, self.o + 30, "or")

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or not (self.o + 30 <= r.tod < self.o + 160) or not _ok(r.or_high, r.or_low):
            return None
        hi, lo = r.or_high, r.or_low
        rng = hi - lo
        if rng <= 0:
            return None
        if r.close > hi:
            day["was_above"] = True
        elif r.close < lo:
            day["was_below"] = True
        elif day.get("was_above"):
            return Signal(-1, hi + self.params["stop_frac"] * rng, r.close - self.params["tgt_frac"] * rng, "failed upside ORB")
        elif day.get("was_below"):
            return Signal(+1, lo - self.params["stop_frac"] * rng, r.close + self.params["tgt_frac"] * rng, "failed downside ORB")
        return None


class OvernightMove(Strategy):
    """Large overnight move into the COMEX open. mode=+1 follows it, -1 fades it.
    Decided on the 08:20 bar close. Stop/target in daily-ATR units. Exit 12:00."""

    def __init__(self, mode: int = -1, k: float = 0.5, stop_d: float = 0.3, reward_r: float = 1.5, open_at: str = "08:20"):
        super().__init__("overnight_" + ("fade" if mode < 0 else "follow"),
                         {"mode": mode, "k": k, "stop_d": stop_d, "reward_r": reward_r}, exit_by=_add(open_at, 220))
        self.o = hm(open_at)

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or r.tod != self.o or not _ok(r.pdc, r.datr):
            return None
        move = r.close - r.pdc
        if abs(move) < self.params["k"] * r.datr:
            return None
        side = int(np.sign(move)) * self.params["mode"]
        risk = self.params["stop_d"] * r.datr
        return Signal(side, r.close - side * risk, r.close + side * risk * self.params["reward_r"],
                      f"overnight move {move:+.1f}")


class ReleaseBar(Strategy):
    """Economic-release bar (08:30 or 10:00 ET): if that 5-min bar is large, follow it.
    Stop at the bar's far end, target reward_r x risk, exit after 2 hours."""

    def __init__(self, at: str = "08:30", k: float = 2.0, reward_r: float = 1.5):
        end = f"{(hm(at) + 120) // 60:02d}:{(hm(at) + 120) % 60:02d}"
        super().__init__(f"release_{at.replace(':', '')}", {"at": at, "k": k, "reward_r": reward_r}, exit_by=end)

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or r.tod != hm(self.params["at"]) or not _ok(r.atr):
            return None
        rng = r.high - r.low
        if rng < self.params["k"] * r.atr or r.close == r.open:
            return None
        side = 1 if r.close > r.open else -1
        stop = r.low if side > 0 else r.high
        return Signal(side, stop, r.close + side * abs(r.close - stop) * self.params["reward_r"], "release bar")


class VwapReversion(Strategy):
    """Midday stretch from VWAP snaps back. 10:30-14:00 ET, fade closes > k ATR from VWAP,
    target VWAP, stop m ATR beyond entry."""

    def __init__(self, k: float = 3.0, m: float = 1.5):
        super().__init__("vwap_reversion", {"k": k, "m": m}, exit_by="15:00")

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or not (hm("10:30") <= r.tod < hm("14:00")) or not _ok(r.atr, r.vwap):
            return None
        d = r.close - r.vwap
        if abs(d) < self.params["k"] * r.atr:
            return None
        side = -1 if d > 0 else 1
        return Signal(side, r.close - side * self.params["m"] * r.atr, r.vwap, "stretched from VWAP")


class PriorDayLevel(Strategy):
    """Prior-day high/low during NY hours (08:20-12:00). mode=+1 breakout follow,
    -1 fade a poke-through that closes back inside."""

    def __init__(self, mode: int = 1, stop_d: float = 0.25, reward_r: float = 2.0, open_at: str = "08:20"):
        super().__init__("pdhl_" + ("break" if mode > 0 else "fade"),
                         {"mode": mode, "stop_d": stop_d, "reward_r": reward_r}, exit_by="14:00")
        self.o = hm(open_at)

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or not (self.o <= r.tod < hm("12:00")) or not _ok(r.pdh, r.pdl, r.datr):
            return None
        risk = self.params["stop_d"] * r.datr
        R = self.params["reward_r"]
        if self.params["mode"] > 0:
            if r.close > r.pdh:
                return Signal(+1, r.close - risk, r.close + R * risk, "break above prior-day high")
            if r.close < r.pdl:
                return Signal(-1, r.close + risk, r.close - R * risk, "break below prior-day low")
        else:
            if r.high > r.pdh and r.close < r.pdh:
                return Signal(-1, r.high + 0.1 * risk, r.close - R * (r.high + 0.1 * risk - r.close), "rejected prior-day high")
            if r.low < r.pdl and r.close > r.pdl:
                return Signal(+1, r.low - 0.1 * risk, r.close + R * (r.close - r.low + 0.1 * risk), "rejected prior-day low")
        return None


class ChannelTrend(Strategy):
    """1-hour channel breakout in the direction of the EMA trend, 09:30-14:00 ET."""

    def __init__(self, m: float = 1.5, reward_r: float = 2.0):
        super().__init__("channel_trend", {"m": m, "reward_r": reward_r}, exit_by="15:30")

    def signal(self, r, day):
        if day.get("trades", 0) >= 2 or not (hm("09:30") <= r.tod < hm("14:00")) or not _ok(r.atr, r.hh12, r.ll12):
            return None
        risk = self.params["m"] * r.atr
        R = self.params["reward_r"]
        if r.close > r.hh12 and r.ema20 > r.ema50:
            return Signal(+1, r.close - risk, r.close + R * risk, "1h breakout up, trend up")
        if r.close < r.ll12 and r.ema20 < r.ema50:
            return Signal(-1, r.close + risk, r.close - R * risk, "1h breakout down, trend down")
        return None


class AsiaFade(Strategy):
    """London fails: price closes outside the Asian range (02:00-05:00 ET) then back inside -> fade
    toward the other side of the range."""

    def __init__(self, stop_frac: float = 0.3, tgt_frac: float = 0.5):
        super().__init__("asia_fade", {"stop_frac": stop_frac, "tgt_frac": tgt_frac}, exit_by="08:00")

    def precompute(self, df):
        return _window_range(df.copy(), hm("19:00"), hm("02:00"), "asia")

    def signal(self, r, day):
        if day.get("trades", 0) >= 1 or not (hm("02:00") <= r.tod < hm("05:00")) or not _ok(r.asia_high, r.asia_low):
            return None
        hi, lo = r.asia_high, r.asia_low
        rng = hi - lo
        if rng <= 0:
            return None
        if r.close > hi:
            day["was_above"] = True
        elif r.close < lo:
            day["was_below"] = True
        elif day.get("was_above"):
            return Signal(-1, hi + self.params["stop_frac"] * rng, r.close - self.params["tgt_frac"] * rng, "failed London up-break")
        elif day.get("was_below"):
            return Signal(+1, lo - self.params["stop_frac"] * rng, r.close + self.params["tgt_frac"] * rng, "failed London down-break")
        return None


REGISTRY = {
    "comex_orb": ComexORB,
    "vwap_pullback": VwapPullback,
    "london_breakout": LondonBreakout,
    "orb_fade": OrbFade,
    "overnight_fade": lambda **p: OvernightMove(mode=-1, **p),
    "overnight_follow": lambda **p: OvernightMove(mode=1, **p),
    "release_0830": lambda **p: ReleaseBar(at="08:30", **p),
    "release_1000": lambda **p: ReleaseBar(at="10:00", **p),
    "vwap_reversion": VwapReversion,
    "pdhl_break": lambda **p: PriorDayLevel(mode=1, **p),
    "pdhl_fade": lambda **p: PriorDayLevel(mode=-1, **p),
    "channel_trend": ChannelTrend,
    "asia_fade": AsiaFade,
}


class Inverse(Strategy):
    """The same rule set traded the opposite way: side flipped, the old target becomes the stop and the old stop
    becomes the target. Used to test 'flip the script'."""

    def __init__(self, inner: Strategy):
        super().__init__("inv_" + inner.name, inner.params, exit_by=inner.exit_by)
        self.inner = inner

    def precompute(self, df):
        return self.inner.precompute(df)

    def signal(self, r, day):
        s = self.inner.signal(r, day)
        if s is None:
            return None
        return Signal(-s.side, s.target, s.stop, "flipped: " + s.reason)


class NightShift(Strategy):
    """
    Overnight drift (documented: Boyarchenko, Larsen & Whelan, NY Fed 2023).
      Filter: yesterday's close above its 100-day average (trend on)
      Entry:  long at the start of the Topstep day (signal on the 6:00 PM ET bar, filled on the next bar)
      Exit:   9:30 AM ET, or the protective stop at stop_d x the average daily range below entry
    """

    def __init__(self, stop_d: float = 1.0, sma: int = 100, exit_at: str = "09:30", mode: str = "long", daily_closes=None, reentries: int = 0, tgt_d: float = 20.0, min_vol: float = 0.0):
        super().__init__("night_shift", {"stop_d": stop_d, "sma": sma, "exit_at": exit_at, "mode": mode, "reentries": reentries, "tgt_d": tgt_d, "min_vol": min_vol}, exit_by=exit_at)
        self.daily_closes = daily_closes          # optional pd.Series of daily closes (live engine), indexed by trade date
        self.daily_ranges = None                  # optional pd.Series of daily high-low (live engine) for the quiet filter

    def precompute(self, df):
        df = df.copy()
        dc = df.groupby("trade_date")["close"].last()
        if self.daily_closes is not None:          # longer history than the 5-minute bars cover
            dc = pd.concat([self.daily_closes[~self.daily_closes.index.isin(dc.index)], dc]).sort_index()
        if self.params["sma"] <= 0:          # v2/v3: no trend filter, the drift is traded every night
            df["trend_up"] = True
            if self.params.get("min_vol", 0) > 0:   # v3: skip dead-quiet markets (5-day avg range < min_vol x its 100-day avg)
                g = df.groupby("trade_date"); rng = g["high"].max() - g["low"].min()
                if self.daily_ranges is not None:
                    rng = pd.concat([self.daily_ranges[~self.daily_ranges.index.isin(rng.index)], rng]).sort_index()
                ok = (rng.rolling(5).mean() / rng.rolling(100).mean() > self.params["min_vol"]).shift(1)
                df["trend_up"] = df["trade_date"].map(ok).fillna(False).astype(bool)
            return df
        up = (dc > dc.rolling(self.params["sma"]).mean()).shift(1)
        df["trend_up"] = df["trade_date"].map(up).fillna(False).astype(bool)
        return df

    def signal(self, r, day):
        if not _ok(r.datr):
            return None
        n = day.get("trades", 0)
        if n >= 1:
            # re-entry: only after the first trade was stopped out, when price closes back above the first entry, before 4 AM ET
            first = day.get("first_entry")
            if (n > self.params.get("reentries", 0) or first is None or not r.trend_up or self.params["mode"] != "long"
                    or not (r.tod >= 18 * 60 or r.tod < 4 * 60) or r.close <= first):
                return None
            d = self.params["stop_d"] * r.datr
            return Signal(+1, r.close - d, r.close + 20 * r.datr, "re-entry: back above the first entry")
        if r.tod != 18 * 60:
            return None
        day["first_entry"] = r.close
        mode, d = self.params["mode"], self.params["stop_d"] * r.datr
        if r.trend_up and mode in ("long", "both"):
            return Signal(+1, r.close - d, r.close + self.params.get("tgt_d", 20.0) * r.datr, "overnight long")
        if (not r.trend_up) and mode in ("short", "both"):
            return Signal(-1, r.close + d, r.close - self.params.get("tgt_d", 20.0) * r.datr, "overnight short")
        return None


REGISTRY["night_shift"] = NightShift
