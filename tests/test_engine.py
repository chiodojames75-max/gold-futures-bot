"""Engine correctness tests. Run: python -m pytest -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oro import backtest  # noqa: E402
from oro.config import OroConfig, RiskConfig  # noqa: E402
from oro.data import add_indicators, add_session_columns, prepare, synthetic_gold  # noqa: E402
from oro.risk import AccountState, RiskManager  # noqa: E402
from oro.strategies import ComexORB, Signal, Strategy  # noqa: E402

ET = "America/New_York"


def bars(rows, start="2026-03-03 08:00"):
    """rows: list of (open, high, low, close) for consecutive 5-min bars."""
    idx = pd.date_range(pd.Timestamp(start, tz=ET), periods=len(rows), freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return add_indicators(add_session_columns(df))


class OneShot(Strategy):
    """Fires a single known signal on bar `at`."""

    def __init__(self, at_tod, sig):
        super().__init__("oneshot", {}, exit_by="16:05")
        self.at, self.sig = at_tod, sig

    def signal(self, r, day):
        return self.sig if r.tod == self.at and day.get("trades", 0) == 0 else None


def cfg(**risk):
    c = OroConfig()
    c.risk = RiskConfig(**{**RiskConfig().__dict__, **risk})
    return c


def flat(n, px=4000.0):
    return [(px, px + 0.5, px - 0.5, px)] * n


def test_long_hits_target_exact_pnl():
    # signal on 08:00 bar close -> entry at 08:05 open 4000.0 + 0.1 slippage
    rows = flat(1) + [(4000.0, 4000.5, 3999.5, 4000.2), (4000.2, 4006.5, 4000.0, 4006.0)] + flat(3, 4006)
    sig = Signal(+1, stop=3995.0, target=4006.0)
    res = backtest.run(bars(rows), OneShot(8 * 60, sig), cfg(risk_per_trade=1000))
    t = res.trades.iloc[0]
    assert t.exit_reason == "target"
    assert t.entry == pytest.approx(4000.1)
    # size = floor(1000 / (5.1 pts * $10)) = 19 -> capped at max_contracts 10
    assert t["size"] == 10
    assert t.pnl == pytest.approx((4006.0 - 4000.1) * 10 * 10 - 1.5 * 10)


def test_stop_assumed_first_when_both_in_one_bar():
    rows = flat(1) + [(4000.0, 4000.5, 3999.5, 4000.0), (4000.0, 4010.0, 3990.0, 4000.0)] + flat(2)
    sig = Signal(+1, stop=3995.0, target=4005.0)
    res = backtest.run(bars(rows), OneShot(8 * 60, sig), cfg())
    assert res.trades.iloc[0].exit_reason == "stop"
    assert res.trades.iloc[0].exit == pytest.approx(3995.0 - 0.1)


def test_gap_through_stop_fills_at_open():
    rows = flat(1) + [(4000.0, 4000.5, 3999.5, 4000.0), (3990.0, 3991.0, 3988.0, 3989.0)] + flat(2, 3989)
    res = backtest.run(bars(rows), OneShot(8 * 60, Signal(+1, 3995.0, 4010.0)), cfg())
    assert res.trades.iloc[0].exit == pytest.approx(3990.0 - 0.1)


def test_target_needs_trade_through():
    # high touches target exactly but not 1 tick through -> no fill, later time exit
    rows = flat(1) + [(4000.0, 4000.5, 3999.5, 4000.0), (4000.0, 4005.0, 3999.0, 4004.0)] + flat(2, 4004)
    s = OneShot(8 * 60, Signal(+1, 3995.0, 4005.0))
    s.exit_by = "08:20"
    res = backtest.run(bars(rows), s, cfg())
    assert res.trades.iloc[0].exit_reason == "time exit"


def test_no_lookahead_future_changes_dont_change_past_trades():
    raw = synthetic_gold(days=40, seed=3)
    df = prepare(raw, 5)
    a = backtest.run(df, ComexORB(), OroConfig()).trades
    cut = df.index[len(df) // 2]
    df2 = df.copy()
    later = df2.index > cut
    for c in ("open", "high", "low", "close"):
        df2.loc[later, c] = df2.loc[later, c] * 1.05   # rewrite the future
    df2 = add_indicators(df2.drop(columns=["atr", "ema20", "ema50", "vwap"]))
    b = backtest.run(df2, ComexORB(), OroConfig()).trades
    pa = a[a.exit_time < cut].reset_index(drop=True)
    pb = b[b.exit_time < cut].reset_index(drop=True)
    assert len(pa) > 3
    pd.testing.assert_frame_equal(pa, pb)


def test_mll_trails_eod_and_locks():
    c = OroConfig()
    rm = RiskManager(c)
    st = AccountState(balance=50_000, mll=48_000)
    st.balance = 50_500; rm.end_of_day(st); assert st.mll == 48_500
    st.balance = 50_000; rm.end_of_day(st); assert st.mll == 48_500      # never moves down
    st.balance = 53_000; rm.end_of_day(st); assert st.mll == 50_000      # locks at start balance
    st.balance = 55_000; rm.end_of_day(st); assert st.mll == 50_000


def test_risk_blocks_after_daily_loss():
    c = OroConfig()
    rm = RiskManager(c)
    st = AccountState(balance=50_000, mll=48_000)
    rm.after_trade(st, -450)
    ok, why = rm.can_open(st)
    assert not ok and "loss" in why or "locked" in why


def test_noise_has_no_edge_over_many_days():
    """On 3 years of random data, no strategy should pass the verdict gates."""
    from oro.validate import full_report
    df = prepare(synthetic_gold(days=750, seed=11), 5)
    rep = full_report(df, OroConfig())
    for name, r in rep.items():
        assert not r["verdict"].startswith("CANDIDATE"), (name, r["verdict"])
