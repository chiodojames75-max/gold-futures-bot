"""
Bar-by-bar backtester with a Topstep 50K account simulation.

Fill model (deliberately pessimistic):
  - Signal at bar i close -> market entry at bar i+1 OPEN, plus slippage.
  - Stop: if a bar's range reaches the stop, it fills at the stop (or the open, if the bar
    gapped through it), plus slippage.
  - Target: fills only if price trades one tick THROUGH the target. No slippage.
  - If stop and target are both inside the same bar, the STOP is assumed to have hit first.
  - Time exits fill at the bar's open, plus slippage.
  - Round-turn commission charged per contract.
  - MLL breach is checked against the worst open-trade price inside each bar.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .config import OroConfig
from .risk import AccountState, RiskManager
from .strategies import Strategy, Signal, hm


@dataclass
class Trade:
    strategy: str
    trade_date: object
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    size: int
    entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str
    pnl: float           # $ net of costs
    r_multiple: float    # pnl / initial $ risk
    reason: str


@dataclass
class Result:
    trades: pd.DataFrame
    daily: pd.Series           # daily net P&L indexed by trade_date
    equity: pd.Series          # end-of-day balance
    account_failed: bool
    failed_on: Optional[object]
    combine_passed: bool
    passed_on: Optional[object]


def _sm(tod: int) -> int:
    return (tod - 18 * 60) % 1440


def run(df: pd.DataFrame, strat: Strategy, cfg: OroConfig, stop_on_fail: bool = False,
        enforce_mll: bool = True) -> Result:
    """enforce_mll=False measures the strategy itself over the whole history (research mode);
    the Topstep account outcome is then estimated separately by Monte Carlo. The first
    would-be breach is still reported in failed_on."""
    df = strat.precompute(df).reset_index().rename(columns={"index": "ts"})
    if "ts" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ts"})
    rm = RiskManager(cfg)
    c, costs, ts_rules = cfg.contract, cfg.costs, cfg.topstep
    slip = costs.slippage_ticks * c.tick_size
    pv = c.point_value

    st = AccountState(balance=ts_rules.start_balance, mll=ts_rules.start_balance - ts_rules.max_loss_limit)
    flatten_sm = _sm(hm(ts_rules.flatten_et))
    strat_exit_sm = _sm(hm(strat.exit_by))

    trades: list[Trade] = []
    events: list[dict] = []   # signals that were NOT executed, and why
    daily: dict = {}
    eod: dict = {}
    day_ctx: dict = {}
    pos = None          # dict for the open trade
    pending: Optional[Signal] = None
    failed_on = passed_on = None
    cur_day = None

    O, H, L, C = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    SM = df["sm"].to_numpy()
    TD = df["trade_date"].to_numpy()
    TS = df["ts"]
    ROWS = list(df.itertuples(index=False))

    def close_pos(i, price, why):
        nonlocal pos
        gross = (price - pos["entry"]) * pos["side"] * pos["size"] * pv
        net = gross - costs.commission_rt * pos["size"]
        risk_usd = abs(pos["entry"] - pos["stop"]) * pos["size"] * pv
        trades.append(Trade(strat.name, pos["day"], pos["t"], TS.iloc[i], pos["side"], pos["size"],
                            pos["entry"], pos["stop"], pos["target"], price, why, net,
                            net / risk_usd if risk_usd else 0.0, pos["reason"]))
        rm.after_trade(st, net)
        daily[pos["day"]] = daily.get(pos["day"], 0.0) + net
        pos = None

    for i in range(len(df)):
        day = TD[i]
        if day != cur_day:
            if cur_day is not None:
                if pos is not None:  # safety: should never happen because of flatten
                    close_pos(i - 1, C[i - 1], "session end")
                rm.end_of_day(st)
                eod[cur_day] = st.balance
                daily.setdefault(cur_day, 0.0)
                # combine check at end of day
                profit = st.balance - ts_rules.start_balance
                best = max(daily.values()) if daily else 0.0
                if passed_on is None and not st.failed and profit >= ts_rules.profit_target \
                        and best <= ts_rules.consistency_pct * profit:
                    passed_on = cur_day
            cur_day = day
            day_ctx = {"trades": 0}
            pending = None

        if st.failed:
            if stop_on_fail:
                break
            continue

        # 1) fill pending entry at this bar's open
        if pending is not None and pos is None:
            ok, why = rm.can_open(st, check_mll=enforce_mll)
            in_time = SM[i] < min(flatten_sm, strat_exit_sm)
            side = pending.side
            entry = O[i] + side * slip
            stop_ok = (entry - pending.stop) * side > 0      # stop still on the correct side
            size = rm.size_for(entry, pending.stop) if stop_ok else 0
            if ok and in_time and size > 0:
                pos = {"side": side, "size": size, "entry": entry, "stop": pending.stop,
                       "target": pending.target, "t": TS.iloc[i], "day": day, "reason": pending.reason}
                day_ctx["trades"] += 1
            else:
                events.append({"time": TS.iloc[i], "side": side, "price": float(O[i]), "reason": pending.reason,
                               "blocked_by": why if not ok else ("past trading window" if not in_time
                               else "stop too wide for the $ risk limit")})
            pending = None

        # 2) manage open position inside this bar
        if pos is not None:
            s = pos["side"]
            # time exits first: Topstep flatten or strategy exit window
            if SM[i] >= flatten_sm or SM[i] >= strat_exit_sm:
                close_pos(i, O[i] - s * slip, "time exit")
            else:
                worst = L[i] if s > 0 else H[i]
                open_pnl_worst = (worst - pos["entry"]) * s * pos["size"] * pv
                if enforce_mll and st.balance + open_pnl_worst <= st.mll:
                    # account would be blown intrabar unless the stop protected it first
                    stop_px = pos["stop"]
                    stop_pnl = (stop_px - pos["entry"]) * s * pos["size"] * pv
                    if st.balance + stop_pnl > st.mll:
                        pass  # stop fills before MLL; handled below
                    else:
                        close_pos(i, worst, "MLL breach")
                        st.failed = True
                        failed_on = day
                        continue
                hit_stop = (L[i] <= pos["stop"]) if s > 0 else (H[i] >= pos["stop"])
                hit_tgt = (H[i] >= pos["target"] + c.tick_size) if s > 0 else (L[i] <= pos["target"] - c.tick_size)
                if hit_stop:
                    gap_through = (O[i] <= pos["stop"]) if s > 0 else (O[i] >= pos["stop"])
                    px = (O[i] if gap_through else pos["stop"]) - s * slip
                    close_pos(i, px, "stop")
                elif hit_tgt:
                    close_pos(i, pos["target"], "target")
                if st.balance <= st.mll and failed_on is None:
                    failed_on = day
                    st.failed = enforce_mll

        # 3) new signal at this bar's close (filled next bar)
        if pos is None and not st.failed and SM[i] < min(flatten_sm, strat_exit_sm):
            sig = strat.signal(ROWS[i], day_ctx)
            if sig is not None:
                pending = sig

    if cur_day is not None:
        if pos is not None:
            close_pos(len(df) - 1, C[-1], "data end")
        daily.setdefault(cur_day, 0.0)
        eod[cur_day] = st.balance

    tdf = pd.DataFrame([asdict(t) for t in trades])
    res = Result(tdf, pd.Series(daily, dtype=float).sort_index(), pd.Series(eod, dtype=float).sort_index(),
                 st.failed, failed_on, passed_on is not None, passed_on)
    res.blocked = pd.DataFrame(events)
    return res


def stats(res: Result, cfg: OroConfig) -> dict:
    t = res.trades
    if t.empty:
        return {"trades": 0}
    wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
    eq = res.daily.cumsum()
    dd = (eq - eq.cummax()).min()
    active = res.daily[res.daily != 0]
    sharpe = (active.mean() / active.std() * np.sqrt(252)) if len(active) > 2 and active.std() > 0 else float("nan")
    gross_win, gross_loss = wins.pnl.sum(), -losses.pnl.sum()
    return {
        "trades": len(t),
        "win_rate": round(len(wins) / len(t), 3),
        "avg_R": round(t.r_multiple.mean(), 3),
        "expectancy_$": round(t.pnl.mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "net_$": round(t.pnl.sum(), 2),
        "max_drawdown_$": round(dd, 2),
        "sharpe_daily_ann": round(sharpe, 2),
        "best_day_$": round(res.daily.max(), 2),
        "worst_day_$": round(res.daily.min(), 2),
        "exit_mix": t.exit_reason.value_counts().to_dict(),
        "account_failed": res.account_failed,
        "combine_passed": res.combine_passed,
    }
