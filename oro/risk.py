"""
Risk layer. Shared by the backtester and the live bot so both obey identical rules.
It can veto any trade. It never makes a trade.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .config import OroConfig


@dataclass
class AccountState:
    balance: float            # realized balance
    mll: float                # current Topstep maximum-loss-limit level
    day_pnl: float = 0.0
    day_trades: int = 0
    locked_for_day: bool = False
    failed: bool = False      # MLL breached: account is dead


class RiskManager:
    def __init__(self, cfg: OroConfig):
        self.cfg = cfg

    def size_for(self, entry: float, stop: float) -> int:
        """Contracts so that (stop distance x size) <= risk_per_trade. 0 = skip the trade."""
        dist = abs(entry - stop)
        if dist <= 0:
            return 0
        per_contract = dist * self.cfg.contract.point_value
        n = math.floor(self.cfg.risk.risk_per_trade / per_contract)
        return max(0, min(n, self.cfg.risk.max_contracts, self.cfg.topstep.max_contracts))

    def can_open(self, st: AccountState, run_dir: str = ".", check_mll: bool = True) -> tuple[bool, str]:
        r = self.cfg.risk
        if st.failed:
            return False, "account failed (MLL breached)"
        if os.path.exists(os.path.join(run_dir, r.kill_switch_file)):
            return False, "kill switch file present"
        if st.locked_for_day:
            return False, "locked for the day"
        if st.day_trades >= r.max_trades_per_day:
            return False, "max trades for the day"
        if st.day_pnl <= -r.daily_loss_stop:
            return False, "daily loss stop hit"
        if st.day_pnl >= r.daily_profit_stop:
            return False, "daily profit stop hit (consistency protection)"
        if check_mll and st.balance - st.mll <= r.mll_buffer + r.risk_per_trade:
            return False, "too close to the maximum loss limit"
        return True, "ok"

    def after_trade(self, st: AccountState, pnl: float) -> None:
        st.balance += pnl
        st.day_pnl += pnl
        st.day_trades += 1
        r = self.cfg.risk
        if st.day_pnl <= -r.daily_loss_stop or st.day_pnl >= r.daily_profit_stop:
            st.locked_for_day = True

    def end_of_day(self, st: AccountState) -> None:
        """Topstep MLL trails the end-of-day balance and locks at the starting balance."""
        t = self.cfg.topstep
        new_mll = min(st.balance - t.max_loss_limit, t.start_balance)
        st.mll = max(st.mll, new_mll)
        st.day_pnl = 0.0
        st.day_trades = 0
        st.locked_for_day = False
