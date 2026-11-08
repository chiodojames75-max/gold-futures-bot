"""
Oro configuration. Every number that affects money lives here, in one place.

Contract: MGC (CME Micro Gold). 10 troy oz per contract.
  tick size  = 0.10 points
  tick value = $1.00          -> $10 per 1.00 point move per contract
GC (full-size gold, 100 oz) has the same price, so GC history can drive MGC tests.

Topstep 50K Trading Combine (as published Oct 2026 - re-check before going live):
  profit target            $3,000
  maximum loss limit (MLL) $2,000, trails end-of-day balance, locks at starting balance
  daily loss limit (DLL)   $1,000 (optional on Topstep's side; Oro enforces its own, tighter)
  consistency              best day must be <= 55% of total profit
  max size                 50 micros
  session                  5:00 PM CT -> 3:10 PM CT (6:00 PM -> 4:10 PM ET); must be flat by 3:10 PM CT
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contract:
    symbol: str = "MGC"
    tick_size: float = 0.10
    tick_value: float = 1.00

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size  # $ per 1.00 point per contract


@dataclass(frozen=True)
class Costs:
    # Round-turn commission+exchange fees per micro contract. Estimate; set to your real Topstep statement number.
    commission_rt: float = 1.50
    # Slippage in ticks applied to market/stop fills (each side). Limit (target) fills get none
    # but require price to trade THROUGH the target by one tick.
    slippage_ticks: int = 1


@dataclass(frozen=True)
class TopstepRules:
    start_balance: float = 50_000.0
    profit_target: float = 3_000.0
    max_loss_limit: float = 2_000.0
    consistency_pct: float = 0.55
    max_contracts: int = 50
    flatten_et: str = "16:05"  # 5 min before Topstep's 4:10 PM ET (3:10 PM CT) cutoff


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 150.0        # $ risked per trade (stop distance x size)
    max_contracts: int = 10              # Oro's own cap, far below Topstep's 50
    daily_loss_stop: float = 400.0       # stop trading for the day after this loss (Topstep DLL is $1,000)
    daily_profit_stop: float = 1_200.0   # stop for the day after this gain (protects the 55% consistency rule)
    max_trades_per_day: int = 3
    mll_buffer: float = 300.0            # refuse new trades if equity is within this much of the MLL
    kill_switch_file: str = "KILL"       # create this file in the run folder and Oro flattens and stops


@dataclass
class OroConfig:
    contract: Contract = field(default_factory=Contract)
    costs: Costs = field(default_factory=Costs)
    topstep: TopstepRules = field(default_factory=TopstepRules)
    risk: RiskConfig = field(default_factory=RiskConfig)
    bar_minutes: int = 5
    timezone: str = "America/New_York"
