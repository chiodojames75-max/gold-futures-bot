"""
Honest testing. This is where most strategies should die.

1. Split the history: tune on the first part (in-sample), judge ONLY on the untouched
   last part (out-of-sample). A strategy that only works in-sample is curve-fit.
2. Parameter neighbourhood: a real edge works across nearby settings, not one magic value.
3. Monte Carlo the out-of-sample daily P&L into many simulated Topstep Combines to estimate
   pass vs. fail odds and how long passing takes.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import backtest
from .config import OroConfig
from .strategies import REGISTRY


def split(df: pd.DataFrame, oos_frac: float = 0.3):
    days = pd.Index(sorted(df["trade_date"].unique()))
    cut = days[int(len(days) * (1 - oos_frac))]
    return df[df["trade_date"] < cut], df[df["trade_date"] >= cut]


def grid(name: str, df_is: pd.DataFrame, cfg: OroConfig, space: dict) -> pd.DataFrame:
    rows = []
    keys = list(space)
    for combo in itertools.product(*space.values()):
        p = dict(zip(keys, combo))
        res = backtest.run(df_is, REGISTRY[name](**p), cfg, enforce_mll=False)
        s = backtest.stats(res, cfg)
        rows.append({**p, **{k: s.get(k) for k in ("trades", "expectancy_$", "profit_factor", "net_$", "max_drawdown_$")}})
    return pd.DataFrame(rows).sort_values("net_$", ascending=False)


def monte_carlo_combine(daily: pd.Series, cfg: OroConfig, n: int = 5000, max_days: int = 60, seed: int = 1) -> dict:
    """Resample real trading days (with replacement) into simulated Combines."""
    d = daily.to_numpy()
    if len(d) < 10:
        return {"note": "not enough days"}
    t = cfg.topstep
    rng = np.random.default_rng(seed)
    passed = failed = 0
    days_to_pass = []
    for _ in range(n):
        bal, mll, best = t.start_balance, t.start_balance - t.max_loss_limit, -np.inf
        for k in range(1, max_days + 1):
            pnl = d[rng.integers(len(d))]
            bal += pnl
            best = max(best, pnl)
            if bal <= mll:
                failed += 1
                break
            mll = max(mll, min(bal - t.max_loss_limit, t.start_balance))
            profit = bal - t.start_balance
            if profit >= t.profit_target and best <= t.consistency_pct * profit:
                passed += 1
                days_to_pass.append(k)
                break
    return {
        "p_pass": round(passed / n, 3),
        "p_fail": round(failed / n, 3),
        "p_unresolved_after_%d_days" % max_days: round(1 - (passed + failed) / n, 3),
        "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else None,
    }


DEFAULT_SPACES = {
    "comex_orb": {"reward_r": [1.0, 1.5, 2.0, 3.0], "max_range_atr": [2.0, 3.0, 4.0, 6.0]},
    "vwap_pullback": {"reward_r": [1.0, 1.5, 2.0, 3.0], "max_trades": [1, 2]},
    "london_breakout": {"reward_r": [1.0, 1.5, 2.0], "max_range_atr": [4.0, 6.0, 8.0]},
}


def full_report(df: pd.DataFrame, cfg: OroConfig, oos_frac: float = 0.3) -> dict:
    df_is, df_oos = split(df, oos_frac)
    report = {}
    # only strategies with a parameter grid can be tuned and validated here
    for name in (n for n in REGISTRY if n in DEFAULT_SPACES):
        g = grid(name, df_is, cfg, DEFAULT_SPACES[name])
        best = g.iloc[0]
        params = {k: (int(best[k]) if float(best[k]).is_integer() and k == "max_trades" else float(best[k]))
                  for k in DEFAULT_SPACES[name]}
        # how many neighbouring settings were also profitable in-sample?
        robust = float((g["net_$"] > 0).mean())
        oos = backtest.run(df_oos, REGISTRY[name](**params), cfg, enforce_mll=False)
        oos_stats = backtest.stats(oos, cfg)
        report[name] = {
            "chosen_params(in-sample best)": params,
            "in_sample_best": best[["trades", "expectancy_$", "profit_factor", "net_$"]].to_dict(),
            "share_of_grid_profitable_in_sample": round(robust, 2),
            "out_of_sample": oos_stats,
            "combine_monte_carlo(out_of_sample days)": monte_carlo_combine(oos.daily, cfg),
            "expectancy_90pct_CI_$": (ci := expectancy_ci(oos.trades)),
            "verdict": verdict(oos_stats, robust, ci),
            "_oos_trades": oos.trades,
        }
    return report


def expectancy_ci(trades: pd.DataFrame, n: int = 5000, seed: int = 2) -> tuple:
    """Bootstrap 90% interval for average $ per trade. If the low end is below zero,
    the result could easily be luck."""
    if trades.empty:
        return (None, None)
    p = trades["pnl"].to_numpy()
    rng = np.random.default_rng(seed)
    means = p[rng.integers(0, len(p), (n, len(p)))].mean(axis=1)
    return (round(float(np.percentile(means, 5)), 2), round(float(np.percentile(means, 95)), 2))


def verdict(oos: dict, robust: float, ci: tuple) -> str:
    """Three gates, all required: enough trades, the edge isn't luck, and it isn't one magic setting."""
    if oos.get("trades", 0) < 100:
        return "INCONCLUSIVE - fewer than 100 out-of-sample trades; get more data"
    if ci[0] is None or ci[0] <= 0:
        return "REJECT - out-of-sample result is indistinguishable from luck"
    if robust < 0.5:
        return "REJECT - only a narrow set of settings worked in-sample (curve-fit)"
    if oos.get("profit_factor", 0) >= 1.3:
        return "CANDIDATE - passed all gates; next step is 4+ weeks of paper trading"
    return "WEAK - real but thin; real-world slippage may erase it"
