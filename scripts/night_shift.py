"""Night Shift through the full engine (pessimistic fills, costs, stops) on gold and Nasdaq."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oro.config import OroConfig, RiskConfig
from oro.data import load_csv, add_session_columns, add_indicators
from oro.markets import MARKETS
from oro import backtest
from oro.strategies import NightShift
from oro.validate import monte_carlo_combine

def run(mk, stop_d, size):
    df = add_indicators(add_session_columns(load_csv(f"data/{mk}_5m_front.csv")))
    c = OroConfig(); c.contract = MARKETS[mk]["contract"]
    c.risk = RiskConfig(risk_per_trade=1e9, max_contracts=size, daily_loss_stop=1e9, daily_profit_stop=1e9, mll_buffer=0)
    r = backtest.run(df, NightShift(stop_d=stop_d), c, enforce_mll=False)
    return r.trades.assign(mkt=mk)

if __name__ == "__main__":
    out = {}
    for sd in [0.5, 1.0, 2.0]:
        legs = [run("GC", sd, 1), run("NQ", sd, 1)]
        T = pd.concat(legs)
        d = T.groupby("trade_date").pnl.sum(); d.index = pd.to_datetime(pd.Index(d.index).astype(str)); d = d.sort_index()
        eq = d.cumsum(); last = d[d.index >= "2025-10-20"]
        t = T.pnl.mean() / (T.pnl.std() / np.sqrt(len(T)))
        mc = monte_carlo_combine(d, OroConfig())
        print(f"stop {sd}x daily range | trades {len(T)} win {(T.pnl>0).mean():.1%} net ${T.pnl.sum():,.0f} t {t:.2f} worstDD ${(eq-eq.cummax()).min():,.0f} stopped {(T.exit_reason=='stop').mean():.0%} | final yr ${last.sum():,.0f} | pass {mc['p_pass']:.0%} fail {mc['p_fail']:.0%}", flush=True)
        for mk in ["GC", "NQ"]:
            x = T[T.mkt == mk]; print(f"    {mk}: net ${x.pnl.sum():,.0f} t {x.pnl.mean()/(x.pnl.std()/np.sqrt(len(x))):.2f}")
        out[sd] = T
    pd.to_pickle(out, "results/night_shift_trades.pkl")
