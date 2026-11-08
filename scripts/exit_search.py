import sys, itertools, pickle; sys.path.insert(0,'.')
import pandas as pd
from multiprocessing import Pool
from oro.config import OroConfig, RiskConfig
from oro.data import load_csv, add_session_columns, add_indicators
from oro import backtest
from oro.strategies import NightShift
df = add_indicators(add_session_columns(load_csv('data/GC_5m_front.csv')))
c = OroConfig(); c.risk = RiskConfig(risk_per_trade=1e9, max_contracts=1, daily_loss_stop=1e9, daily_profit_stop=1e9, mll_buffer=0, max_trades_per_day=5)
def run(a):
    ex, tg, st = a
    return a, backtest.run(df, NightShift(stop_d=st, sma=0, exit_at=ex, tgt_d=tg, min_vol=0.8), c, enforce_mll=False).trades.reset_index(drop=True)
if __name__ == "__main__":
    grid = list(itertools.product(["02:00", "04:00", "06:00", "08:00", "09:30"], [0.75, 1.0, 1.5], [0.5]))
    with Pool(8) as p: res = dict(p.map(run, grid))
    pickle.dump(res, open("results/exit_trades.pkl", "wb")); print("ok", len(res))
