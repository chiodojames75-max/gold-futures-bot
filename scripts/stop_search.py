import sys, itertools, pickle; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from multiprocessing import Pool
from oro.config import OroConfig, RiskConfig
from oro.data import load_csv, add_session_columns, add_indicators
from oro import backtest
from oro.strategies import NightShift, Signal, _ok
df = add_indicators(add_session_columns(load_csv('data/GC_5m_front.csv')))
c = OroConfig(); c.risk = RiskConfig(risk_per_trade=1e9, max_contracts=1, daily_loss_stop=1e9, daily_profit_stop=1e9, mll_buffer=0, max_trades_per_day=5)
class Struct(NightShift):
    def __init__(self, stop_kind, tgt_kind):
        super().__init__(stop_d=0.5, sma=0, exit_at="08:00", tgt_d=1.0, min_vol=0.8); self.sk, self.tk = stop_kind, tgt_kind; self.name = f"struct_{stop_kind}_{tgt_kind}"
    def precompute(self, d):
        d = super().precompute(d)
        d["low12"] = d["low"].rolling(12).min(); d["low48"] = d["low"].rolling(48).min()
        g = d.groupby("trade_date"); pl = g["low"].min().shift(1); ph = g["high"].max().shift(1)
        d["plow"] = d["trade_date"].map(pl); d["phigh"] = d["trade_date"].map(ph); return d
    def signal(self, r, day):
        s = super().signal(r, day)
        if s is None: return None
        a, e, buf = r.datr, r.close, 0.05 * r.datr
        lv = {"hour": r.low12, "4h": r.low48, "session": r.plow}
        if self.sk in lv: stop = lv[self.sk] - buf
        else:   # nearest level at least 0.25 range below entry
            cands = sorted([v - buf for v in (r.low12, r.low48, r.plow) if v == v and e - (v - buf) >= 0.25 * a], reverse=True)
            stop = cands[0] if cands else e - 0.5 * a
        if stop != stop: stop = e - 0.5 * a
        dist = min(max(e - stop, 0.25 * a), 0.75 * a); stop = e - dist
        if self.tk == "1x": tgt = e + a
        elif self.tk == "1.5R": tgt = e + 1.5 * dist
        elif self.tk == "2R": tgt = e + 2 * dist
        else: tgt = r.phigh if (r.phigh == r.phigh and r.phigh - e >= 0.25 * a) else e + a
        return Signal(+1, stop, tgt, f"{self.sk} stop / {self.tk} target")
def run(a):
    sk, tk = a
    return a, backtest.run(df, Struct(sk, tk), c, enforce_mll=False).trades.reset_index(drop=True)
if __name__ == "__main__":
    grid = list(itertools.product(["hour", "4h", "session", "nearest"], ["1x", "1.5R", "2R", "level"]))
    with Pool(8) as p: res = dict(p.map(run, grid))
    pickle.dump(res, open("results/stop_trades.pkl", "wb")); print("ok", len(res))
