"""
The "learning" loop, done safely: Oro never edits its own rules while trading.

Weekly, run:
  python -m oro.review --journal journal.csv --oos-trades results/comex_orb_oos_trades.csv

It compares live/paper results with what the backtest said to expect and returns one of:
  OK        - results are inside the range the backtest predicted
  WATCH     - drifting toward the bad edge of that range
  HALT      - outside the range: the edge may be gone. Stop, re-test on fresh data, and decide.
Any parameter change goes back through scripts/run_research.py and you approve it by editing config.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def load_exits(journal: str) -> pd.Series:
    j = pd.read_csv(journal)
    ex = j[j.event == "exit"]
    pnl = ex["detail"].map(lambda d: json.loads(d).get("pnl"))
    return pd.Series(pnl.astype(float).to_numpy())


def review(live_pnl: pd.Series, expected_trade_pnl: float, expected_trade_sd: float) -> dict:
    n = len(live_pnl)
    if n < 20:
        return {"status": "TOO EARLY", "trades": n, "note": "need 20+ trades before judging"}
    mean = live_pnl.mean()
    se = expected_trade_sd / np.sqrt(n)
    z = (mean - expected_trade_pnl) / se if se > 0 else 0.0
    status = "OK" if z > -1.0 else ("WATCH" if z > -2.0 else "HALT")
    return {"status": status, "trades": n, "live_avg_$": round(mean, 2),
            "expected_avg_$": round(expected_trade_pnl, 2), "z_score": round(z, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="journal.csv")
    ap.add_argument("--oos-trades", required=True,
                    help="CSV of out-of-sample backtest trades for this strategy (results/<name>_oos_trades.csv)")
    a = ap.parse_args()
    bt = pd.read_csv(a.oos_trades)["pnl"]
    print(json.dumps(review(load_exits(a.journal), bt.mean(), bt.std()), indent=2))


if __name__ == "__main__":
    main()
