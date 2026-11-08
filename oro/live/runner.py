"""
Oro live loop. Runs on YOUR computer (Topstep forbids VPS/VPN).

  python -m oro.live.runner --strategy comex_orb            # PAPER (default): real data, simulated fills
  python -m oro.live.runner --strategy comex_orb --live     # LIVE: sends real orders to TopstepX

Every 5 minutes, just after a bar closes:
  1. pull recent bars from TopstepX
  2. run the SAME strategy code the backtester used, on the bar that just closed
  3. pass any signal through the SAME RiskManager
  4. paper: simulate the fill pessimistically | live: market entry + stop order + limit target (OCO managed here)
  5. write every decision to journal.csv

Fixed rules only. Oro never changes its own parameters. See oro/review.py for the human-approved loop.
Kill switch: create a file named KILL in the run folder -> Oro flattens and exits.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

from ..config import OroConfig
from ..data import add_session_columns, add_indicators
from ..risk import AccountState, RiskManager
from ..strategies import REGISTRY, hm
from .topstepx import TopstepX, TopstepXError, MARKET, LIMIT, STOP, BUY, SELL

STATE_FILE = "oro_state.json"
JOURNAL = "journal.csv"


def sm(tod: int) -> int:
    return (tod - 18 * 60) % 1440


def log(event: str, **kw):
    row = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **kw}
    new = not os.path.exists(JOURNAL)
    with open(JOURNAL, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "event", "detail"])
        if new:
            w.writeheader()
        w.writerow({"time": row["time"], "event": event, "detail": json.dumps(kw, default=str)})
    print(f"[{row['time']}] {event} {kw if kw else ''}", flush=True)


class Runner:
    def __init__(self, strategy_name: str, live: bool, cfg: OroConfig):
        self.cfg, self.live = cfg, live
        self.strat = REGISTRY[strategy_name]()
        self.rm = RiskManager(cfg)
        self.api = TopstepX(os.environ.get("TOPSTEPX_BASE_URL", ""),
                            os.environ["TOPSTEPX_USERNAME"], os.environ["TOPSTEPX_API_KEY"])
        self.contract = self.api.find_contract(cfg.contract.symbol)
        self.cid = self.contract["id"]
        acct_name = os.environ.get("TOPSTEPX_ACCOUNT", "")
        accts = [a for a in self.api.accounts() if a.get("canTrade")]
        if acct_name:
            accts = [a for a in accts if a.get("name") == acct_name]
        if not accts:
            raise SystemExit("No tradable account found (set TOPSTEPX_ACCOUNT to the exact account name).")
        self.account = accts[0]
        self.aid = self.account["id"]
        self.state = self._load_state()
        self.pos = self.state.get("pos")          # open position dict or None
        self.day_ctx = self.state.get("day_ctx", {"trades": 0})
        self.last_bar = self.state.get("last_bar")
        log("start", mode="LIVE" if live else "PAPER", strategy=self.strat.name, params=self.strat.params,
            account=self.account.get("name"), contract=self.contract.get("name"))

    # ---------- persistence: a restart must not reset daily loss counters ----------
    def _load_state(self) -> dict:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                s = json.load(f)
            self.acct = AccountState(**s["acct"])
            return s
        t = self.cfg.topstep
        bal = float(self.account.get("balance", t.start_balance))   # VERIFY field name
        self.acct = AccountState(balance=bal, mll=bal - t.max_loss_limit)
        return {"trade_date": None}

    def _save(self):
        self.state.update({"acct": self.acct.__dict__, "pos": self.pos, "day_ctx": self.day_ctx,
                           "last_bar": self.last_bar})
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, default=str, indent=1)

    # ---------- main loop ----------
    def loop(self):
        bar = self.cfg.bar_minutes
        while True:
            if os.path.exists(self.cfg.risk.kill_switch_file):
                self.flatten("kill switch")
                log("stopped", reason="kill switch file")
                return
            try:
                self.step()
            except TopstepXError as e:
                log("api_error", error=str(e))
            except Exception as e:  # never die silently with a position on
                log("crash", error=repr(e))
                self.flatten("unexpected error")
                raise
            now = time.time()
            nxt = (now // (bar * 60) + 1) * bar * 60 + 5   # 5s after the next bar closes
            time.sleep(max(1.0, nxt - now))

    def step(self):
        raw = self.api.bars(self.cid, self.cfg.bar_minutes)
        if raw.empty:
            return
        df = self.strat.precompute(add_indicators(add_session_columns(raw)))
        last_ts = str(df.index[-1])
        if last_ts == self.last_bar:
            return                       # no new closed bar yet
        self.last_bar = last_ts
        r = next(df.tail(1).itertuples(index=False))
        cur_sm = sm(int(r.tod))

        # new trading day -> roll risk counters and Topstep MLL
        if str(r.trade_date) != str(self.state.get("trade_date")):
            if self.state.get("trade_date") is not None:
                self.rm.end_of_day(self.acct)
                log("new_day", balance=self.acct.balance, mll=self.acct.mll)
            self.state["trade_date"] = str(r.trade_date)
            self.day_ctx = {"trades": 0}

        if self.live:
            self.sync_live()

        # manage the open position
        if self.pos:
            flat_by = min(sm(hm(self.cfg.topstep.flatten_et)), sm(hm(self.strat.exit_by)))
            if cur_sm >= flat_by:
                self.flatten("time exit", price=r.close)
            elif not self.live:
                self.paper_manage(r)

        # look for a new entry on the bar that just closed
        flat_by = min(sm(hm(self.cfg.topstep.flatten_et)), sm(hm(self.strat.exit_by)))
        if not self.pos and cur_sm < flat_by:
            sig = self.strat.signal(r, self.day_ctx)
            if sig:
                self.enter(sig, ref_price=r.close)
        self._save()

    # ---------- entries ----------
    def enter(self, sig, ref_price: float):
        ok, why = self.rm.can_open(self.acct)
        if not ok:
            log("signal_blocked", reason=why, signal=sig.__dict__)
            return
        size = self.rm.size_for(ref_price, sig.stop)
        if size == 0:
            log("signal_skipped", reason="stop too wide for risk_per_trade", signal=sig.__dict__)
            return
        tick = self.cfg.contract.tick_size
        rnd = lambda p: round(round(p / tick) * tick, 2)
        stop, target = rnd(sig.stop), rnd(sig.target)
        self.day_ctx["trades"] = self.day_ctx.get("trades", 0) + 1
        self.pos = {"side": sig.side, "size": size, "entry": ref_price, "stop": stop, "target": target,
                    "reason": sig.reason, "opened": datetime.now().isoformat()}
        if self.live:
            entry_side, exit_side = (BUY, SELL) if sig.side > 0 else (SELL, BUY)
            self.api.place(self.aid, self.cid, entry_side, size, MARKET)
            self.pos["stop_id"] = self.api.place(self.aid, self.cid, exit_side, size, STOP, stop_price=stop)
            self.pos["target_id"] = self.api.place(self.aid, self.cid, exit_side, size, LIMIT, limit_price=target)
        log("entry", **self.pos)

    # ---------- paper fills (same pessimistic rules as the backtester) ----------
    def paper_manage(self, r):
        p, c = self.pos, self.cfg
        s, slip = p["side"], c.costs.slippage_ticks * c.contract.tick_size
        hit_stop = r.low <= p["stop"] if s > 0 else r.high >= p["stop"]
        hit_tgt = r.high >= p["target"] + c.contract.tick_size if s > 0 else r.low <= p["target"] - c.contract.tick_size
        if hit_stop:
            self._paper_close(p["stop"] - s * slip, "stop")
        elif hit_tgt:
            self._paper_close(p["target"], "target")

    def _paper_close(self, price: float, why: str):
        p = self.pos
        pnl = (price - p["entry"]) * p["side"] * p["size"] * self.cfg.contract.point_value \
            - self.cfg.costs.commission_rt * p["size"]
        self.rm.after_trade(self.acct, pnl)
        log("exit", why=why, price=price, pnl=round(pnl, 2), balance=round(self.acct.balance, 2), **p)
        self.pos = None

    # ---------- live position sync / OCO ----------
    def sync_live(self):
        if not self.pos:
            return
        positions = [x for x in self.api.open_positions(self.aid) if x.get("contractId") == self.cid]
        if positions:
            return
        # flat at the broker -> one bracket leg filled; cancel the other leg (OCO)
        for key in ("stop_id", "target_id"):
            try:
                self.api.cancel(self.aid, self.pos[key])
            except (TopstepXError, KeyError):
                pass
        bal_before = self.acct.balance
        acct = next((a for a in self.api.accounts() if a["id"] == self.aid), {})
        new_bal = float(acct.get("balance", bal_before))       # VERIFY field name
        self.rm.after_trade(self.acct, new_bal - bal_before)
        log("exit", why="bracket filled", pnl=round(new_bal - bal_before, 2), balance=new_bal)
        self.pos = None

    def flatten(self, why: str, price: float | None = None):
        if not self.pos:
            return
        if self.live:
            self.api.flatten(self.aid, self.cid)
            self.sync_live()
            if self.pos:   # broker still shows flat-pending; record and clear
                log("flatten_sent", why=why)
                self.pos = None
        else:
            slip = self.cfg.costs.slippage_ticks * self.cfg.contract.tick_size
            self._paper_close((price if price is not None else self.pos["entry"]) - self.pos["side"] * slip, why)
        self._save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=list(REGISTRY))
    ap.add_argument("--live", action="store_true", help="send REAL orders (default is paper)")
    a = ap.parse_args()
    for k in ("TOPSTEPX_BASE_URL", "TOPSTEPX_USERNAME", "TOPSTEPX_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"Missing environment variable {k}. See README.")
    if a.live:
        print("LIVE MODE: real orders on your Topstep account.")
        if input("Type ORO-LIVE to continue: ").strip() != "ORO-LIVE":
            sys.exit("Aborted.")
    Runner(a.strategy, a.live, OroConfig()).loop()


if __name__ == "__main__":
    main()
