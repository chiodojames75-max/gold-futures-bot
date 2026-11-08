"""
Oro live engine (v2): every rule set on every market, scored by the Brain, filtered by the locked risk rules.

    python -m oro.live.engine            # PAPER (default): live TopstepX prices, simulated fills
    python -m oro.live.engine --live     # LIVE orders on the TopstepX account (asks once to arm)

Runs on YOUR Mac (Topstep forbids VPS/VPN). Open http://localhost:8765 for the live Oro dashboard.
Nothing here can be adjusted at run time: rules come from oro/rules.py (fingerprinted), the model from
results/brain2_model.pkl (retrained monthly by scripts/brain2.py). Create a file named KILL to flatten and stop;
that is a technical emergency stop, not a trading control.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from ..config import OroConfig
from ..data import add_session_columns, add_indicators
from ..markets import MARKETS, OPEN_AWARE
from ..rules import RULES, verify
from ..strategies import REGISTRY, Signal, hm
from .topstepx import TopstepX, TopstepXError, MARKET, LIMIT, STOP, BUY, SELL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RISK_FILE = os.path.join(ROOT, "TOPSTEP_RISK.json")


PROFILE_DIST = {150000: 4500.0, 100000: 3000.0, 50000: 2000.0}   # Topstep trailing max-loss by account size


def topstep_setting(key):
    try:
        with open(RISK_FILE) as f:
            v = json.load(f).get(key)
        return float(v) if v else None
    except Exception:
        return None


def topstep_lockout():
    """James's TopstepX daily loss lockout in $ (e.g. 450), from TOPSTEP_RISK.json. None if not set."""
    try:
        with open(RISK_FILE) as f:
            v = float(json.load(f).get("daily_loss_lockout_usd") or 0)
        return v if v > 0 else None
    except Exception:
        return None
STATE = {"mode": "Live · paper", "updated": None, "markets": {}, "trades": [], "brain": {}, "account": {}}
LOCK = threading.Lock()
JOURNAL = "journal.csv"


def log(event, **kw):
    line = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **kw}
    new = not os.path.exists(JOURNAL)
    with open(JOURNAL, "a") as f:
        if new:
            f.write("time,event,detail\n")
        f.write(f'{line["time"]},{event},"{json.dumps(kw, default=str).replace(chr(34), chr(34) * 2)}"\n')
    print(f"[{line['time']}] {event} {kw}", flush=True)


def sm(tod):
    return (tod - 18 * 60) % 1440


class Brain:
    """Loads the monthly model and scores a signal from its conditions."""

    def __init__(self, path):
        b = pickle.load(open(path, "rb"))
        self.model, self.cats, self.order, self.nums, self.thr = b["model"], b["cats"], b["cats_order"], b["nums"], b["threshold"]
        self.cfg_stats = b.get("cfg_stats", {})

    def code(self, feat, val):
        cats = self.cats[feat]
        v = val if feat in ("hour", "dow", "with_trend") else str(val)
        try:
            return float(cats.index(v))
        except ValueError:
            return np.nan

    def score(self, row: dict) -> float:
        x = [self.code(c, row[c]) for c in self.order] + [row.get(n, np.nan) for n in self.nums]
        return float(self.model.predict(np.array([x], dtype=float))[0])


class Engine:
    def __init__(self, live: bool):
        verify()                                        # refuse to run on edited rules
        self.live = live
        self.brain = None                               # Night Shift has no model gate
        self.api = TopstepX(os.environ.get("TOPSTEPX_BASE_URL") or "https://api.topstepx.com", os.environ["TOPSTEPX_USERNAME"], os.environ["TOPSTEPX_API_KEY"])
        accts = [a for a in self.api.accounts() if a.get("canTrade")]
        name = os.environ.get("TOPSTEPX_ACCOUNT", "")
        af = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ACCOUNT")
        if not name and os.path.exists(af):
            name = open(af).read().strip()
        if not name and len(accts) > 1:
            raise SystemExit("More than one tradable account: put the account name in the ACCOUNT file or set TOPSTEPX_ACCOUNT.")
        accts = [a for a in accts if not name or a.get("name") == name]
        if not accts:
            raise SystemExit("No tradable TopstepX account found (set TOPSTEPX_ACCOUNT).")
        self.aid = accts[0]["id"]
        self.account_name = accts[0].get("name", "")
        self.contracts = {mk: self.api.find_contract(MARKETS[mk]["contract"].symbol) for mk in MARKETS}
        from ..strategies import NightShift
        self.mkts = [mk for mk in MARKETS if MARKETS[mk]["contract"].symbol in RULES["markets"]]
        self.contracts = {mk: self.contracts[mk] for mk in self.mkts}
        self.cool_n = 10**9 if RULES["cool_down"] == "none" else int(RULES["cool_down"].split("last ")[1].split(" ")[0])
        self.strats = {(mk, "night_shift", "{}"): NightShift(stop_d=0.5, sma=0, exit_at=RULES["exit_et"], tgt_d=1.5, min_vol=0.8) for mk in self.mkts}
        self.hist = []          # outcomes of the rule's last nights, taken or watched (for the cool-down)
        self.shadow = {}        # watched trades during a cool-down: managed like real ones, never sent or counted
        self.day = None
        self.cfg_day = {}
        self.pos = {}                                   # market -> open position
        self.bal = float(accts[0].get("balance", 50_000.0))   # ProjectX Account/search returns "balance"
        # Topstep's max-loss line trails $2,000 under the best end-of-day balance and locks at $50,000. The API doesn't
        # report it, so James can type the real one from the TopstepX dashboard into TOPSTEP_RISK.json (max_loss_line_usd).
        size = topstep_setting("account_size") or (150_000.0 if self.bal > 100_000 else 50_000.0)
        self.acct = {"size": size, "dist": PROFILE_DIST.get(int(size), 2000.0)}   # 150K: $4,500 trailing max loss; 50K: $2,000
        self.mll = topstep_setting("max_loss_line_usd") or min(self.bal - self.acct["dist"], size)
        self.day_p, self.day_n = 0.0, 0
        self.last_bar = {}
        log("start", mode="LIVE" if live else "PAPER", rules=RULES)

    # ---------- one cycle, right after each 5-minute bar closes ----------
    def step(self):
        cands = []
        for mk in self.mkts:
            raw = self.api.bars(self.contracts[mk]["id"], 5, lookback_hours=24 * 45)    # ~9k bars: under the API's 20,000-bar cap; daily filters use daily bars
            if raw.empty:
                continue
            df = add_indicators(add_session_columns(raw))
            self._last_df = getattr(self, "_last_df", {}); self._last_df[mk] = df
            try:
                self.strats[(mk, "night_shift", "{}")].daily_closes = self.api.daily_closes(self.contracts[mk]["id"])
                self.strats[(mk, "night_shift", "{}")].daily_ranges = self.api.daily_ranges(self.contracts[mk]["id"])
            except Exception as e:                      # without daily history the 200-day filter stays off (no trades)
                log("daily_bars_error", error=str(e))
            if not self.hist and not getattr(self, "_seeded", False):
                self._seeded = True                     # seed the cool-down with the rule's real recent nights
                from .. import backtest
                from ..config import RiskConfig
                c = OroConfig(); c.contract = MARKETS[mk]["contract"]
                c.risk = RiskConfig(risk_per_trade=1e9, max_contracts=1, daily_loss_stop=1e9, daily_profit_stop=1e9, mll_buffer=0)
                past = backtest.run(df.iloc[:-1], self.strats[(mk, "night_shift", "{}")], c, enforce_mll=False).trades
                self.hist = [float(v) for v in past["pnl"].tail(self.cool_n)] if not past.empty else []
                log("cool_down_seeded", nights=len(self.hist), last10=round(sum(self.hist), 2))
            d = df.groupby("trade_date")["datr"].first()
            pct = d.rolling(100, min_periods=30).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
            df["vol_pct"] = df["trade_date"].map(pct)
            df["trend"] = np.sign(df["ema20"] - df["ema50"])
            r = next(df.tail(1).itertuples())
            self.last_close = getattr(self, "last_close", {}); self.last_close[mk] = (str(r.Index), float(r.close))
            ts = str(r.Index)
            if self.last_bar.get(mk) == ts:
                continue
            self.last_bar[mk] = ts
            if self.live:                                # live: the broker's balance is the truth, not our own book
                try:
                    a = [x for x in self.api.accounts() if x.get("id") == self.aid]
                    if a and a[0].get("balance") is not None:
                        self.bal = float(a[0]["balance"])
                except TopstepXError as e:
                    log("balance_error", error=str(e))
            self.roll_day(r.trade_date)
            self.manage(mk, r)
            STATE["markets"][mk] = {"name": MARKETS[mk]["name"], "district": MARKETS[mk]["district"],
                                    "symbol": MARKETS[mk]["contract"].symbol,
                                    "bars": [[i.isoformat(), float(c)] for i, c in df["close"].tail(300).items()], "picks": []}
            for key, strat in self.strats.items():
                if key[0] != mk:
                    continue
                pre = strat.precompute(df)
                row = next(pre.tail(1).itertuples(index=False))
                ctx = self.cfg_day.setdefault(key, {"trades": 0})
                sig = strat.signal(row, ctx)
                if not sig:
                    continue
                ctx["trades"] = ctx.get("trades", 0) + 1
                cands.append((1.0, mk, key, sig, r))
        for score, mk, key, sig, r in sorted(cands, key=lambda c: -c[0]):
            self.consider(score, mk, key, sig, r)
        self.publish()

    @staticmethod
    def vol_bucket(p):
        if p != p:
            return "nan"
        return "low" if p <= 0.33 else ("mid" if p <= 0.67 else "high")

    def roll_day(self, td):
        if td != self.day:
            if self.day is not None:
                self.mll = max(self.mll, min(self.bal - self.acct["dist"], self.acct["size"]))
                log("new_day", balance=self.bal, max_loss_line=self.mll)
            self.day, self.day_p, self.day_n, self.cfg_day = td, 0.0, 0, {}

    # ---------- the locked risk layer ----------
    def consider(self, score, mk, key, sig, r):
        R = RULES
        spec = MARKETS[mk]["contract"]
        pv = spec.tick_value / spec.tick_size
        risk1 = abs(r.close - sig.stop) * pv
        # James's TopstepX daily loss lockout: Topstep flattens the account at -$L, so a stop further away than that
        # is fiction. Pull the stop in so one micro can never lose more than the lockout (keeps $20 for fees + slippage).
        # Only ever makes a trade smaller or tighter. Tested: scripts/lockout_test.py.
        L = topstep_lockout(); budget = (L - 20) if L else None
        if budget and risk1 + 1.5 > budget:
            d = (budget - 1.5) / pv
            sig = Signal(sig.side, r.close - sig.side * d, sig.target, sig.reason + f" · stop pulled in to fit the ${L:,.0f} lockout")
            risk1 = abs(r.close - sig.stop) * pv
        rule = key[1]
        why = ""
        if mk in self.pos or mk in self.shadow:
            why = "position already open in this market"
        elif len(self.hist) >= self.cool_n and sum(self.hist[-self.cool_n:]) <= 0:
            why = f"Cool-down: the last {self.cool_n} nights lost money, watching instead"
        elif self.bal - self.mll < R["brake_stand_down_usd"] and getattr(self, "halt", False):
            why = "Brake: under $500 above the max-loss line, watching until the rule wins a night"
        elif sm(r.tod) >= sm(hm(R["flatten_et"])):
            why = "after the flatten time"
        elif risk1 <= 0 or risk1 > R["max_risk_per_contract_usd"]:
            why = f"stop risk ${risk1:,.0f} is over the ${R['max_risk_per_contract_usd']} limit"
        rec = {"m": mk, "rule": rule, "house": 0, "in": r.Index.isoformat(), "side": sig.side, "entry": float(r.close),
               "stop": round(sig.stop, 4), "target": round(sig.target, 4), "score": round(score, 3), "note": sig.reason}
        if why.startswith(("Cool-down", "Brake")):
            spec_ = MARKETS[mk]["contract"]
            self.shadow[mk] = {**rec, "size": 1, "stop": sig.stop, "target": sig.target, "shadow": True}
        if why:
            if True:
                with LOCK:
                    STATE["trades"].append({**rec, "status": "blocked", "blocked_by": why, "pnl": 0, "size": 0})
            return
        # locked sizing: floor(room / $700), 1..2; 1 micro once the room is under $1,000 (Sentinel's brake)
        room = self.bal - self.mll
        size = max(1, min(R["max_contracts"], int(room // R["max_risk_per_contract_usd"])))
        if room < R["brake_one_micro_usd"]:
            size = 1
        if budget:                                   # never size past the lockout: 2 micros only when 2 stops fit inside it
            size = max(1, min(size, int(budget // (risk1 + 1.5))))
        tick = spec.tick_size
        rnd = lambda p: round(round(p / tick) * tick, 6)
        pos = {**rec, "size": size, "stop": rnd(sig.stop), "target": rnd(sig.target), "status": "filled", "pnl": 0.0, "out": None}
        if self.live:
            cid = self.contracts[mk]["id"]
            e_side, x_side = (BUY, SELL) if sig.side > 0 else (SELL, BUY)
            self.api.place(self.aid, cid, e_side, size, MARKET)
            try:                                         # a position never sits without its stop: if the stop can't be placed, get out
                pos["stop_id"] = self.api.place(self.aid, cid, x_side, size, STOP, stop_price=pos["stop"])
                pos["target_id"] = self.api.place(self.aid, cid, x_side, size, LIMIT, limit_price=pos["target"])
            except TopstepXError as e:
                log("protect_failed_flattening", error=str(e))
                self.api.flatten(self.aid, cid)
                with LOCK:
                    STATE["trades"].append({**rec, "status": "blocked", "blocked_by": f"stop/target order failed, flattened: {e}", "pnl": 0, "size": 0})
                return
        self.pos[mk] = pos
        self.day_n += 1
        with LOCK:
            STATE["trades"].append(pos)
        log("entry", market=mk, **{k: v for k, v in pos.items() if k not in ("m",)})

    # ---------- manage open positions (paper fills use the backtest's pessimistic rules) ----------
    def manage(self, mk, r):
        if mk in self.shadow:
            self._close_if_done(self.shadow, mk, r, shadow=True)
        p = self.pos.get(mk)
        if not p:
            return
        if self.live and self._broker_flat(mk):
            # the broker's stop or target already filled (it can fill on a touch our 5-minute bars don't show):
            # cancel the other resting order so it can't open a new position, then book the exit
            self.api.flatten_orders(self.aid, self.contracts[mk]["id"])
            px, why = (p["target"], "target") if r.high >= p["target"] else (p["stop"], "stop")
            if sm(r.tod) >= sm(hm(RULES["exit_et"])):
                px, why = r.close, "time exit"
            pv = MARKETS[mk]["contract"].tick_value / MARKETS[mk]["contract"].tick_size
            pnl = (px - p["entry"]) * p["side"] * p["size"] * pv - 1.5 * p["size"]
            del self.pos[mk]
            self._remember(mk, p, px, why + " (broker)", pnl, False)
            p.update(out=r.Index.isoformat(), exit=px, why=why + " (broker)", pnl=round(pnl, 2))
            self.day_p += pnl
            log("exit", market=mk, why=why + " (broker)", price=px, pnl=round(pnl, 2))
            return
        self._close_if_done(self.pos, mk, r, shadow=False)

    def _broker_flat(self, mk):
        try:
            cid = self.contracts[mk]["id"]
            return not any(str(x.get("contractId")) == str(cid) and x.get("size", 0) for x in self.api.open_positions(self.aid))
        except TopstepXError as e:
            log("position_check_error", error=str(e))
            return False

    def _close_if_done(self, book, mk, r, shadow):
        p = book[mk]
        spec = MARKETS[mk]["contract"]
        pv, slip = spec.tick_value / spec.tick_size, spec.tick_size
        s = p["side"]
        exit_px, why = None, ""
        if sm(r.tod) >= sm(hm(RULES["exit_et"])):
            exit_px, why = r.close - s * slip, "time exit"
        elif (r.low <= p["stop"]) if s > 0 else (r.high >= p["stop"]):
            exit_px, why = p["stop"] - s * slip, "stop"
        elif (r.high >= p["target"] + slip) if s > 0 else (r.low <= p["target"] - slip):
            exit_px, why = p["target"], "target"
        if exit_px is None:
            return
        pnl = (exit_px - p["entry"]) * s * p["size"] * pv - 1.5 * p["size"]
        self.hist.append(pnl)
        del book[mk]
        self._remember(mk, p, exit_px, why, pnl, shadow)
        if shadow:
            self.halt = not (pnl > 0)            # the brake lifts after the rule's next winning night
            log("watched_exit", market=mk, why=why, pnl=round(pnl, 2))
            return
        if self.live:
            self.api.flatten(self.aid, self.contracts[mk]["id"])
        p.update(out=r.Index.isoformat(), exit=exit_px, why=why, pnl=round(pnl, 2))
        self.bal += pnl; self.day_p += pnl
        self.halt = (self.bal - self.mll < RULES["brake_stand_down_usd"]) and pnl <= 0
        log("exit", market=mk, why=why, price=exit_px, pnl=round(pnl, 2), balance=round(self.bal, 2))

    def _remember(self, mk, p, exit_px, why, pnl, shadow):
        """Write the trade into the Obsidian vault and re-run the learner. Never affects trading."""
        try:
            from oro import memory
            df = self._last_df[mk]; g = df.groupby("trade_date")
            daily = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
            daily.index = pd.to_datetime(pd.Index(daily.index).astype(str))
            d = str(df["trade_date"].iloc[-1]); agents = None
            tj = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "desk", "today.json")
            if os.path.exists(tj):
                agents = {a["id"]: a.get("view") for a in json.load(open(tj)).get("agents", [])}
            memory.write_trade_note({"date": d, "side": p["side"], "entry": p["entry"], "stop": p["stop"], "target": p["target"], "exit": exit_px,
                                     "pnl": pnl, "exit_reason": why, "size": p.get("size", 1)}, memory.features_at(daily, d), agents,
                                    source="watched (brake)" if shadow else ("live" if self.live else "live-paper"))
            memory.learn()
            try:
                from oro import mind
                mind.reflect()
            except Exception as e:
                log("mind_error", error=str(e))
        except Exception as e:
            log("memory_error", error=str(e))

    def publish(self):
        with LOCK:
            STATE["updated"] = datetime.now(timezone.utc).isoformat()
            STATE["mode"] = "Live · " + ("real orders" if self.live else "paper")
            STATE["account"] = {"balance": round(self.bal, 2), "max_loss_line": self.mll, "day_pnl": round(self.day_p, 2),
                                "trades_today": self.day_n, "cushion": round(self.bal - self.mll, 2), "live": bool(self.live),
                                "name": getattr(self, "account_name", "")}
            STATE["brain"] = {"rules": RULES}
            STATE["pip"] = self._pip_status()
            STATE["risk"] = self._risk_status()
            try:
                from oro.live import news
                STATE["news"] = {"headlines": news.headlines(), "calendar": news.calendar(), "note": "display only: news never triggers a trade"}
            except Exception:
                pass
            STATE["trades"] = STATE["trades"][-400:]

    def _risk_status(self):
        L = topstep_lockout()
        return {"lockout": L, "account_size": int(self.acct["size"]), "max_loss_usd": self.acct["dist"], "source": "TOPSTEP_RISK.json"}

    def _pip_status(self):
        """What Pip is doing right now, for the floor. Read-only: describes state, decides nothing."""
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York")); mins = et.hour * 60 + et.minute
        room = self.bal - self.mll
        out = {"et": et.strftime("%a ") + et.strftime("%I:%M %p").lstrip("0"), "cushion": round(room, 2), "live": bool(self.live), "account": getattr(self, "account_name", "")}
        if _kill_exists():
            out.update(phase="stopped", text="Stopped: KILL file present")
            return out
        for mk, p in self.pos.items():
            ts, last = getattr(self, "last_close", {}).get(mk, (None, p["entry"]))
            upnl = (last - p["entry"]) * p["side"] * p["size"] * 10.0 - 1.5 * p["size"]
            to_exit = (hm(RULES["exit_et"]) - mins) % 1440
            out.update(phase="in_trade", text=f"{'Long' if p['side'] > 0 else 'Short'} {p['size']} MGC @ {p['entry']:.1f}", side=p["side"], size=p["size"],
                       entry=p["entry"], stop=p["stop"], target=p["target"], last=last, last_bar=ts, unrealized=round(upnl, 2), minutes_to_exit=int(to_exit), opened=p.get("in"))
            return out
        if self.shadow or (room < RULES["brake_stand_down_usd"] and getattr(self, "halt", False)):
            out.update(phase="standing_down", text="Standing down: Sentinel's brake (watching on paper until the rule wins a night)")
            return out
        if self.day_n > 0:
            out.update(phase="done", text="Done for the night", day_pnl=round(self.day_p, 2))
            return out
        entry = hm(RULES["entry_et"])
        out.update(phase="waiting", text="Waiting for the 6:05 PM entry", minutes_to_entry=int((entry - mins) % 1440))
        return out

    def loop(self):
        while True:
            if _kill_exists():
                for mk in list(self.pos):
                    if self.live:
                        self.api.flatten(self.aid, self.contracts[mk]["id"])
                log("stopped", reason="KILL file")
                return
            try:
                self.step()
            except TopstepXError as e:
                log("api_error", error=str(e))
            now = time.time()
            time.sleep(max(1.0, (now // 300 + 1) * 300 + 6 - now))


def _kill_exists():
    return os.path.exists("KILL") or os.path.exists(os.path.join(ROOT, "KILL"))


def _brain_path():
    from .. import memory
    return os.path.join(memory.VAULT, "Mind", "brain.json")


VAULTS = {"vault": os.path.abspath(os.path.join(ROOT, "..", "vault")),
          "brainvault": os.path.expanduser("~/Documents/James HQ/Businesses/Oro")}


def _vault_index():
    """List the markdown notes in both vaults (read-only), for the floor's Brain tab."""
    out = {}
    for key, base in VAULTS.items():
        notes = []
        if os.path.isdir(base):
            for dp, dns, fs in os.walk(base):
                dns[:] = [d for d in dns if not d.startswith(".")]
                rel_dir = os.path.relpath(dp, base)
                if key == "vault" and rel_dir.startswith("Trades") and rel_dir.count(os.sep) >= 1:
                    fs = [f for f in fs if f[:-3] == os.path.basename(dp)]      # only the year index, not 900 trade notes
                for f in fs:
                    if f.endswith(".md"):
                        notes.append(os.path.normpath(os.path.join(rel_dir, f)).replace(os.sep, "/"))
        out[key] = {"name": os.path.basename(base), "notes": sorted(notes)}
    return out


def _vault_read(key, rel):
    base = VAULTS.get(key)
    if not base:
        return None
    p = os.path.abspath(os.path.join(base, rel))
    if not p.startswith(base + os.sep) or not p.endswith(".md") or not os.path.isfile(p):
        return None
    return open(p, "rb").read()


def serve(port=8765):
    if not os.path.exists(_brain_path()):          # first run: let Pip's mind write its summary (read-only, background)
        def _first():
            try:
                from .. import mind
                mind.reflect()
            except Exception as e:
                log("mind_error", error=str(e))
        threading.Thread(target=_first, daemon=True).start()
    app_dir = os.path.join(ROOT, "app")

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=app_dir, **k)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            if self.path.startswith("/state.json"):
                with LOCK:
                    body = json.dumps(STATE, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/brain.json"):
                try:
                    body = open(_brain_path(), "rb").read()
                except OSError:
                    body = json.dumps({"available": False, "reason": "Pip's mind hasn't written brain.json yet"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/vault/index.json"):
                body = json.dumps(_vault_index()).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body); return
            if self.path.startswith("/vault/") or self.path.startswith("/brainvault/"):
                from urllib.parse import unquote
                key, _, rel = self.path.lstrip("/").partition("/")
                body = _vault_read(key, unquote(rel.split("?")[0]))
                if body is None:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200); self.send_header("Content-Type", "text/markdown; charset=utf-8"); self.end_headers(); self.wfile.write(body); return
            if self.path in ("/", ""):
                self.path = "/floor.html"
            return super().do_GET()

        def do_POST(self):
            """Only one control exists: Stop Oro (creates the KILL file). Localhost only, needs an explicit confirm."""
            if self.path != "/stop" or self.client_address[0] not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
                self.send_response(403); self.end_headers(); return
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
                ok = json.loads(self.rfile.read(n) or b"{}").get("confirm") == "STOP"
            except Exception:
                ok = False
            if not ok:
                self.send_response(400); self.end_headers(); return
            open(os.path.join(ROOT, "KILL"), "w").write(f"created from the floor at {datetime.now().isoformat(timespec='seconds')}\n")
            log("kill_requested", source="floor")
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"stopping": true}')

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Oro dashboard: http://localhost:{port}  (iPhone on the same Wi-Fi: http://<this Mac's IP>:{port})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="send real orders (default: paper)")
    ap.add_argument("--check", action="store_true", help="read-only connection test: account, contract, bars; places nothing")
    ap.add_argument("--viewer", action="store_true", help="serve the floor + brain only (no engine, no API, cannot trade)")
    a = ap.parse_args()
    if a.viewer:
        STATE.update({"mode": "Viewer · engine not running", "offline": "viewer", "risk": {"lockout": topstep_lockout(), "account_size": int(topstep_setting("account_size") or 150000), "source": "TOPSTEP_RISK.json"}})
        serve()
        print("Oro viewer: floor + brain at http://localhost:8765 · nothing can trade in this mode · Ctrl+C to quit")
        while True:
            time.sleep(3600)
    os.environ.setdefault("TOPSTEPX_BASE_URL", "https://api.topstepx.com")
    if not os.environ.get("TOPSTEPX_BASE_URL"):
        os.environ["TOPSTEPX_BASE_URL"] = "https://api.topstepx.com"
    for k in ("TOPSTEPX_USERNAME", "TOPSTEPX_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"Missing {k}. Add it to ~/.zshrc (see GO_LIVE.md).")
    if a.check:
        verify()
        api = TopstepX(os.environ["TOPSTEPX_BASE_URL"], os.environ["TOPSTEPX_USERNAME"], os.environ["TOPSTEPX_API_KEY"])
        api.login(); print("login ok")
        for ac in api.accounts(): print("account:", ac.get("name"), ac.get("id"), "balance", ac.get("balance"), "canTrade", ac.get("canTrade"))
        c = api.find_contract("MGC"); print("contract:", c.get("name"), c.get("id"))
        b = api.bars(c["id"], 5, lookback_hours=24); print("5-min bars:", len(b), "last", b.index[-1] if len(b) else None, b.close.iloc[-1] if len(b) else None)
        dr = api.daily_ranges(c["id"]); dc = api.daily_closes(c["id"]); print("daily bars:", len(dc), "ranges:", len(dr), "(need 100+ for the quiet filter)")
        print("CHECK PASSED: nothing was ordered."); return
    if a.live and input("Type ORO-LIVE to arm real orders: ").strip().strip(".").upper() != "ORO-LIVE":
        sys.exit("Not armed.")
    serve()
    Engine(a.live).loop()


if __name__ == "__main__":
    main()
