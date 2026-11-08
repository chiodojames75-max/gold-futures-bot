"""Oro memory: an Obsidian vault that records every trade and learns slowly from all of them.

Each trade (real, paper or backtest) becomes a markdown note with YAML frontmatter that Obsidian can
search, graph and query. After every trade `learn()` re-reads the whole vault and rewrites Lessons.md:
win rate and average result for every market condition, shrunk toward the overall average so a handful
of trades can't create a fake rule.

Guardrails (why it learns slowly on purpose):
- a condition is only FLAGGED when it has >= 50 trades AND the same sign shows up in both the older and
  the newer half of the history;
- flags are proposals for the desk. The locked rules (rules.py, fingerprint-checked) never change by
  themselves: James approves, the change is backtested on unseen data, then re-locked.
"""
from __future__ import annotations
import json, os, re
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAULT = os.environ.get("ORO_VAULT", os.path.join(ROOT, "..", "vault"))
FEATURES = ["weekday", "trend", "vol_bucket", "prev_day", "prev_week", "stretch"]

def _bucket(name, v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "n/a"
    if name == "vol":   return "quiet" if v <= 0.8 else "normal" if v <= 1.25 else "busy"
    if name == "ret1":  return "big down" if v <= -0.01 else "down" if v < -0.003 else "flat" if v <= 0.003 else "up" if v < 0.01 else "big up"
    if name == "ret5":  return "down week" if v < 0 else "up week"
    if name == "z20":   return "stretched low" if v < -1.5 else "low" if v < -0.5 else "middle" if v <= 0.5 else "high" if v <= 1.5 else "stretched high"
    return str(v)

def features_at(daily: pd.DataFrame, trade_date) -> dict:
    """daily: index = trade dates, columns close/high/low/open (finished sessions). Values known at the 6:05 PM entry."""
    d = daily[daily.index < pd.Timestamp(trade_date)]
    if len(d) < 25: return {}
    c, rng = d["close"], d["high"] - d["low"]; sma = c.rolling(200).mean()
    trend = "n/a"
    if not np.isnan(sma.iloc[-1]):
        up = c.iloc[-1] > sma.iloc[-1]; rising = sma.iloc[-1] > sma.iloc[-21]
        trend = "on" if up and rising else "off" if (not up and not rising) else "mixed"
    vol = rng.iloc[-5:].mean() / rng.iloc[-100:].mean() if len(rng) >= 100 else np.nan
    return {"weekday": pd.Timestamp(trade_date).day_name(), "trend": trend, "vol_bucket": _bucket("vol", vol),
            "prev_day": _bucket("ret1", c.iloc[-1] / c.iloc[-2] - 1), "prev_week": _bucket("ret5", c.iloc[-1] / c.iloc[-6] - 1),
            "stretch": _bucket("z20", (c.iloc[-1] - c.iloc[-20:].mean()) / c.iloc[-20:].std()), "vol_ratio": None if np.isnan(vol) else round(float(vol), 2)}

def write_trade_note(t: dict, feats: dict, agents: dict | None = None, source: str = "live-paper", lesson: str | None = None) -> str:
    """t needs: date, entry_time, exit_time, side, entry, stop, target, exit, pnl, exit_reason, size."""
    date = str(t["date"])[:10]; yr = date[:4]; folder = os.path.join(VAULT, "Trades", yr); os.makedirs(folder, exist_ok=True)
    res = "win" if t["pnl"] > 0 else "loss"
    r_mult = (t["exit"] - t["entry"]) / max(1e-9, abs(t["entry"] - t["stop"])) * (1 if t["side"] > 0 else -1)
    if lesson is None:
        lesson = {"target": "Drift ran all the way to target.", "stop": "Stopped out: the overnight drift failed this night.",
                  "time exit": "Closed at 8:00 AM, " + ("ahead." if res == "win" else "behind.")}.get(t["exit_reason"], "")
    fm = {"date": date, "source": source, "result": res, "pnl": round(float(t["pnl"]), 2), "r": round(float(r_mult), 2),
          "side": "long" if t["side"] > 0 else "short", "size": int(t.get("size", 1)), "entry": t["entry"], "stop": t["stop"],
          "target": t["target"], "exit": t["exit"], "exit_reason": t["exit_reason"], **{k: feats.get(k) for k in FEATURES + ["vol_ratio"]}}
    if agents: fm.update({f"agent_{k}": v for k, v in agents.items()})
    y = "\n".join(f"{k}: {json.dumps(v) if isinstance(v, str) else ('null' if v is None else v)}" for k, v in fm.items())
    body = (f"# {date} · {fm['side'].upper()} · {'+' if t['pnl'] > 0 else '−'}${abs(t['pnl']):,.0f}\n\n"
            f"Entry {t['entry']} → exit {t['exit']} ({t['exit_reason']}) · stop {t['stop']} · target {t['target']} · {fm['r']}R\n\n"
            f"**Conditions:** trend [[Trend {str(fm['trend']).replace('/', '-')}]] · volatility [[Vol {str(fm['vol_bucket']).replace('/', '-')}]] · {fm['weekday']} · prev day {fm['prev_day']} · {fm['prev_week']} · {fm['stretch']}\n\n"
            f"**Lesson:** {lesson}\n\nSee [[Lessons]].\n")
    path = os.path.join(folder, f"{date}.md")
    with open(path, "w") as f: f.write(f"---\n{y}\n---\n{body}")
    return path

def _read_notes() -> pd.DataFrame:
    rows = []
    for dp, _, fs in os.walk(os.path.join(VAULT, "Trades")):
        for fn in fs:
            if not fn.endswith(".md"): continue
            txt = open(os.path.join(dp, fn)).read(); m = re.match(r"---\n(.*?)\n---", txt, re.S)
            if not m: continue
            r = {}
            for line in m.group(1).splitlines():
                k, _, v = line.partition(": ")
                try: r[k] = json.loads(v)
                except Exception: r[k] = v
            if r.get("type") == "index" or "pnl" not in r:   # skip index notes (Trades/<year>/<year>.md), only real trade notes count
                continue
            rows.append(r)
    return pd.DataFrame(rows)

def learn(k: int = 30, min_n: int = 50) -> dict:
    df = _read_notes()
    if df.empty: return {}
    df["pnl"] = df["pnl"].astype(float); df = df.sort_values("date"); mid = df["date"].iloc[len(df) // 2]
    base = df["pnl"].mean(); out = {"trades": len(df), "avg": round(base, 1), "win": round((df.pnl > 0).mean() * 100, 1), "flags": [], "tables": {}}
    for f in FEATURES:
        if f not in df: continue
        rows = []
        for v, g in df.groupby(f):
            n = len(g); shr = (g.pnl.sum() + k * base) / (n + k)
            a, b = g[g.date < mid].pnl, g[g.date >= mid].pnl
            rows.append({"value": v, "n": n, "win%": round((g.pnl > 0).mean() * 100, 1), "avg": round(g.pnl.mean(), 1), "shrunk": round(shr, 1),
                         "older": round(a.mean(), 1) if len(a) else None, "newer": round(b.mean(), 1) if len(b) else None})
            if n >= min_n and len(a) >= 15 and len(b) >= 15:
                if a.mean() < 0 and b.mean() < 0: out["flags"].append(f"AVOID? {f} = {v}: loses in both halves (older {a.mean():+.0f}, newer {b.mean():+.0f} per trade, n={n})")
                elif a.mean() > 2 * base and b.mean() > 2 * base: out["flags"].append(f"FAVOR? {f} = {v}: more than 2x the average in both halves (n={n})")
        out["tables"][f] = rows
    os.makedirs(VAULT, exist_ok=True)
    md = [f"# Lessons\n\n_Updated {datetime.now():%Y-%m-%d %H:%M} from {len(df)} trades · win rate {out['win']}% · average {base:+.1f} per trade per micro._\n",
          "## Proposals for the desk (never applied automatically)\n"] + [f"- {x}" for x in (out["flags"] or ["none yet: no condition passes the 50-trade, both-halves test"])]
    for f, rows in out["tables"].items():
        md.append(f"\n## By {f.replace('_', ' ')}\n\n| value | trades | win % | avg $ | shrunk $ | older half | newer half |\n|---|---|---|---|---|---|---|")
        md += [f"| {r['value']} | {r['n']} | {r['win%']} | {r['avg']} | {r['shrunk']} | {r['older']} | {r['newer']} |" for r in sorted(rows, key=lambda r: -r["shrunk"])]
    md.append("\n_Shrunk $ pulls small samples toward the overall average (30-trade prior), so a lucky streak can't masquerade as an edge._\n")
    open(os.path.join(VAULT, "Lessons.md"), "w").write("\n".join(md))
    json.dump(out, open(os.path.join(VAULT, "lessons.json"), "w"), indent=1, default=str)
    return out
