"""Pip's mind: turns the trade notes in the Obsidian vault into reflection notes.

Writes vault/Mind/ (dashboard, wins, losses, mistakes, watch list, nightly journal).
Read-only with respect to trading: it never edits rules.py, never sizes, never orders.
Anything it notices is a *proposal* for James; changing a rule still needs his yes
plus an unseen-data backtest.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from . import memory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESK = os.path.join(ROOT, "desk")
MIND = os.path.join(memory.VAULT, "Mind")
UNSEEN_WEEK = 186.0      # avg $/week per micro on unseen 2024-26 backtest
UNSEEN_SD = 645.0        # weekly SD per micro
ALARM_3WK = -1676.0      # 3-week total below this (per micro) = 2 SD under expectation


def _w(name: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.join(MIND, name)), exist_ok=True)
    with open(os.path.join(MIND, name), "w") as f:
        f.write(text)


def _link(d) -> str:
    d = str(d)[:10]
    return f"[[{d}]]"


def _money(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"


def _streaks(pnl: pd.Series) -> tuple[int, int, int]:
    """(current streak signed, longest win streak, longest loss streak)"""
    cur = best_w = best_l = 0
    for p in pnl:
        if p > 0:
            cur = cur + 1 if cur > 0 else 1
        else:
            cur = cur - 1 if cur < 0 else -1
        best_w, best_l = max(best_w, cur), max(best_l, -cur)
    return cur, best_w, best_l


def _row(r) -> str:
    return (f"| {_link(r.date)} | {r.get('source', '')} | {_money(float(r.pnl))} | {r.get('r', '')}R | {r.get('exit_reason', '')} "
            f"| {r.get('trend', '')} | {r.get('vol_bucket', '')} | {r.get('weekday', '')} |")


HDR = "| Date | Source | P&L | R | Exit | Trend | Vol | Day |\n|---|---|---|---|---|---|---|---|"


CONCEPTS = {
    "Trend on": "Gold is above a **rising** 200-day average (Vector's trend method says yes).",
    "Trend off": "Gold is below a **falling** 200-day average: the downtrend regime.",
    "Trend mixed": "Price and the 200-day average disagree (above but falling, or below but rising).",
    "Trend n-a": "Not enough history yet to compute a 200-day average.",
    "Vol quiet": "Last 5 sessions' range is at or below 0.8× the 100-session average. Rule 1 **skips** these nights.",
    "Vol normal": "Last 5 sessions' range is 0.8–1.25× the 100-session average.",
    "Vol busy": "Last 5 sessions' range is above 1.25× the 100-session average.",
    "Vol n-a": "Not enough history to measure volatility.",
}


def _indexes(df: pd.DataFrame, lessons: dict, now: str) -> None:
    v = memory.VAULT
    # concept notes
    cdir = os.path.join(v, "Concepts"); os.makedirs(cdir, exist_ok=True)
    tables = lessons.get("tables") or {}
    for name, desc in CONCEPTS.items():
        field, val = ("trend", name[6:]) if name.startswith("Trend") else ("vol_bucket", name[4:])
        val = "n/a" if val == "n-a" else val
        row = next((r for r in tables.get(field, []) if r["value"] == val), None)
        sub = df[df.get(field, pd.Series(dtype=str)) == val] if field in df else df.head(0)
        stats = (f"| Trades | Win % | Avg $ | Older half | Newer half |\n|---|---|---|---|---|\n"
                 f"| {row['n']} | {row['win%']} | {row['avg']} | {row['older']} | {row['newer']} |" if row else "_No trades yet._")
        recent = "\n".join(f"- {_link(d)} {_money(p)}" for d, p in sub.sort_values("date").tail(10)[["date", "pnl"]].itertuples(index=False))
        open(os.path.join(cdir, f"{name}.md"), "w").write(
            f"---\ntype: concept\nupdated: {now}\n---\n# {name}\n\n{desc}\n\n## How Pip does in this condition\n\n{stats}\n\n"
            f"## Last 10 trades here\n\n{recent or '- none'}\n\nAll trades in this condition show up under **Backlinks** (right sidebar). "
            f"Up: [[Concepts]]\n")
    open(os.path.join(cdir, "Concepts.md"), "w").write(
        "---\ntype: index\n---\n# Concepts\n\nThe market conditions Pip records on every trade. Open one to see how the rule does there.\n\n"
        + "\n".join(f"- [[{n}]]: {d}" for n, d in CONCEPTS.items()) + "\n\nUp: [[Home]]\n")
    # trade indexes
    tdir = os.path.join(v, "Trades")
    years = sorted(d for d in os.listdir(tdir) if d.isdigit())
    for y in years:
        sub = df[df.date.str.startswith(y)].sort_values("date")
        rows = "\n".join(f"| {_link(r.date)} | {r.get('source', '')} | {_money(float(r.pnl))} | {r.get('exit_reason', '')} |" for _, r in sub.iterrows())
        open(os.path.join(tdir, y, f"{y}.md"), "w").write(
            f"---\ntype: index\n---\n# Trades · {y}\n\n{len(sub)} trades · {_money(float(sub.pnl.sum()))} per micro · "
            f"{(sub.pnl > 0).mean() * 100 if len(sub) else 0:.0f}% wins\n\n| Date | Source | P&L | Exit |\n|---|---|---|---|\n{rows}\n\nUp: [[Trades]]\n")
    yl = []
    for y in years:
        sub = df[df.date.str.startswith(y)]
        yl.append(f"- [[{y}]]: {len(sub)} trades, {_money(float(sub.pnl.sum()))} per micro")
    open(os.path.join(tdir, "Trades.md"), "w").write(
        "---\ntype: index\n---\n# Trades\n\nOne note per trade (backtest, paper, live). The engine writes them; Pip reflects on them in [[Pip's Mind]].\n\n"
        + "\n".join(yl) + "\n\nUp: [[Home]]\n")
    # journal index
    jdir = os.path.join(MIND, "Journal"); os.makedirs(jdir, exist_ok=True)
    js = sorted((f[:-3] for f in os.listdir(jdir) if f.endswith(" reflection.md")), reverse=True)
    open(os.path.join(jdir, "Journal.md"), "w").write(
        "---\ntype: index\n---\n# Journal\n\nOne reflection per paper/live night, newest first. Add your own note at the bottom of each.\n\n"
        + ("\n".join(f"- [[{j}]]" for j in js) or "- none yet: the first paper night is the next weeknight at 6:05 PM ET")
        + "\n\nUp: [[Pip's Mind]]\n")


def reflect() -> dict:
    df = memory._read_notes()
    if df.empty:
        return {}
    df["pnl"] = df["pnl"].astype(float)
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)
    src = df.get("source", pd.Series(["backtest"] * len(df)))
    real = df[src != "backtest"]
    bt = df[src == "backtest"]
    now = f"{datetime.now():%Y-%m-%d %H:%M}"
    lessons = {}
    lj = os.path.join(memory.VAULT, "lessons.json")
    if os.path.exists(lj):
        lessons = json.load(open(lj))

    # ---------- Wins / Losses ----------
    for name, sub, asc in [("Wins.md", df[df.pnl > 0], False), ("Losses.md", df[df.pnl <= 0], True)]:
        top = sub.sort_values("pnl", ascending=asc).head(15)
        rl = sub[sub.source != "backtest"].sort_values("date", ascending=False).head(20) if "source" in sub else sub.head(0)
        txt = [f"---\ntype: mind\nupdated: {now}\n---\n# {name[:-3]}\n",
               f"_{len(sub)} of {len(df)} trades · {len(rl)} recent paper/live shown first._\n",
               "## Paper / live (newest first)\n", HDR] + [_row(r) for _, r in rl.iterrows()]
        if rl.empty:
            txt.append("| — | none yet | | | | | | |")
        txt += [f"\n## {'Biggest' if not asc else 'Worst'} 15 ever (all sources)\n", HDR] + [_row(r) for _, r in top.iterrows()]
        if name == "Losses.md":
            _, _, ll = _streaks(df.pnl)
            roll = df.pnl.rolling(5).sum()
            i = int(roll.idxmin())
            txt += ["\n## Worst stretches\n", f"- Longest losing streak: **{ll} nights** in a row.",
                    f"- Worst 5-trade run: **{_money(roll.min())}** per micro, ending {_link(df.date[i])}."]
        txt.append("\nBack to [[Pip's Mind]].\n")
        _w(name, "\n".join(txt))

    # ---------- Mistakes ----------
    m = [f"---\ntype: mind\nupdated: {now}\n---\n# Mistakes\n",
         "> A **mistake** is a broken process, not a losing night. Losses are the cost of the rule; mistakes are what to fix.\n",
         "## Process mistakes (paper / live)\n",
         "Logged when a trade broke a rule: wrong time, wrong size, no resting stop/target, slippage > 2 ticks, traded during a stand-down or after KILL.\n"]
    flagged = []
    for _, r in real.iterrows():
        why = []
        if float(r.get("size", 1) or 1) > 2:
            why.append("size above 2 micros")
        if r.get("side") == "short":
            why.append("short (Rule 1 is long-only)")
        if r.get("exit_reason") not in ("target", "stop", "time exit"):
            why.append(f"unusual exit: {r.get('exit_reason')}")
        if why:
            flagged.append(f"- {_link(r.date)}: " + "; ".join(why))
    m += flagged or ["- none so far"]
    m += ["\n## Losing patterns (what the losers have in common)\n",
          "Conditions where losers cluster most, all sources. Below 50 trades or one-sided across history = noise, not a rule.\n",
          "| Condition | Losers | Share of all losers | Share of all trades |", "|---|---|---|---|"]
    losers = df[df.pnl <= 0]
    for f in ["trend", "vol_bucket", "weekday", "prev_day", "stretch"]:
        if f in df:
            vc_l, vc_a = losers[f].value_counts(normalize=True), df[f].value_counts(normalize=True)
            for v in vc_l.index[:2]:
                over = vc_l[v] - vc_a.get(v, 0)
                if over > 0.01:
                    m.append(f"| {f} = {v} | {int((losers[f] == v).sum())} | {vc_l[v]:.0%} | {vc_a.get(v, 0):.0%} |")
    m.append("\nBack to [[Pip's Mind]].\n")
    _w("Mistakes.md", "\n".join(m))

    # ---------- Watching (hypotheses) ----------
    wl = [f"---\ntype: mind\nupdated: {now}\n---\n# Watching\n",
          "Hunches Pip is tracking (helps/hurts = vs the average night). A hunch becomes a **proposal** only at 50+ trades with the same sign in both halves of history; "
          "it becomes a **rule** only with James's yes and an unseen-data backtest.\n",
          "| Hunch | Trades | Older half $ | Newer half $ | Status |", "|---|---|---|---|---|"]
    base = float(lessons.get("avg", df.pnl.mean()))
    hunches = []
    for f, rows in (lessons.get("tables") or {}).items():
        for r in rows:
            a, b, n = r.get("older"), r.get("newer"), r.get("n", 0)
            if a is None or b is None:
                continue
            same = (a - base > 0) == (b - base > 0)
            strong = abs(r["shrunk"] - base) > 10
            if strong:
                status = ("👀 consistent in both halves, not strong enough to propose yet" if n >= 50 and same
                          else "👀 consistent, needs 50+ trades" if same else "⚠️ flips between halves (likely noise)")
                verb = "helps" if r["shrunk"] > base else "hurts"
                wl.append(f"| {f.replace('_', ' ')} = {r['value']} {verb} | {n} | {a:+.0f} | {b:+.0f} | {status} |")
                hunches.append({"hunch": f"{f.replace('_', ' ')} = {r['value']} {verb}", "trades": int(n), "older": round(a, 1), "newer": round(b, 1),
                                "status": status, "rank": (0 if (n >= 50 and same) else 1 if same else 2, -abs(r["shrunk"] - base))})
    props = lessons.get("flags") or []
    wl += ["\n## Proposals waiting for James\n"] + ([f"- {p}" for p in props] or ["- none yet"]) + ["\nBack to [[Pip's Mind]].\n"]
    _w("Watching.md", "\n".join(wl))

    # ---------- Journal (one per paper/live night) ----------
    for _, r in real.iterrows():
        d = str(r.date)[:10]
        dj = os.path.join(DESK, "journal", f"{d}.md")
        call = ""
        if os.path.exists(dj):
            for line in open(dj):
                if line.startswith("| **Pip"):
                    call = line.split("|")[2].strip()
        exp = bt[bt.date == d]
        expect = _money(float(exp.pnl.iloc[0])) if len(exp) else "not in backtest"
        res = "win" if r.pnl > 0 else "loss"
        _w(os.path.join("Journal", f"{d} reflection.md"),
           f"---\ntype: reflection\ndate: {d}\nresult: {res}\npnl: {round(float(r.pnl), 2)}\n---\n"
           f"# {d} · what I learned\n\n"
           f"- **What the desk said:** {call or 'no desk entry'}\n"
           f"- **What happened:** {res}, {_money(float(r.pnl))} ({r.get('exit_reason')}), {r.get('r')}R · trade note {_link(d)}\n"
           f"- **Backtest expected:** {expect}\n"
           f"- **Conditions:** trend {r.get('trend')} · vol {r.get('vol_bucket')} · {r.get('weekday')}\n"
           f"- **Was it a mistake or just the cost of the rule?** {'Check [[Mistakes]].' if res == 'loss' else 'Not a mistake. Keep doing it.'}\n"
           f"- **James's note:** \n")

    # ---------- Dashboard ----------
    live = real if len(real) else bt
    label = "paper / live" if len(real) else "backtest (no paper trades yet)"
    cur, bw, bl = _streaks(live.pnl)
    last = live.tail(10)
    wk = pd.Series(dtype=float)
    if len(real):
        dts = pd.to_datetime(real.date)
        wk = real.pnl.groupby(dts.dt.to_period("W")).sum()
    three = float(wk.tail(3).sum()) if len(wk) else 0.0
    alarm = len(wk) >= 3 and three < ALARM_3WK * max(1.0, float(real["size"].astype(float).mean()) if len(real) and "size" in real else 1.0)
    dash = [f"---\ntype: mind\nupdated: {now}\n---\n# Pip's Mind\n",
            "_I'm Pip, Oro's trader. I trade one locked rule (Night Shift v3), write down every night, and look for patterns. "
            "I can propose changes. I can't make them: James decides, and only after a test on data I haven't seen._\n",
            "## Right now\n",
            "- **Mode:** paper on live prices (no real orders)",
            f"- **Trades on record:** {len(real)} paper/live · {len(bt)} backtest",
            f"- **Form ({label}):** last 10 = {_money(float(last.pnl.sum()))} · {int((last.pnl > 0).sum())}/{len(last)} wins · "
            f"current streak {'+' if cur > 0 else ''}{cur}",
            (f"- **Last 3 weeks vs expectation:** {_money(three)} vs about {_money(UNSEEN_WEEK * 3)} expected · alarm below {_money(ALARM_3WK)} per micro"
             if len(wk) else f"- **Last 3 weeks vs expectation:** starts after the first paper week (expect about {_money(UNSEEN_WEEK)}/week per micro; alarm if a 3-week total drops below {_money(ALARM_3WK)})")
            + (" · 🚨 **ALARM: stand down and review**" if alarm else ""),
            "\n## Last 10 nights\n", HDR] + [_row(r) for _, r in last.iloc[::-1].iterrows()]
    dash += ["\n## My notebook\n",
             "- [[Wins]]: what working looks like",
             "- [[Losses]]: the cost of the rule, worst streaks",
             "- [[Mistakes]]: broken process, and what losers have in common",
             "- [[Watching]]: hunches, and proposals waiting for James",
             "- [[Journal]]: one reflection per paper/live night (add your own note at the bottom)",
             "- [[Trades]] · [[Concepts]]: every trade, and the conditions they happened in",
             "- [[Lessons]]: the full condition tables",
             f"\n_Updated {now} after every trade. Nothing here changes how Oro trades._\n"]
    _w("Pip's Mind.md", "\n".join(dash))
    _indexes(df, lessons, now)
    # ---------- brain.json: a read-only summary the engine serves at /brain.json (never used for trading) ----------
    jdir = os.path.join(MIND, "Journal")
    js = sorted((f for f in os.listdir(jdir) if f.endswith(" reflection.md")), reverse=True) if os.path.isdir(jdir) else []
    last_ref = None
    if js:
        txt = open(os.path.join(jdir, js[0])).read()
        body = [ln[2:] for ln in txt.splitlines() if ln.startswith("- **")]
        last_ref = {"date": js[0][:10], "note": "Journal/" + js[0][:-3], "lines": body[:5]}
    brain = {
        "generated": now, "read_only": True,
        "mode_label": label,
        "trades": {"paper_live": int(len(real)), "backtest": int(len(bt))},
        "last10": [{"date": str(r.date)[:10], "pnl": round(float(r.pnl), 2), "exit": r.get("exit_reason"), "source": r.get("source")} for _, r in last.iloc[::-1].iterrows()],
        "form": {"last10_pnl": round(float(last.pnl.sum()), 2), "last10_wins": int((last.pnl > 0).sum()), "n": int(len(last))},
        "streak": int(cur), "longest_win_streak": int(bw), "longest_loss_streak": int(bl),
        "three_week": {"weeks": int(len(wk)), "total": round(three, 2), "expected": round(UNSEEN_WEEK * 3, 2), "per_week_expected": UNSEEN_WEEK,
                       "alarm_line": ALARM_3WK, "alarm": bool(alarm), "active": bool(len(wk) >= 3)},
        "watching": [{k: v for k, v in h.items() if k != "rank"} for h in sorted(hunches, key=lambda h: h["rank"])[:3]],
        "proposals": props,
        "mistakes": [m_[2:] for m_ in flagged][:5],
        "last_reflection": last_ref,
        "notes": {"mind": "Mind/Pip's Mind", "wins": "Mind/Wins", "losses": "Mind/Losses", "mistakes": "Mind/Mistakes", "watching": "Mind/Watching", "journal": "Mind/Journal/Journal"},
    }
    _w("brain.json", json.dumps(brain, indent=1, default=str))
    return {"trades": len(df), "real": len(real), "alarm": alarm}


if __name__ == "__main__":
    print(reflect())
