"""Desk wire for the live engine: public sources only, display only.

Pip reads these but never trades them. Release-time and Fed-decision setups were backtested
(scripts/news_test.py) and lost money out of sample, so news can't open, close or size a trade.
Sources: the Federal Reserve's public press-release RSS feed, plus the FOMC decision calendar.
No logins and no paid feeds.
"""
from __future__ import annotations
import time
import xml.etree.ElementTree as ET
from datetime import date
import requests

FED_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"
FOMC_2026 = ["2026-10-28", "2026-12-09"]  # VERIFY against federalreserve.gov/monetarypolicy/fomccalendars.htm each January

_cache = {"t": 0.0, "items": []}

def headlines(max_items: int = 8, ttl: int = 600) -> list[dict]:
    if time.time() - _cache["t"] < ttl:
        return _cache["items"]
    items = []
    try:
        r = requests.get(FED_RSS, timeout=8, headers={"User-Agent": "oro-desk-wire/1.0"})
        r.raise_for_status()
        for it in ET.fromstring(r.content).iter("item"):
            items.append({"source": "Federal Reserve", "title": (it.findtext("title") or "").strip(),
                          "time": (it.findtext("pubDate") or "").strip(), "link": (it.findtext("link") or "").strip()})
            if len(items) >= max_items:
                break
    except Exception as e:  # never let the news feed affect trading
        items = _cache["items"] or [{"source": "desk wire", "title": f"Fed feed unavailable ({type(e).__name__})", "time": "", "link": ""}]
    _cache.update(t=time.time(), items=items)
    return items

def calendar(today: date | None = None) -> list[dict]:
    today = today or date.today()
    return [{"date": d, "event": "Fed rate decision · 2:00 PM ET"} for d in FOMC_2026 if d >= today.isoformat()]
