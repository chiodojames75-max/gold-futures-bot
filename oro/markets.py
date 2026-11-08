"""Markets Oro can trade on a Topstep account (micro contracts), with their specs."""
from .config import Contract

MARKETS = {
    #       micro symbol          tick size  $ per tick   session open used by open-range rules
    "GC": {"contract": Contract("MGC", 0.10, 1.00), "open_at": "08:20", "name": "Gold", "district": "Gold Quarter"},
    "SI": {"contract": Contract("SIL", 0.005, 5.00), "open_at": "08:25", "name": "Silver", "district": "Silver Works"},
    "NQ": {"contract": Contract("MNQ", 0.25, 0.50), "open_at": "09:30", "name": "Nasdaq", "district": "Nasdaq Heights"},
}
OPEN_AWARE = {"comex_orb", "orb_fade", "overnight_fade", "overnight_follow", "pdhl_break", "pdhl_fade"}
