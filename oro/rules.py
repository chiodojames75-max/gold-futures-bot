"""
Oro's locked rules. No settings screen, no overrides, no "just this once".

These numbers are fingerprinted. If anyone edits this file, the live engine sees that the fingerprint no longer
matches and refuses to start. Rules change only by an offline research run that re-tests them on unseen data,
and then the new fingerprint is written by that run, not by a person.
"""
import hashlib
import json

RULES = {
    "account": "Topstep 50K Combine",
    "strategy": "Night Shift v3 (overnight drift, every night except dead-quiet markets)",
    "markets": ["MGC"],
    "trend_filter": "none: the 200-day filter left gaps of up to 53 trading days; trading every night kept 2021-23 positive and cut the longest gap to 16",
    "entry_et": "18:05",                 # long at the start of the Topstep trading day
    "exit_et": "08:00",                  # out before the COMEX open and US data releases
    "protective_stop": "0.5 x average daily range below entry",
    "profit_target": "1.5 x average daily range above entry",
    "cool_down": "none",
    "quiet_filter": "skip the night when the last 5 sessions' average range is at or below 0.8x its 100-session average",
    "contracts_per_market": "floor(room above the max-loss line / $700), min 1, max 2",
    "max_contracts": 2,
    "brake_one_micro_usd": 1000,
    "brake_stand_down_usd": 500,
    "brake": "under $1,000 of room trade 1 micro; under $500 after a losing night, stand down (watch on paper) until the rule wins a night",
    "shorts": "off: every-night shorts lost $3,092 out of sample (40% win); the monthly learning loop can propose them per regime on paper",           # fixed size: no scaling up after wins, no doubling after losses
    "max_risk_per_contract_usd": 700,    # skip a night when one contract's stop would risk more than this
    "trade_cap": "none: every time the setup appears it is taken (re-entry after a stop was tested and lost money, so it is not part of the setup)",
    "max_loss_line_buffer_usd": 400,     # no new trades within this distance of the trailing max-loss line
    "flatten_et": "16:05",
    "every_trade_has_stop": True,
}

FINGERPRINT = "74b15cd70aeb97eb"


def fingerprint(rules: dict = RULES) -> str:
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:16]


def verify() -> None:
    fp = fingerprint()
    if fp != FINGERPRINT:
        raise SystemExit(
            f"Oro will not start: the locked rules were changed (fingerprint {fp}, expected {FINGERPRINT}).\n"
            "Rules only change through a research run that re-tests them. Restore oro/rules.py."
        )
