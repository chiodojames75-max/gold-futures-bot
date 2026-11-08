# Gold Futures Trading Bot

A rules-based trading bot for micro gold futures (MGC), with a research engine that tries to prove a strategy wrong before it ever trades.

**Status:** paper trading only, on a practice account. No real money. Every result below is a backtest, which is hypothetical.

## The idea

Most trading strategies that look good on a chart fall apart on new data. This project is built to catch that:

1. **Research:** test a rule on years of gold price history with deliberately pessimistic fills.
2. **Validate:** tune on 2021–2023, then judge on 2024–2026 data the strategy never saw.
3. **Paper trade:** run it on live prices with simulated fills before anything real.
4. **Live (not yet):** same code, same risk rules, real orders.

The bot never changes its own rules. A locked rules file is fingerprint-checked, and the engine refuses to run if it's been edited.

## Current strategy: Night Shift

Based on documented research on overnight drift in futures (Boyarchenko, Larsen & Whelan, New York Fed, 2023).

- **Entry:** buy MGC at 6:05 PM ET on weeknights, skipping unusually quiet markets.
- **Stop:** 0.5× the 20-day average daily range below entry.
- **Target:** 1.5× that range above entry.
- **Exit:** 8:00 AM ET if neither is hit.

### Backtest (1 micro contract, hypothetical)

| | Tuning period (2021–2023) | Unseen period (2024–Oct 2026) |
|---|---|---|
| Net result | slightly negative | **+$26,909** |
| Win rate | | 59% |
| Profit factor | | 1.52 |
| Worst drawdown | | −$3,920 |

The strategy lost a little in the tuning years and made money on recent data. That means the edge is recent and not guaranteed, which is why it's paper trading, not live.

### Strategies tested and rejected

Shorting every night, trading news releases (CPI, jobs, Fed), swing-low stops, and several opening-range and VWAP setups all failed on unseen data.

## Fill model (deliberately pessimistic)

- Entries fill at the next bar's open plus 1 tick.
- Stops fill with 1 tick of slippage, and at the open if price gaps through.
- Targets fill only if price trades 1 tick through.
- If the stop and target are both hit inside one bar, the stop wins.

## Risk limits

| Rule | Setting |
|---|---|
| Max stop risk | $700 per micro; the trade is skipped if the stop is farther |
| Trades per day | 1 |
| Size | 1–2 micros, cut to 1 near the account's loss limit |
| Kill switch | an empty file named `KILL` flattens all positions and stops the bot |
| Broker orders | every order carries a resting stop and target at the broker |

## Project structure

```
oro/
  backtest.py      Backtest engine and fill model
  strategies.py    Strategy rules, including NightShift
  validate.py      Out-of-sample checks and prop-firm evaluation simulation
  risk.py          Position sizing and loss limits
  rules.py         Locked rules (fingerprint-checked)
  data.py          Loading bars, sessions, indicators
  live/            Live and paper engine, TopstepX API client
scripts/           Research runs and data download
tests/             Engine correctness and no-look-ahead tests
results/           Exit, stop, and filter research tables
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q                              # 23 tests: fills, risk, no look-ahead
python scripts/run_research.py --synthetic       # sanity check: every strategy should fail on random data
```

Real data comes from Databento (CME Globex, `GC`, 1-minute bars). Broker credentials are read from environment variables (`TOPSTEPX_USERNAME`, `TOPSTEPX_API_KEY`) and are never stored in the code.

## How I built it

I build with AI. I set the goals, the risk rules, and what counted as passing, and directed the build in [Claude Code](https://claude.com/claude-code), which wrote the code. My part was deciding what to test, pushing back on results that looked too good, and cutting every strategy that failed on unseen data.

## Disclaimer

For research and education only. Not financial advice. Futures trading carries substantial risk of loss.

## License

Copyright © 2026. All rights reserved. See [LICENSE](LICENSE).
