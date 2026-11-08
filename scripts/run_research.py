"""
Run the full research pass on a CSV of gold futures bars.

  python scripts/run_research.py data/GC_1m.csv
  python scripts/run_research.py --synthetic        # engine self-test on random data (no edge exists)

Writes research_report.json and per-strategy out-of-sample trade logs to ./results/.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oro.config import OroConfig  # noqa: E402
from oro.data import load_csv, prepare, synthetic_gold  # noqa: E402
from oro.validate import full_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", help="one or more CSV files (or a folder)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--oos", type=float, default=0.3, help="fraction of days held out for out-of-sample")
    a = ap.parse_args()
    cfg = OroConfig()

    files = []
    for c in a.csv:
        files += sorted(glob.glob(os.path.join(c, "*.csv"))) if os.path.isdir(c) else [c]
    raw = synthetic_gold() if a.synthetic else load_csv(files)
    df = prepare(raw, cfg.bar_minutes)
    print(f"Bars: {len(df):,}  Days: {df.trade_date.nunique()}  {df.index[0]} -> {df.index[-1]}")
    report = full_report(df, cfg, a.oos)

    os.makedirs("results", exist_ok=True)
    for name, r in report.items():
        r.pop("_oos_trades").to_csv(os.path.join("results", f"{name}_oos_trades.csv"), index=False)
    out = os.path.join("results", "research_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    for name, r in report.items():
        oos = r["out_of_sample"]
        print(f"\n== {name} ==  {r['verdict']}")
        print(f"   params {r['chosen_params(in-sample best)']}  | grid profitable in-sample: "
              f"{r['share_of_grid_profitable_in_sample']:.0%}")
        print(f"   OOS trades {oos.get('trades')}  win {oos.get('win_rate')}  PF {oos.get('profit_factor')}  "
              f"exp ${oos.get('expectancy_$')}  net ${oos.get('net_$')}  maxDD ${oos.get('max_drawdown_$')}")
        print(f"   90% CI of $ per trade: {r['expectancy_90pct_CI_$']}")
        print(f"   Combine MC: {r['combine_monte_carlo(out_of_sample days)']}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
