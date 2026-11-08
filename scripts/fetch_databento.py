"""
Download gold futures 1-minute bars from Databento (CME Globex, dataset GLBX.MDP3).

  pip install databento
  set DATABENTO_API_KEY=db-xxxxxxxx        (Windows)   |   export DATABENTO_API_KEY=db-xxxxxxxx   (Mac)
  python scripts/fetch_databento.py --start 2021-01-01 --end 2026-10-01

Uses GC (full-size gold): same price as MGC, longer and cleaner history.
Continuous front-month symbol GC.c.0 rolls automatically.
ALWAYS run the cost estimate first; it prints the price before downloading anything.
"""
import argparse
import os

import databento as db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-10-01")
    ap.add_argument("--symbol", default="GC.c.0")
    ap.add_argument("--out", default="data/GC_1m.csv")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()

    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    q = dict(dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=[a.symbol], stype_in="continuous",
             start=a.start, end=a.end)
    cost = client.metadata.get_cost(**q)
    print(f"Estimated cost: ${cost:,.2f}")
    if not a.yes and input("Download? [y/N] ").strip().lower() != "y":
        return
    data = client.timeseries.get_range(**q)
    df = data.to_df()  # prices already scaled to dollars
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.reset_index()[["ts_event", "open", "high", "low", "close", "volume"]].to_csv(a.out, index=False)
    print(f"Saved {len(df):,} bars to {a.out}")


if __name__ == "__main__":
    main()
