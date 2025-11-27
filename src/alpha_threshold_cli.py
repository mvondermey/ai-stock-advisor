# src/alpha_threshold_cli.py
"""
CLI to compute alpha-optimal probability threshold from CSVs or stdin.
CSV must contain columns: date, proba, future_ret, bench_future_ret
"""
from __future__ import annotations

import argparse
import sys
import pandas as pd
from pathlib import Path
from alpha_training import AlphaThresholdConfig, select_threshold_by_alpha

def load_df(path: str) -> pd.DataFrame:
    if path == "-":
        return pd.read_csv(sys.stdin)
    return pd.read_csv(path)

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV path or '-' for stdin")
    p.add_argument("--freq", default="D", choices=["D","W","M"])
    p.add_argument("--metric", default="alpha", choices=["alpha","active_ir"])
    p.add_argument("--costs-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--thresholds", default="0.10:0.90:0.02", help="start:stop:step")
    args = p.parse_args(argv)

    df = load_df(args.csv)
    required = {"date","proba","future_ret","bench_future_ret"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Missing columns: {sorted(missing)}", file=sys.stderr)
        return 2
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    proba = pd.Series(df["proba"].values, index=df["date"])
    fut = pd.Series(df["future_ret"].values, index=df["date"])
    bench = pd.Series(df["bench_future_ret"].values, index=df["date"])

    start, stop, step = [float(x) for x in args.thresholds.split(":")]
    import numpy as np
    cfg = AlphaThresholdConfig(
        thresholds=tuple(np.round(np.arange(start, stop + 1e-9, step), 4)),
        rebalance_freq=args.freq,
        costs_bps=args.costs_bps,
        slippage_bps=args.slippage_bps,
        metric=args.metric,
    )
    t, entries, score, diag = select_threshold_by_alpha(proba, fut, bench, cfg)
    print(f"best_threshold={t:.4f}, score={score:.6f}, metric={args.metric}")
    # Emit per-date entries for inspection
    out = df[["date"]].copy()
    out["entry"] = entries.values.astype(int)
    out.to_csv(sys.stdout, index=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
