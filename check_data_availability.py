#!/usr/bin/env python3
"""Check if problematic tickers have 1h vs daily data on Yahoo Finance."""
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Redirect all output to file
output_file = open("data_availability_results.txt", "w")
def log(msg):
    print(msg)
    output_file.write(msg + "\n")
    output_file.flush()

# Problematic tickers from output.log
problematic_tickers = [
    # ETFs
    "AGG", "BND", "DIA", "EEM", "EFA", "EWJ", "FAS", "FAZ", "GDX", "GLD",
    "FXI", "IWM", "QQQ", "SPY", "TLT", "VXX", "XLF", "XLE", "XLK",
    # Swiss
    "ACHI.SW", "AFP.SW", "BALN.SW", "CSGN.SW", "NESN.SW", "NOVN.SW",
    # Spanish
    "AENA.MC", "SAN.MC",
    # US stocks that failed
    "PLTR", "AAPL", "MSFT", "NVDA",
]

end = datetime.now()
start_1h = end - timedelta(days=30)  # 1h data limited to ~730 days
start_daily = end - timedelta(days=365)

results = []

for ticker in problematic_tickers:
    log(f"Checking {ticker}...")

    # Check 1h data
    try:
        df_1h = yf.download(ticker, start=start_1h, end=end, interval="1h", progress=False)
        rows_1h = len(df_1h) if df_1h is not None else 0
    except Exception as e:
        rows_1h = 0
        log(f"  1h error: {e}")

    # Check daily data
    try:
        df_1d = yf.download(ticker, start=start_daily, end=end, interval="1d", progress=False)
        rows_1d = len(df_1d) if df_1d is not None else 0
    except Exception as e:
        rows_1d = 0
        log(f"  1d error: {e}")

    results.append({
        "ticker": ticker,
        "1h_rows": rows_1h,
        "1d_rows": rows_1d,
        "has_1h": rows_1h > 0,
        "has_1d": rows_1d > 0,
    })
    log(f"  1h: {rows_1h} rows, 1d: {rows_1d} rows")

# Summary
log("\n" + "="*80)
log("SUMMARY")
log("="*80)
df = pd.DataFrame(results)
log(df.to_string(index=False))

log("\n--- Statistics ---")
log(f"Total tickers checked: {len(df)}")
log(f"Have 1h data: {df['has_1h'].sum()}")
log(f"Have 1d data: {df['has_1d'].sum()}")
log(f"Have 1d but NOT 1h: {((df['has_1d']) & (~df['has_1h'])).sum()}")

# Save to file
df.to_csv("data_availability_check.csv", index=False)
log("\nResults saved to data_availability_check.csv")
output_file.close()
