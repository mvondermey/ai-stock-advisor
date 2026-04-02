#!/usr/bin/env python3
"""
Test script to check if Yahoo Finance provides 1h data for ETFs.
Run this in WSL: python3 test_etf_download.py
"""

import yfinance as yf
from datetime import datetime, timedelta

# ETFs that are failing to download
test_tickers = [
    # Sector ETFs
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB',
    'GDX', 'USO', 'TLT',
    # Inverse ETFs
    'SH', 'PSQ', 'DOG', 'RWM', 'SDS', 'QID', 'DXD', 'TWM',
    'SOXS', 'SQQQ', 'SPXU', 'FAZ', 'TZA', 'TECS',
    # Control - known working stock
    'AAPL', 'SPY'
]

end = datetime.now()
start = end - timedelta(days=30)

print("=" * 70)
print("Testing Yahoo Finance 1h data for ETFs")
print(f"Date range: {start.date()} to {end.date()}")
print("=" * 70)
print()

results = {'ok': [], 'empty': [], 'error': []}

for ticker in test_tickers:
    try:
        df = yf.download(ticker, start=start, end=end, interval='1h', progress=False)
        if df.empty:
            print(f"  {ticker:8s}: EMPTY (no 1h data from Yahoo)")
            results['empty'].append(ticker)
        else:
            print(f"  {ticker:8s}: OK - {len(df):4d} rows, {df.index[0].date()} to {df.index[-1].date()}")
            results['ok'].append(ticker)
    except Exception as e:
        print(f"  {ticker:8s}: ERROR - {str(e)[:50]}")
        results['error'].append(ticker)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  OK:    {len(results['ok']):3d} tickers - {', '.join(results['ok'][:10])}{'...' if len(results['ok']) > 10 else ''}")
print(f"  EMPTY: {len(results['empty']):3d} tickers - {', '.join(results['empty'][:10])}{'...' if len(results['empty']) > 10 else ''}")
print(f"  ERROR: {len(results['error']):3d} tickers - {', '.join(results['error'][:10])}{'...' if len(results['error']) > 10 else ''}")
print()

if results['empty']:
    print("CONCLUSION: Yahoo Finance does NOT provide 1h data for these ETFs.")
    print("Options:")
    print("  1. Use DATA_INTERVAL = '1d' (daily data) for the whole system")
    print("  2. Add a daily fallback for ETFs that don't have 1h data")
    print("  3. Remove these ETFs from Sector Rotation / Inverse ETF strategies")
else:
    print("CONCLUSION: All ETFs have 1h data available. Issue may be rate limiting or temporary.")
