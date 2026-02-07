"""
Test script to verify Yahoo Finance hourly data limits.
Checks how much historical data Yahoo actually provides for hourly intervals.
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

print("=" * 80)
print("YAHOO FINANCE HOURLY DATA TEST")
print("=" * 80)

# Test tickers
tickers = ['AAPL', 'WDC', 'SNDK', 'SPY']

# Calculate date ranges
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # Request 2 years of data

print(f"\n📅 Requesting data from {start_date.date()} to {end_date.date()}")
print(f"📅 Requested period: 730 days (2 years)")
print(f"📅 Yahoo limit for hourly: 729 days")
print()

for ticker in tickers:
    print(f"\n{'='*80}")
    print(f"Testing: {ticker}")
    print(f"{'='*80}")
    
    try:
        # Fetch hourly data
        print(f"\n⬇️  Fetching 1h data...")
        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)
        
        if data.empty:
            print(f"❌ No data returned")
            continue
            
        actual_start = data.index.min()
        actual_end = data.index.max()
        days_available = (actual_end - actual_start).days
        rows = len(data)
        
        print(f"✅ Data received:")
        print(f"   📊 Rows: {rows}")
        print(f"   📅 Date range: {actual_start.date()} to {actual_end.date()}")
        print(f"   📅 Days available: {days_available}")
        print(f"   📅 Days requested: 730")
        print(f"   📅 Days missing: {730 - days_available}")
        
        # Also test daily for comparison
        print(f"\n⬇️  Fetching 1d data (for comparison)...")
        data_daily = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        if not data_daily.empty:
            daily_start = data_daily.index.min()
            daily_end = data_daily.index.max()
            daily_days = (daily_end - daily_start).days
            print(f"   📅 Daily data range: {daily_start.date()} to {daily_end.date()} ({daily_days} days)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print("""
Yahoo Finance API behavior:
- Hourly (1h): Limited to ~729 days from today
- Daily (1d): Much longer history available (10+ years)

The actual amount of data returned depends on:
1. Ticker liquidity (less liquid = less data)
2. Corporate actions (splits, mergers)
3. API availability at request time

For backtesting with 1-year lookbacks, hourly data may be insufficient.
Daily data is recommended for strategies requiring >1 year of history.
""")
