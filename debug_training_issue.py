#!/usr/bin/env python3
"""
Debug script to diagnose why 0 ticker model tasks are being generated.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from config import TRAIN_LOOKBACK_DAYS, N_TOP_TICKERS, BACKTEST_DAYS
from ticker_selection import get_all_tickers
from data_fetcher import _download_batch_robust

print("=" * 80)
print("🐛 TRAINING ISSUE DEBUG SCRIPT")
print("=" * 80)

# Step 1: Check config
print(f"\n📊 Configuration:")
print(f"  - N_TOP_TICKERS: {N_TOP_TICKERS}")
print(f"  - TRAIN_LOOKBACK_DAYS: {TRAIN_LOOKBACK_DAYS}")
print(f"  - BACKTEST_DAYS: {BACKTEST_DAYS}")

# Step 2: Get tickers
print(f"\n🔍 Fetching tickers...")
try:
    all_available_tickers = get_all_tickers()
    print(f"  ✅ Found {len(all_available_tickers)} tickers")
    print(f"  Sample tickers: {all_available_tickers[:10]}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Download data for a few tickers
print(f"\n📥 Downloading sample data (first 5 tickers)...")
sample_tickers = all_available_tickers[:5]
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=TRAIN_LOOKBACK_DAYS + BACKTEST_DAYS)

print(f"  Date range: {start_date.date()} to {end_date.date()}")
print(f"  Tickers: {sample_tickers}")

try:
    all_tickers_data_list = _download_batch_robust(sample_tickers, start_date, end_date)
    
    if not all_tickers_data_list:
        print("  ❌ No data downloaded!")
        sys.exit(1)
    
    all_tickers_data = pd.concat(all_tickers_data_list, axis=0, ignore_index=True)
    print(f"  ✅ Downloaded {len(all_tickers_data)} rows")
    print(f"\n📊 Data structure:")
    print(f"  - Shape: {all_tickers_data.shape}")
    print(f"  - Columns: {list(all_tickers_data.columns)}")
    print(f"  - Data types:\n{all_tickers_data.dtypes}")
    
    # Check date column
    if 'date' in all_tickers_data.columns:
        print(f"\n📅 Date column analysis:")
        print(f"  - Type: {all_tickers_data['date'].dtype}")
        print(f"  - Sample: {all_tickers_data['date'].iloc[0]}")
        print(f"  - Min: {all_tickers_data['date'].min()}")
        print(f"  - Max: {all_tickers_data['date'].max()}")
        sample_date = all_tickers_data['date'].iloc[0]
        print(f"  - Has timezone: {hasattr(sample_date, 'tzinfo') and sample_date.tzinfo is not None}")
        if hasattr(sample_date, 'tzinfo'):
            print(f"  - Timezone: {sample_date.tzinfo}")
    
    # Check ticker column
    if 'ticker' in all_tickers_data.columns:
        print(f"\n🎯 Ticker column analysis:")
        unique_tickers = all_tickers_data['ticker'].unique()
        print(f"  - Unique tickers: {len(unique_tickers)}")
        print(f"  - Tickers: {list(unique_tickers)}")
        
        # Check data per ticker
        print(f"\n📈 Data per ticker:")
        for ticker in unique_tickers:
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
            print(f"  - {ticker}: {len(ticker_data)} rows, date range: {ticker_data['date'].min()} to {ticker_data['date'].max()}")
    
    # Step 4: Simulate the filtering that happens in parallel_training.py
    print(f"\n🧪 Simulating parallel_training.py filtering...")
    
    # Calculate train_start and train_end as they would be in the actual code
    bt_end = end_date
    bt_start = bt_end - timedelta(days=BACKTEST_DAYS)
    train_start = bt_start - timedelta(days=TRAIN_LOOKBACK_DAYS)
    train_end = bt_start - timedelta(days=1)
    
    print(f"  Training window: {train_start} to {train_end}")
    print(f"  Training window (dates): {train_start.date()} to {train_end.date()}")
    
    # Normalize dates like in the fix
    if 'date' in all_tickers_data.columns:
        sample_date = all_tickers_data['date'].iloc[0]
        print(f"\n  🔧 Date normalization:")
        print(f"    - DataFrame date has timezone: {hasattr(sample_date, 'tzinfo') and sample_date.tzinfo is not None}")
        print(f"    - train_start has timezone: {train_start.tzinfo is not None}")
        
        if hasattr(sample_date, 'tzinfo') and sample_date.tzinfo is not None:
            # DataFrame has timezone-aware dates
            if train_start.tzinfo is None:
                train_start = train_start.replace(tzinfo=sample_date.tzinfo)
            else:
                train_start = train_start.astimezone(sample_date.tzinfo)
            if train_end.tzinfo is None:
                train_end = train_end.replace(tzinfo=sample_date.tzinfo)
            else:
                train_end = train_end.astimezone(sample_date.tzinfo)
        elif sample_date is not None:
            # DataFrame has timezone-naive dates
            if train_start.tzinfo is not None:
                train_start = train_start.replace(tzinfo=None)
            if train_end.tzinfo is not None:
                train_end = train_end.replace(tzinfo=None)
        
        print(f"    - After normalization:")
        print(f"      - train_start: {train_start} (tz: {train_start.tzinfo})")
        print(f"      - train_end: {train_end} (tz: {train_end.tzinfo})")
        
        # Try filtering
        print(f"\n  🔬 Testing filter for each ticker:")
        for ticker in unique_tickers:
            df_filtered = all_tickers_data[
                (all_tickers_data['ticker'] == ticker) &
                (all_tickers_data['date'] >= train_start) &
                (all_tickers_data['date'] <= train_end)
            ]
            print(f"    - {ticker}: {len(df_filtered)} rows (>= 50? {len(df_filtered) >= 50})")
            if len(df_filtered) == 0:
                # Debug why it's empty
                ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
                print(f"      ⚠️  Total rows for {ticker}: {len(ticker_data)}")
                print(f"      ⚠️  Date range in data: {ticker_data['date'].min()} to {ticker_data['date'].max()}")
                print(f"      ⚠️  Requested range: {train_start} to {train_end}")
                
                # Check if dates overlap
                data_min = ticker_data['date'].min()
                data_max = ticker_data['date'].max()
                if data_max < train_start:
                    print(f"      ❌ Data ends before training period starts!")
                elif data_min > train_end:
                    print(f"      ❌ Data starts after training period ends!")
                else:
                    print(f"      ❓ Data overlaps but filter returned empty - check date comparison")
                    # Try to understand the comparison
                    print(f"         Comparing {type(ticker_data['date'].iloc[0])} vs {type(train_start)}")
    
    print(f"\n✅ Debug complete!")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

