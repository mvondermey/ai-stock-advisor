#!/usr/bin/env python3
"""
Download missing S&P 400/600 ticker data with proper rate limiting.
Run this script to populate the cache before running the backtest.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

# Configuration
DATA_CACHE_DIR = Path("data_cache")
LOOKBACK_DAYS = 700  # Stay within Yahoo's 730-day limit
INTERVAL = "1h"
PAUSE_BETWEEN_CALLS = 0.3  # seconds between API calls

def get_missing_tickers():
    """Get list of tickers that need data download by fetching S&P 400/600 lists."""
    import requests
    from io import StringIO
    
    missing_tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Fetch S&P 400 MidCap
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        response = requests.get(url, headers=headers)
        table = pd.read_html(StringIO(response.text))[0]
        col = "Symbol" if "Symbol" in table.columns else table.columns[0]
        tickers = [s.replace('.', '-') for s in table[col].tolist()]
        missing_tickers.update(tickers)
        print(f"Fetched {len(tickers)} S&P 400 MidCap tickers")
    except Exception as e:
        print(f"Could not fetch S&P 400: {e}")
    
    # Fetch S&P 600 SmallCap
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        response = requests.get(url, headers=headers)
        table = pd.read_html(StringIO(response.text))[0]
        col = "Symbol" if "Symbol" in table.columns else table.columns[0]
        tickers = [s.replace('.', '-') for s in table[col].tolist()]
        missing_tickers.update(tickers)
        print(f"Fetched {len(tickers)} S&P 600 SmallCap tickers")
    except Exception as e:
        print(f"Could not fetch S&P 600: {e}")
    
    # Filter out tickers that already have cache files
    cached_tickers = set()
    if DATA_CACHE_DIR.exists():
        for f in DATA_CACHE_DIR.glob("*.csv"):
            cached_tickers.add(f.stem)
    
    tickers_to_download = missing_tickers - cached_tickers
    print(f"Already cached: {len(cached_tickers)}, need to download: {len(tickers_to_download)}")
    
    return sorted(list(tickers_to_download))

def download_ticker_data(ticker, start_date, end_date):
    """Download hourly data for a single ticker."""
    try:
        df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            interval=INTERVAL, 
            auto_adjust=True, 
            progress=False,
            multi_level_index=False
        )
        return df
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return pd.DataFrame()

def save_to_cache(ticker, df):
    """Save DataFrame to cache file."""
    if df.empty:
        return False
    
    # Ensure cache directory exists
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean up columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(col).capitalize() for col in df.columns]
    
    if "Close" not in df.columns and "Adj close" in df.columns:
        df = df.rename(columns={"Adj close": "Close"})
    
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0).astype(int)
    else:
        df["Volume"] = 0
    
    # Ensure timezone
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    # Save to CSV
    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
    df.to_csv(cache_file)
    return True

def clear_missing_markers():
    """Clear the _missing folder to allow re-download."""
    missing_dir = DATA_CACHE_DIR / "_missing"
    if missing_dir.exists():
        import shutil
        shutil.rmtree(missing_dir)
        print(f"Cleared {missing_dir}")

def main():
    print("=" * 60)
    print("Download Missing S&P 400/600 Ticker Data")
    print("=" * 60)
    
    # Clear missing markers first
    clear_missing_markers()
    
    # Calculate date range (700 days back from today)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    print(f"Date range: {start_date.date()} to {end_date.date()} ({LOOKBACK_DAYS} days)")
    print(f"Interval: {INTERVAL}")
    print()
    
    # Get list of tickers to download
    missing_tickers = get_missing_tickers()
    print(f"Found {len(missing_tickers)} tickers to download")
    print()
    
    # Download each ticker
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(missing_tickers, 1):
        # Check if already cached
        cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
        if cache_file.exists():
            print(f"[{i}/{len(missing_tickers)}] {ticker}: Already cached, skipping")
            success_count += 1
            continue
        
        print(f"[{i}/{len(missing_tickers)}] {ticker}: Downloading...", end=" ", flush=True)
        
        df = download_ticker_data(ticker, start_date, end_date)
        
        if not df.empty:
            if save_to_cache(ticker, df):
                print(f"OK ({len(df)} rows)")
                success_count += 1
            else:
                print("Failed to save")
                fail_count += 1
        else:
            print("No data")
            fail_count += 1
        
        # Rate limiting
        time.sleep(PAUSE_BETWEEN_CALLS)
    
    print()
    print("=" * 60)
    print(f"Download complete: {success_count} success, {fail_count} failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
