#!/usr/bin/env python3
"""
Simple test to verify 1Y/3M Ratio strategy optimization.
"""

import time
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple():
    """Simple test of 1Y/3M Ratio strategy."""
    
    print("=" * 60)
    print("Simple 1Y/3M Ratio Strategy Test")
    print("=" * 60)
    
    try:
        # Import the strategy
        from shared_strategies import select_1y_3m_ratio_stocks
        from config import DATA_FILE, TOP_N_TICKERS
        import pandas as pd
        
        print(f"Loading data from {DATA_FILE}...")
        
        # Load data
        ticker_data_grouped = pd.read_pickle(DATA_FILE)
        print(f"Loaded {len(ticker_data_grouped)} tickers")
        
        # Get top tickers for testing
        all_tickers = list(ticker_data_grouped.keys())[:100]  # Test with 100 tickers
        print(f"Testing with {len(all_tickers)} tickers")
        
        # Test dates
        current_date = datetime(2026, 2, 16)
        
        print(f"\nTesting date: {current_date.strftime('%Y-%m-%d')}")
        
        # Test 1: Without cache
        print("\n" + "-" * 40)
        print("TEST 1: Without Cache")
        print("-" * 40)
        
        start_time = time.time()
        selected_stocks = select_1y_3m_ratio_stocks(
            all_tickers=all_tickers,
            ticker_data_grouped=ticker_data_grouped,
            current_date=current_date,
            top_n=10,
            price_history_cache=None  # No cache
        )
        elapsed_no_cache = time.time() - start_time
        
        print(f"Time without cache: {elapsed_no_cache:.2f}s")
        print(f"Selected: {selected_stocks}")
        
        # Test 2: With cache
        print("\n" + "-" * 40)
        print("TEST 2: With Cache")
        print("-" * 40)
        
        # Build cache
        from parallel_backtest import build_price_history_cache
        print("Building cache...")
        cache_start = time.time()
        price_history_cache = build_price_history_cache(
            ticker_data_grouped, 
            current_date, 
            lookback_days=365
        )
        cache_time = time.time() - cache_start
        print(f"Cache built in {cache_time:.2f}s")
        
        start_time = time.time()
        selected_stocks_cached = select_1y_3m_ratio_stocks(
            all_tickers=all_tickers,
            ticker_data_grouped=ticker_data_grouped,
            current_date=current_date,
            top_n=10,
            price_history_cache=price_history_cache  # With cache
        )
        elapsed_with_cache = time.time() - start_time
        
        print(f"Time with cache: {elapsed_with_cache:.2f}s")
        print(f"Selected: {selected_stocks_cached}")
        
        # Compare results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        match = set(selected_stocks) == set(selected_stocks_cached)
        speedup = elapsed_no_cache / elapsed_with_cache if elapsed_with_cache > 0 else float('inf')
        
        print(f"Without cache: {elapsed_no_cache:.2f}s")
        print(f"With cache:    {elapsed_with_cache:.2f}s")
        print(f"Cache build:   {cache_time:.2f}s")
        print(f"Speedup:       {speedup:.1f}x")
        print(f"Results match: {'YES' if match else 'NO'}")
        
        if not match:
            print(f"Without cache: {selected_stocks}")
            print(f"With cache:    {selected_stocks_cached}")
        
        if speedup > 5:
            print("SUCCESS: Significant speedup achieved!")
        else:
            print("WARNING: Speedup less than expected")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple()
    sys.exit(0 if success else 1)
