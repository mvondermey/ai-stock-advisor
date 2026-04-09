#!/usr/bin/env python3
"""
Test script to measure 1Y/3M Ratio strategy execution time with and without cache.
"""

import time
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def find_data_file():
    """Find the actual data file."""
    import glob
    possible_paths = [
        'data_cache/ticker_data.pkl',
        'data_cache/ticker_data.pickle',
        'ticker_data.pkl',
        'ticker_data.pickle',
        'data/ticker_data.pkl',
        'data/ticker_data.pickle'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try glob patterns
    patterns = [
        'data_cache/*.pkl',
        'data_cache/*.pickle', 
        '*.pkl',
        '*.pickle'
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def test_strategy_timing():
    """Test 1Y/3M Ratio strategy timing with and without cache."""
    
    print("=" * 60)
    print("1Y/3M Ratio Strategy Timing Test")
    print("=" * 60)
    
    try:
        # Import required modules
        from shared_strategies import select_1y_3m_ratio_stocks
        from parallel_backtest import build_price_history_cache, calculate_cached_performance
        import pandas as pd
        
        # Find data file
        data_file = find_data_file()
        if not data_file:
            print("ERROR: No data file found. Please ensure ticker data is available.")
            return False
        
        print(f"Using data file: {data_file}")
        
        # Load data
        print("Loading data...")
        ticker_data_grouped = pd.read_pickle(data_file)
        print(f"Loaded {len(ticker_data_grouped)} tickers")
        
        # Use subset for testing (to make test faster)
        all_tickers = list(ticker_data_grouped.keys())[:500]  # Test with 500 tickers
        print(f"Testing with {len(all_tickers)} tickers")
        
        # Test dates
        current_date = datetime(2026, 2, 16)
        print(f"Test date: {current_date.strftime('%Y-%m-%d')}")
        
        # Test 1: Without cache (original slow method)
        print("\n" + "-" * 50)
        print("TEST 1: Without Cache (Original Method)")
        print("-" * 50)
        
        start_time = time.time()
        selected_stocks_no_cache = select_1y_3m_ratio_stocks(
            all_tickers=all_tickers,
            ticker_data_grouped=ticker_data_grouped,
            current_date=current_date,
            top_n=10,
            price_history_cache=None  # No cache
        )
        time_no_cache = time.time() - start_time
        
        print(f"Execution time: {time_no_cache:.2f} seconds")
        print(f"Selected stocks: {selected_stocks_no_cache}")
        
        # Test 2: With cache (optimized method)
        print("\n" + "-" * 50)
        print("TEST 2: With Cache (Optimized Method)")
        print("-" * 50)
        
        # Build cache
        print("Building price history cache...")
        cache_start = time.time()
        price_history_cache = build_price_history_cache(
            ticker_data_grouped, 
            current_date, 
            lookback_days=365
        )
        cache_build_time = time.time() - cache_start
        print(f"Cache built in: {cache_build_time:.2f} seconds")
        
        # Run strategy with cache
        start_time = time.time()
        selected_stocks_with_cache = select_1y_3m_ratio_stocks(
            all_tickers=all_tickers,
            ticker_data_grouped=ticker_data_grouped,
            current_date=current_date,
            top_n=10,
            price_history_cache=price_history_cache  # With cache
        )
        time_with_cache = time.time() - start_time
        
        print(f"Execution time: {time_with_cache:.2f} seconds")
        print(f"Selected stocks: {selected_stocks_with_cache}")
        
        # Results analysis
        print("\n" + "=" * 60)
        print("TIMING ANALYSIS")
        print("=" * 60)
        
        # Calculate speedup
        if time_with_cache > 0:
            speedup = time_no_cache / time_with_cache
        else:
            speedup = float('inf')
        
        # Check if results match
        results_match = set(selected_stocks_no_cache) == set(selected_stocks_with_cache)
        
        print(f"Without cache: {time_no_cache:.2f}s")
        print(f"With cache:    {time_with_cache:.2f}s")
        print(f"Cache build:   {cache_build_time:.2f}s")
        print(f"Speedup:       {speedup:.1f}x")
        print(f"Results match: {'YES' if results_match else 'NO'}")
        
        # Total time comparison
        print(f"\nTotal time comparison:")
        print(f"  Without cache (2 days): {time_no_cache * 2:.2f}s")
        print(f"  With cache (2 days):    {cache_build_time + (time_with_cache * 2):.2f}s")
        
        if time_no_cache > 0:
            net_benefit = (time_no_cache * 2) - (cache_build_time + (time_with_cache * 2))
            print(f"  Net benefit (2 days):  {net_benefit:.2f}s")
        
        # Success criteria
        print(f"\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        
        success = True
        
        if speedup < 5:
            print(f"FAIL: Speedup should be at least 5x, got {speedup:.1f}x")
            success = False
        else:
            print(f"PASS: Speedup achieved: {speedup:.1f}x")
        
        if not results_match:
            print(f"FAIL: Results don't match between cached and non-cached versions")
            success = False
        else:
            print(f"PASS: Results match between cached and non-cached versions")
        
        if time_with_cache > 1.0:
            print(f"WARNING: Cached version still slow ({time_with_cache:.2f}s)")
        
        # Overall assessment
        print(f"\nOverall: {'SUCCESS' if success else 'FAILURE'}")
        
        if success:
            print(f"\nOptimization working correctly!")
            print(f"The 1Y/3M Ratio strategy should now be much faster in backtesting.")
        
        return success
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_strategy_timing()
    sys.exit(0 if success else 1)
