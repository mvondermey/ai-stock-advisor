#!/usr/bin/env python3
"""
Test script to verify 1Y/3M Ratio strategy performance optimization.
Runs the strategy for 2 days and measures execution time with and without cache.
"""

import time
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_1y3m_ratio_performance():
    """Test 1Y/3M Ratio strategy performance with cache optimization."""
    
    print("=" * 60)
    print("Testing 1Y/3M Ratio Strategy Performance Optimization")
    print("=" * 60)
    
    try:
        # Import required modules
        from live_trading import _prepare_ticker_data_grouped
        from shared_strategies import select_1y_3m_ratio_stocks
        from parallel_backtest import build_price_history_cache
        from config import DATA_SOURCE, DATA_FILE, TOP_N_TICKERS
        
        print(f"Loading data from {DATA_FILE}...")
        
        # Load data
        if DATA_SOURCE == 'local':
            ticker_data_grouped = pd.read_pickle(DATA_FILE)
        else:
            raise ValueError("Only local data source supported for this test")
        
        print(f"Loaded {len(ticker_data_grouped)} tickers")
        
        # Get top tickers for testing
        all_tickers = list(ticker_data_grouped.keys())[:TOP_N_TICKERS]
        print(f"Testing with top {len(all_tickers)} tickers")
        
        # Prepare ticker data grouped (same as backtesting does)
        ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, ticker_data_grouped)
        
        # Test dates (2 consecutive days)
        end_date = datetime(2026, 2, 16)
        test_dates = [
            end_date - timedelta(days=1),  # Day 1
            end_date                           # Day 2
        ]
        
        print(f"\nTesting dates: {[d.strftime('%Y-%m-%d') for d in test_dates]}")
        
        # Test 1: Without cache (original slow method)
        print("\n" + "-" * 40)
        print("TEST 1: Without Cache (Original Method)")
        print("-" * 40)
        
        total_time_without_cache = 0
        results_without_cache = []
        
        for i, current_date in enumerate(test_dates, 1):
            print(f"\nDay {i}: {current_date.strftime('%Y-%m-%d')}")
            
            start_time = time.time()
            
            # Call strategy without cache
            selected_stocks = select_1y_3m_ratio_stocks(
                all_tickers=all_tickers,
                ticker_data_grouped=ticker_data_grouped,
                current_date=current_date,
                top_n=10,
                price_history_cache=None  # No cache
            )
            
            elapsed = time.time() - start_time
            total_time_without_cache += elapsed
            results_without_cache.append(selected_stocks)
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Selected: {selected_stocks[:5]}{'...' if len(selected_stocks) > 5 else ''}")
        
        avg_time_without_cache = total_time_without_cache / len(test_dates)
        print(f"\nAverage time without cache: {avg_time_without_cache:.2f}s")
        
        # Test 2: With cache (optimized method)
        print("\n" + "-" * 40)
        print("TEST 2: With Cache (Optimized Method)")
        print("-" * 40)
        
        # Build cache once
        print("Building price history cache...")
        cache_start = time.time()
        price_history_cache = build_price_history_cache(
            ticker_data_grouped, 
            end_date, 
            lookback_days=365
        )
        cache_time = time.time() - cache_start
        print(f"Cache built in {cache_time:.2f}s")
        
        total_time_with_cache = 0
        results_with_cache = []
        
        for i, current_date in enumerate(test_dates, 1):
            print(f"\nDay {i}: {current_date.strftime('%Y-%m-%d')}")
            
            start_time = time.time()
            
            # Call strategy with cache
            selected_stocks = select_1y_3m_ratio_stocks(
                all_tickers=all_tickers,
                ticker_data_grouped=ticker_data_grouped,
                current_date=current_date,
                top_n=10,
                price_history_cache=price_history_cache  # With cache
            )
            
            elapsed = time.time() - start_time
            total_time_with_cache += elapsed
            results_with_cache.append(selected_stocks)
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Selected: {selected_stocks[:5]}{'...' if len(selected_stocks) > 5 else ''}")
        
        avg_time_with_cache = total_time_with_cache / len(test_dates)
        print(f"\nAverage time with cache: {avg_time_with_cache:.2f}s")
        
        # Results comparison
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        
        speedup = avg_time_without_cache / avg_time_with_cache if avg_time_with_cache > 0 else float('inf')
        
        print(f"Without cache: {avg_time_without_cache:.2f}s per day")
        print(f"With cache:    {avg_time_with_cache:.2f}s per day")
        print(f"Speedup:       {speedup:.1f}x faster")
        
        # Verify results are similar
        print(f"\nResults verification:")
        for i, (without_cache, with_cache) in enumerate(zip(results_without_cache, results_with_cache), 1):
            match = set(without_cache) == set(with_cache)
            print(f"  Day {i}: {'MATCH' if match else 'MISMATCH'}")
            if not match:
                print(f"    Without cache: {without_cache}")
                print(f"    With cache:    {with_cache}")
        
        print(f"\nCache overhead: {cache_time:.2f}s (one-time cost)")
        print(f"Total time (2 days):")
        print(f"  Without cache: {total_time_without_cache:.2f}s")
        print(f"  With cache:    {cache_time + total_time_with_cache:.2f}s")
        
        if total_time_without_cache > 0:
            net_benefit = total_time_without_cache - (cache_time + total_time_with_cache)
            print(f"  Net benefit:   {net_benefit:.2f}s ({'positive' if net_benefit > 0 else 'negative'})")
        
        # Success criteria
        print("\n" + "=" * 60)
        print("SUCCESS CRITERIA")
        print("=" * 60)
        
        success = True
        
        if speedup < 5:
            print(f"{'FAIL'}: Speedup should be at least 5x, got {speedup:.1f}x")
            success = False
        else:
            print(f"{'PASS'}: Speedup achieved: {speedup:.1f}x")
        
        # Check if results match
        results_match = all(set(w) == set(c) for w, c in zip(results_without_cache, results_with_cache))
        if not results_match:
            print(f"{'FAIL'}: Results don't match between cached and non-cached versions")
            success = False
        else:
            print(f"{'PASS'}: Results match between cached and non-cached versions")
        
        if avg_time_with_cache > 1.0:
            print(f"{'WARNING'}: Cached version still slow ({avg_time_with_cache:.2f}s), may need further optimization")
        
        print(f"\nOverall: {'SUCCESS' if success else 'FAILURE'}")
        
        return success
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_1y3m_ratio_performance()
    sys.exit(0 if success else 1)
