#!/usr/bin/env python3
"""
Test script to measure 1Y/3M Ratio strategy execution time with CSV data.
"""

import time
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_csv_data():
    """Load data from CSV files in data_cache directory."""
    data_cache_dir = 'data_cache'
    ticker_data_grouped = {}
    
    print(f"Loading CSV data from {data_cache_dir}...")
    
    # Get first few CSV files for testing
    csv_files = [f for f in os.listdir(data_cache_dir) if f.endswith('.csv')][:100]  # Test with 100 files
    
    print(f"Loading {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        try:
            ticker = csv_file.replace('.csv', '')
            file_path = os.path.join(data_cache_dir, csv_file)
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime and set as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Keep only necessary columns
            if 'Close' in df.columns:
                df = df[['Close']]
            elif 'close' in df.columns:
                df = df[['close']].rename(columns={'close': 'Close'})
            
            ticker_data_grouped[ticker] = df
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(ticker_data_grouped)} tickers")
    return ticker_data_grouped

def test_strategy_timing():
    """Test 1Y/3M Ratio strategy timing with and without cache."""
    
    print("=" * 60)
    print("1Y/3M Ratio Strategy Timing Test (CSV Data)")
    print("=" * 60)
    
    try:
        # Import required modules
        from shared_strategies import select_1y_3m_ratio_stocks
        from parallel_backtest import build_price_history_cache, calculate_cached_performance
        
        # Load CSV data
        ticker_data_grouped = load_csv_data()
        
        if not ticker_data_grouped:
            print("ERROR: No data loaded. Please check data_cache directory.")
            return False
        
        # Use all loaded tickers for testing
        all_tickers = list(ticker_data_grouped.keys())
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
        price_history_cache = build_price_history_cache(ticker_data_grouped)
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
        
        if speedup < 2:  # Lower threshold for smaller dataset
            print(f"FAIL: Speedup should be at least 2x, got {speedup:.1f}x")
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
