#!/usr/bin/env python3
"""
Test script for the 3M/1Y Ratio Strategy implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from shared_strategies import select_3m_1y_ratio_stocks

def create_test_data():
    """Create sample test data for testing the 3M/1Y ratio strategy."""
    
    # Create test data for 3 stocks over 400 days
    dates = pd.date_range(start='2023-01-01', periods=400, freq='D')
    
    # Stock 1: Strong 3M performance relative to 1Y (accelerating momentum)
    stock1_prices = np.concatenate([
        np.linspace(100, 110, 200),  # First 200 days: +10% (slow growth)
        np.linspace(110, 130, 200)   # Next 200 days: +18% (accelerating)
    ])
    
    # Stock 2: Poor 3M performance relative to 1Y (decelerating momentum)  
    stock2_prices = np.concatenate([
        np.linspace(100, 130, 200),  # First 200 days: +30% (strong growth)
        np.linspace(130, 132, 200)   # Next 200 days: +1.5% (slowing down)
    ])
    
    # Stock 3: Moderate consistent performance
    stock3_prices = np.linspace(100, 120, 400)  # +20% over entire period
    
    # Create DataFrames
    ticker_data = {}
    
    for i, (ticker, prices) in enumerate([('AAPL', stock1_prices), ('MSFT', stock2_prices), ('GOOGL', stock3_prices)]):
        df = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 400)
        })
        df.index = dates
        ticker_data[ticker] = df
    
    return ticker_data

def test_3m_1y_ratio_strategy():
    """Test the 3M/1Y ratio strategy selection logic."""
    
    print("🧪 Testing 3M/1Y Ratio Strategy...")
    
    # Create test data
    ticker_data = create_test_data()
    
    # Test parameters
    all_tickers = ['AAPL', 'MSFT', 'GOOGL']
    current_date = datetime(2024, 3, 15)  # About 74 days into the second period
    top_n = 2
    
    print(f"📊 Test Data:")
    print(f"   - Tickers: {all_tickers}")
    print(f"   - Current Date: {current_date.strftime('%Y-%m-%d')}")
    print(f"   - Top N: {top_n}")
    
    # Calculate expected results manually for verification
    print(f"\n🔍 Manual Calculation:")
    
    for ticker in all_tickers:
        data = ticker_data[ticker]
        
        # 3-month performance (last 90 days)
        three_month_start = current_date - timedelta(days=90)
        three_month_data = data[data.index >= three_month_start]
        three_month_perf = ((three_month_data['Close'].iloc[-1] / three_month_data['Close'].iloc[0]) - 1) * 100
        
        # 1-year performance (last 365 days)
        one_year_start = current_date - timedelta(days=365)
        one_year_data = data[data.index >= one_year_start]
        one_year_perf = ((one_year_data['Close'].iloc[-1] / one_year_data['Close'].iloc[0]) - 1) * 100
        
        ratio = three_month_perf / one_year_perf if one_year_perf != 0 else 0
        
        print(f"   {ticker}: 3M={three_month_perf:+.1f}%, 1Y={one_year_perf:+.1f}%, Ratio={ratio:.2f}")
    
    # Test the strategy
    print(f"\n🎯 Strategy Selection:")
    try:
        selected_stocks = select_3m_1y_ratio_stocks(
            all_tickers=all_tickers,
            ticker_data_grouped=ticker_data,
            current_date=current_date,
            top_n=top_n
        )
        
        print(f"   ✅ Selected stocks: {selected_stocks}")
        
        # Verify results
        if selected_stocks:
            # AAPL should have the highest ratio (accelerating momentum)
            expected_top = 'AAPL'
            if selected_stocks[0] == expected_top:
                print(f"   ✅ Test PASSED: Expected top stock {expected_top} was selected")
            else:
                print(f"   ❌ Test FAILED: Expected {expected_top}, got {selected_stocks[0]}")
        else:
            print(f"   ❌ Test FAILED: No stocks selected")
            
    except Exception as e:
        print(f"   ❌ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3m_1y_ratio_strategy()
