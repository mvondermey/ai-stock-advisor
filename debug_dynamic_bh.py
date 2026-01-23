#!/usr/bin/env python3
"""Debug script to test dynamic_bh_stocks function with sample data"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add src directory to path
project_root = Path(__file__).resolve().parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

def create_sample_data():
    """Create sample ticker data for testing"""
    print("Creating sample data...")
    
    # Create sample data for a few tickers
    dates = pd.date_range(start='2024-01-01', end='2025-01-08', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    all_data = []
    for ticker in tickers:
        # Generate random price data
        import random
        base_price = random.uniform(50, 200)
        
        for date in dates:
            # Simple price movement
            price_change = random.uniform(-0.05, 0.05)
            price = base_price * (1 + price_change)
            
            all_data.append({
                'date': date,
                'ticker': ticker,
                'Close': price,
                'High': price * 1.02,
                'Low': price * 0.98,
                'Volume': random.randint(1000000, 10000000)
            })
        
        base_price = random.uniform(50, 200)  # Reset base price for next ticker
    
    df = pd.DataFrame(all_data)
    df.set_index('date', inplace=True)
    print(f"Created sample data: {df.shape}")
    return df, tickers

def test_dynamic_bh():
    """Test the dynamic_bh_stocks function"""
    try:
        from shared_strategies import select_dynamic_bh_stocks
        
        # Create sample data
        all_tickers_data, tickers = create_sample_data()
        
        # Group data by ticker (same as live_trading.py does)
        ticker_data_grouped = {ticker: group for ticker, group in all_tickers_data.groupby('ticker')}
        
        print(f"Prepared {len(ticker_data_grouped)} ticker data groups")
        
        # Test the function
        current_date = datetime.now()
        selected = select_dynamic_bh_stocks(
            all_tickers=tickers,
            ticker_data_grouped=ticker_data_grouped,
            period='1y',
            current_date=current_date,
            top_n=3
        )
        
        print(f"Selected tickers: {selected}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamic_bh()
