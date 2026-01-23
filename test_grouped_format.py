#!/usr/bin/env python3
"""
Test the fixed multi-task learning with grouped data format
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

def test_grouped_data_format():
    """Test multi-task learning with ticker_data_grouped format."""
    
    print("🧠 Testing Multi-Task Learning with Grouped Data Format")
    print("=" * 60)
    
    try:
        from multitask_strategy import MultiTaskStrategy
        
        # Create sample ticker_data_grouped format (like the real system)
        dates = pd.date_range('2023-01-01', '2023-12-31')
        
        ticker_data_grouped = {}
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        for ticker in tickers:
            # Create DataFrame with date as index (like real system)
            data = pd.DataFrame({
                'Close': np.random.randn(len(dates)) * 10 + 100,
                'High': np.random.randn(len(dates)) * 10 + 105,
                'Low': np.random.randn(len(dates)) * 10 + 95,
                'Open': np.random.randn(len(dates)) * 10 + 100,
                'Volume': np.random.randint(1000000, 5000000, len(dates)),
                'RSI': np.random.uniform(20, 80, len(dates)),
                'MACD': np.random.normal(0, 0.5, len(dates))
            })
            data.index = dates  # Date as index
            ticker_data_grouped[ticker] = data
        
        print(f"✅ Created sample ticker_data_grouped for {len(tickers)} tickers")
        print(f"📊 Sample AAPL data shape: {ticker_data_grouped['AAPL'].shape}")
        print(f"📅 AAPL index type: {type(ticker_data_grouped['AAPL'].index)}")
        print(f"📅 AAPL index name: {ticker_data_grouped['AAPL'].index.name}")
        print(f"📅 Sample AAPL dates: {ticker_data_grouped['AAPL'].index[:3].tolist()}")
        
        # Test strategy
        strategy = MultiTaskStrategy()
        
        # Test data preparation
        all_data_rows = []
        for ticker in tickers:
            ticker_data = ticker_data_grouped[ticker].copy()
            
            # Reset index to make date a column
            if hasattr(ticker_data.index, 'to_series'):
                ticker_data = ticker_data.reset_index()
                if 'index' in ticker_data.columns:
                    ticker_data = ticker_data.rename(columns={'index': 'date'})
            
            ticker_data['ticker'] = ticker
            all_data_rows.append(ticker_data)
        
        all_tickers_data = pd.concat(all_data_rows, ignore_index=True)
        
        print(f"📊 Combined data shape: {all_tickers_data.shape}")
        print(f"📊 Combined data columns: {list(all_tickers_data.columns)}")
        
        # Test prepare_data
        result = strategy.prepare_data(
            all_tickers_data, 
            datetime(2023, 1, 1), 
            datetime(2023, 12, 31)
        )
        
        if result != (None, None, None):
            X, ticker_ids, y = result
            print(f"✅ Data preparation successful: {X.shape}")
            print(f"🎯 Multi-Task Learning fix verified!")
            return True
        else:
            print("❌ Data preparation failed")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_grouped_data_format()
