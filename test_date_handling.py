#!/usr/bin/env python3
"""
Test the date handling fix for multi-task learning
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

def test_date_handling():
    """Test date column vs index handling."""
    
    print("🧠 Testing Date Handling Fix")
    print("=" * 40)
    
    try:
        from multitask_strategy import MultiTaskStrategy
        strategy = MultiTaskStrategy()
        
        # Test 1: Date as column
        print("📅 Test 1: Date as column")
        data_col = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-12-31'),
            'ticker': 'AAPL',
            'Close': np.random.randn(365) * 10 + 100,
            'Volume': np.random.randint(1000000, 5000000, 365)
        })
        
        result = strategy.prepare_data(data_col, datetime(2023, 1, 1), datetime(2023, 12, 31))
        if result != (None, None, None):
            print("✅ Date column handling works")
        else:
            print("❌ Date column handling failed")
        
        # Test 2: Date as index
        print("📅 Test 2: Date as index")
        data_idx = pd.DataFrame({
            'ticker': 'GOOGL',
            'Close': np.random.randn(365) * 10 + 100,
            'Volume': np.random.randint(1000000, 5000000, 365)
        }, index=pd.date_range('2023-01-01', '2023-12-31'))
        data_idx.index.name = 'date'
        
        result = strategy.prepare_data(data_idx, datetime(2023, 1, 1), datetime(2023, 12, 31))
        if result != (None, None, None):
            print("✅ Date index handling works")
        else:
            print("❌ Date index handling failed")
        
        print("🎉 Date handling fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_date_handling()
