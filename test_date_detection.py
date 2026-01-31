#!/usr/bin/env python3
"""
Force restart and test multi-task learning
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Clear any cached modules
if 'multitask_strategy' in sys.modules:
    del sys.modules['multitask_strategy']
if 'shared_strategies' in sys.modules:
    del sys.modules['shared_strategies']

def test_date_detection():
    """Test the enhanced date detection."""
    
    print("🔄 Testing Enhanced Date Detection")
    print("=" * 50)
    
    try:
        from multitask_strategy import MultiTaskStrategy
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        strategy = MultiTaskStrategy()
        
        # Create test data with date as index (like your real data)
        dates = pd.date_range('2023-01-01', '2023-12-31')
        test_data = pd.DataFrame({
            'ticker': ['AAPL'] * len(dates),
            'Close': np.random.randn(len(dates)) * 10 + 100,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        })
        test_data.index = dates
        test_data.index.name = None  # This might be the issue - unnamed index
        
        print(f"📊 Test data created:")
        print(f"   Columns: {list(test_data.columns)}")
        print(f"   Index name: {test_data.index.name}")
        print(f"   Index type: {type(test_data.index)}")
        print(f"   Sample dates: {test_data.index[:3].tolist()}")
        
        # Test prepare_data
        result = strategy.prepare_data(test_data, datetime(2023, 1, 1), datetime(2023, 12, 31))
        
        if result == (None, None, None):
            print("❌ Date detection failed")
        else:
            print("✅ Date detection worked!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_date_detection()
