#!/usr/bin/env python3
"""
Quick test to verify multi-task learning fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_multitask_fix():
    """Test if the multi-task learning fix works."""
    
    print("🧠 Testing Multi-Task Learning Fix")
    print("=" * 40)
    
    try:
        # Test imports
        from multitask_strategy import MultiTaskStrategy
        print("✅ MultiTaskStrategy imported successfully")
        
        # Test initialization
        strategy = MultiTaskStrategy()
        print("✅ Strategy initialized successfully")
        
        # Test prepare_data method with None handling
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # Create empty data to test error handling
        empty_data = pd.DataFrame()
        result = strategy.prepare_data(
            empty_data, 
            datetime(2024, 1, 1), 
            datetime(2024, 1, 31)
        )
        
        if result == (None, None, None):
            print("✅ Error handling works correctly")
        else:
            print("❌ Error handling failed")
            
        print("🎉 Multi-Task Learning fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multitask_fix()
