#!/usr/bin/env python3
"""
Simple test to verify 1Y/3M Ratio strategy accepts cache parameter.
"""

import sys
import os
import inspect

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_cache_parameter():
    """Test if the function accepts price_history_cache parameter."""
    
    print("=" * 60)
    print("Testing 1Y/3M Ratio Strategy Cache Parameter")
    print("=" * 60)
    
    try:
        # Import the strategy
        from shared_strategies import select_1y_3m_ratio_stocks
        
        # Check function signature
        sig = inspect.signature(select_1y_3m_ratio_stocks)
        params = list(sig.parameters.keys())
        
        print(f"Function parameters: {params}")
        
        if 'price_history_cache' in params:
            print("SUCCESS: price_history_cache parameter found")
            
            # Check if it has a default value
            cache_param = sig.parameters['price_history_cache']
            print(f"Cache parameter default: {cache_param.default}")
            
            if cache_param.default is None:
                print("SUCCESS: Cache parameter defaults to None")
            else:
                print("WARNING: Cache parameter has non-None default")
                
            return True
        else:
            print("FAILURE: price_history_cache parameter not found")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import():
    """Test if we can import the required modules."""
    
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    try:
        from shared_strategies import select_1y_3m_ratio_stocks
        print("SUCCESS: select_1y_3m_ratio_stocks imported")
        
        from parallel_backtest import calculate_cached_performance, build_price_history_cache
        print("SUCCESS: parallel_backtest functions imported")
        
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    print("Testing 1Y/3M Ratio Strategy Optimization")
    print()
    
    # Test imports
    import_success = test_import()
    print()
    
    if import_success:
        # Test cache parameter
        cache_success = test_cache_parameter()
        print()
        
        if cache_success:
            print("All tests passed! The strategy should now use the cache.")
        else:
            print("Cache parameter test failed.")
    else:
        print("Import test failed.")
    
    sys.exit(0 if (import_success and cache_success) else 1)
