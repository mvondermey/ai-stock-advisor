#!/usr/bin/env python3
"""
Debug Enhanced Volatility Strategy Integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def debug_enhanced():
    """Debug the enhanced volatility integration."""
    try:
        print("=== DEBUGGING ENHANCED VOLATILITY INTEGRATION ===")
        
        # Check config
        import config
        print(f"1. ENABLE_ENHANCED_VOLATILITY = {config.ENABLE_ENHANCED_VOLATILITY}")
        
        # Check import
        from enhanced_volatility_trader import select_enhanced_volatility_stocks
        print("2. Enhanced volatility import: SUCCESS")
        
        # Check backtesting import
        from backtesting import ENABLE_ENHANCED_VOLATILITY as bt_enabled
        print(f"3. Backtesting ENABLE_ENHANCED_VOLATILITY = {bt_enabled}")
        
        # Check if function exists in backtesting
        import backtesting
        if hasattr(backtesting, '_rebalance_enhanced_volatility_portfolio'):
            print("4. Enhanced rebalancing function: EXISTS")
        else:
            print("4. Enhanced rebalancing function: MISSING")
            
        # Check if enhanced volatility is in strategy values list
        print("5. Checking strategy values integration...")
        
        # Try to run a minimal test
        print("6. Running minimal strategy test...")
        from shared_strategies import select_dynamic_bh_stocks
        
        # Create dummy data
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        ticker_data_grouped = {}
        
        # This should work without errors
        try:
            result = select_enhanced_volatility_stocks(
                test_tickers, 
                ticker_data_grouped, 
                current_date=None, 
                top_n=3
            )
            print(f"7. Strategy execution: SUCCESS (returned {len(result) if result else 0} stocks)")
        except Exception as e:
            print(f"7. Strategy execution: ERROR - {e}")
            
        print("\n=== DEBUG COMPLETE ===")
        
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_enhanced()
