#!/usr/bin/env python3
"""
Quick test to see what's happening with enhanced volatility
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def test_enhanced():
    """Test enhanced volatility selection."""
    try:
        print("=== TESTING ENHANCED VOLATILITY SELECTION ===")
        
        # Import
        from enhanced_volatility_trader import select_enhanced_volatility_stocks
        print("Import successful")
        
        # Test with empty data (like what might be happening in backtest)
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        ticker_data_grouped = {}  # Empty data - this might be the issue!
        
        from datetime import datetime
        current_date = datetime.now()
        
        result = select_enhanced_volatility_stocks(
            test_tickers, 
            ticker_data_grouped,
            current_date=current_date,
            top_n=3
        )
        
        print(f"Selection result: {result}")
        print(f"Number of stocks selected: {len(result)}")
        
        if len(result) == 0:
            print("NO STOCKS SELECTED - This explains the $209 value!")
            print("   The strategy never buys anything, so portfolio value stays at initial cash")
            print("   But something must be draining the cash to $209...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced()
