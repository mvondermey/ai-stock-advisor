#!/usr/bin/env python3
"""
Test Enhanced Volatility Trader Strategy
Run this to see how the new strategy performs with your current data.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def test_enhanced_strategy():
    """Test the enhanced volatility trader strategy."""
    try:
        print("Testing Enhanced Volatility Trader Strategy")
        print("=" * 60)
        
        # Import required modules
        from ticker_selection import get_all_tickers
        from data_utils import load_prices_robust
        from enhanced_volatility_trader import select_enhanced_volatility_stocks
        
        print("Getting stock universe...")
        all_tickers = get_all_tickers()
        print(f"   Found {len(all_tickers)} tickers")
        
        # Limit to smaller set for testing
        test_tickers = all_tickers[:100]  # Test with first 100 tickers
        print(f"   Testing with {len(test_tickers)} tickers")
        
        print("\nLoading price data...")
        current_date = datetime.now()
        train_start_date = current_date.replace(year=current_date.year - 1)
        
        all_tickers_data = load_prices_robust(
            test_tickers, 
            start_date=train_start_date,
            end_date=current_date
        )
        
        print(f"   Loaded data for {len(all_tickers_data.columns) // 5} tickers")
        
        print("\nRunning Enhanced Volatility Trader...")
        selected_stocks = select_enhanced_volatility_stocks(
            test_tickers,
            all_tickers_data,
            current_date=current_date,
            top_n=10
        )
        
        print(f"\nEnhanced Strategy Results:")
        print(f"   Selected {len(selected_stocks)} stocks:")
        for i, stock in enumerate(selected_stocks, 1):
            print(f"   {i:2d}. {stock}")
        
        print(f"\nEnhanced Volatility Trader test completed successfully!")
        print(f"To use in live trading: python src/main.py --live-trading --strategy enhanced_volatility")
        
    except Exception as e:
        print(f"Error testing enhanced strategy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_strategy()
