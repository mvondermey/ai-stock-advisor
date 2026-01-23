#!/usr/bin/env python3
"""
Test Risk-Adjusted Momentum Ticker Selection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.backtesting import _get_current_risk_adj_mom_selections
from src.data_fetcher import load_prices_robust
from src.config import *  # Import all config settings

# Override config for testing
RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION = True
RISK_ADJ_MOM_MIN_CONFIRMATIONS = 2
RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION = True


def main():
    print("🧪 Testing Risk-Adjusted Momentum Selection")
    print("=" * 50)
    
    # Test with these tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Fetch data for these tickers
    print("📥 Downloading test data...")
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Create a mock all_tickers_data structure
    all_tickers_data = {}
    for ticker in test_tickers:
        data = load_prices_robust(ticker, start_date, end_date)
        if data is not None:
            all_tickers_data[ticker] = data
            print(f"   ✅ {ticker}: {len(data)} rows")
        else:
            print(f"   ❌ {ticker}: no data")
    
    # Run selection logic
    print("\n🔍 Running selection logic...")
    selected = _get_current_risk_adj_mom_selections(test_tickers, all_tickers_data)
    
    print("\n🎯 Selected tickers:", selected)
    print("=" * 50)

if __name__ == "__main__":
    main()
