#!/usr/bin/env python3
"""
Check if market is up for AI Elite Market-Up strategy
Uses the same logic as ai_elite_market_up_strategy.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta, timezone
import pandas as pd

def check_market_direction():
    """Check if market is up using AI Elite logic (5-day backward return)"""
    try:
        from data_utils import load_all_market_data
        import config

        # Get current date
        current_date = datetime.now(timezone.utc)

        # Load market data (cache only)
        print("Loading SPY data (cache only)...")

        # Temporarily disable downloads
        orig_download = config.ENABLE_DATA_DOWNLOAD
        config.ENABLE_DATA_DOWNLOAD = False

        try:
            try:
                spy_data = load_all_market_data(['SPY'], end_date=current_date, skip_download=True)
            except TypeError:
                spy_data = load_all_market_data(['SPY'], end_date=current_date)

            if spy_data is None or spy_data.empty or len(spy_data) < 10:
                print("❌ Error: SPY data unavailable or insufficient for 5-day return.")
                return
        finally:
            config.ENABLE_DATA_DOWNLOAD = orig_download

        # Calculate 5-day backward return from SPY
        # Set date as index if needed
        if 'date' in spy_data.columns:
            spy_data = spy_data.set_index('date')

        # Sort by date
        spy_data = spy_data.sort_index()

        # Get the last 6 trading days (need 6 to calculate 5-day return)
        if len(spy_data) < 6:
            print("❌ Error: Not enough SPY data for 5-day return calculation.")
            return

        recent_data = spy_data.tail(6)

        # Get Close prices
        close_col = recent_data['Close']
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]

        current_price = close_col.iloc[-1]
        price_5_days_ago = close_col.iloc[0]

        # Calculate 5-day return
        market_return = ((current_price - price_5_days_ago) / price_5_days_ago) * 100

        print(f"\n📊 SPY Price 5 days ago: ${price_5_days_ago:.2f}")
        print(f"📊 SPY Price today:      ${current_price:.2f}")
        print(f"📊 5-Day Market Return:  {market_return:+.2f}%")

        if market_return > 0:
            print("\n✅ Market is UP")
            print("→ AI Elite Market-Up would REBALANCE today")
        else:
            print("\n❌ Market is DOWN")
            print("→ AI Elite Market-Up would SKIP rebalancing today")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_market_direction()
