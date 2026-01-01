#!/usr/bin/env python3
"""
Live Trading Launcher
Simplified interface to run live trading with the best strategy from your backtest.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.live_trading import run_live_trading

if __name__ == "__main__":
    print("🚀 AI Stock Advisor - Live Trading")
    print("=" * 80)
    print()
    print("⚠️  IMPORTANT:")
    print("   1. First, complete a full backtest: python src/main.py")
    print("   2. Check which strategy performed best in output.log")
    print("   3. Edit src/live_trading.py to set LIVE_TRADING_STRATEGY")
    print("   4. Start with LIVE_TRADING_ENABLED=False for dry-run")
    print("   5. Then enable USE_PAPER_TRADING=True (fake money)")
    print("   6. Only go live after extensive paper trading testing!")
    print()
    print("=" * 80)
    print()
    
    response = input("Ready to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    print()
    run_live_trading()

