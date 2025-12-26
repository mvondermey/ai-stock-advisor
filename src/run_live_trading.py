#!/usr/bin/env python3
"""
Live Trading Runner - Execute daily trades on Alpaca using your chosen strategy

Usage:
1. Configure your strategy in src/live_trading.py
2. Run: python src/run_live_trading.py (from project root)

Available Strategies:
- 'ai': AI Predictions (requires trained models)
- 'static_bh': Static Buy & Hold (top performers from backtest)
- 'dynamic_bh_1y': Dynamic BH rebalancing annually
- 'dynamic_bh_3m': Dynamic BH rebalancing quarterly
- 'dynamic_bh_1m': Dynamic BH rebalancing monthly
- 'risk_adj_mom': Risk-Adjusted Momentum
- 'mean_reversion': Mean Reversion
"""

import sys
from pathlib import Path

# Add current directory to path (assuming we're in project root)
sys.path.insert(0, str(Path.cwd()))

from src.live_trading import run_live_trading

if __name__ == "__main__":
    print(" Starting AI Stock Advisor Live Trading...")
    print("Make sure to configure your strategy in src/live_trading.py first!")
    print()

    try:
        run_live_trading()
        print("\n Live trading execution completed!")
    except KeyboardInterrupt:
        print("\n Live trading interrupted by user")
    except Exception as e:
        print(f"\n Live trading failed: {e}")
        import traceback
        traceback.print_exc()
