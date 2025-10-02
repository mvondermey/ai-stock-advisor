# -*- coding: utf-8 -*-
"""
Performance Tracker for AI Stock Advisor Alpaca Trades
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

# Alpaca API credentials (set as environment variables for security)
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

def track_performance():
    """
    Connects to Alpaca, fetches open positions, and calculates
    the unrealized Profit & Loss for the portfolio.
    """
    print("🚀 Starting Performance Tracker...")

    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        print("⚠️ Alpaca API keys not set. Cannot track performance.")
        return

    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        print("✅ Alpaca Paper Trading Client initialized.")
    except Exception as e:
        print(f"⚠️ Error initializing Alpaca Trading Client: {e}.")
        return

    # Get current open positions from Alpaca
    try:
        positions = trading_client.get_all_positions()
        positions_map = {p.symbol: p for p in positions}
        print(f"✅ Found {len(positions_map)} open positions in Alpaca.")
    except APIError as e:
        print(f"❌ Could not fetch Alpaca positions: {e}")
        return

    # Get account details to include cash balance
    try:
        account = trading_client.get_account()
        cash_balance = float(account.cash)
        print(f"✅ Fetched account details. Cash balance: ${cash_balance:,.2f}")
    except APIError as e:
        print(f"❌ Could not fetch Alpaca account details: {e}")
        cash_balance = 0.0 # Default to 0 if fetch fails

    if not positions_map and cash_balance == 0:
        print("ℹ️ No open positions or cash balance to analyze.")
        return

    print("\n--- Live Portfolio Performance ---")
    print(f"{'Ticker':<10} | {'Qty':>10} | {'Entry Price':>12} | {'Current Price':>15} | {'Market Value':>15} | {'Unrealized P&L':>18} | {'Unrealized P&L %':>18}")
    print("-" * 110)

    total_market_value = 0.0
    total_unrealized_pl = 0.0
    total_cost_basis = 0.0

    for symbol, pos in positions_map.items():
        qty = float(pos.qty)
        market_value = float(pos.market_value)
        avg_entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        unrealized_pl = float(pos.unrealized_pl)
        cost_basis = avg_entry_price * qty
        unrealized_pl_pct = (unrealized_pl / cost_basis) * 100 if cost_basis != 0 else 0.0
        
        total_market_value += market_value
        total_unrealized_pl += unrealized_pl
        total_cost_basis += cost_basis

        print(f"{symbol:<10} | {qty:>10.2f} | ${avg_entry_price:>11.2f} | ${current_price:>14.2f} | ${market_value:>14,.2f} | ${unrealized_pl:>17,.2f} | {unrealized_pl_pct:>17.2f}%")

    print("-" * 110)
    
    total_unrealized_pl_pct = (total_unrealized_pl / total_cost_basis) * 100 if total_cost_basis != 0 else 0.0
    total_portfolio_value = total_market_value + cash_balance
    
    print("\n--- Portfolio Summary ---")
    print(f"Total Market Value of Positions: ${total_market_value:,.2f}")
    print(f"Cash Balance:                    ${cash_balance:,.2f}")
    print(f"Total Portfolio Value:           ${total_portfolio_value:,.2f}")
    print("-" * 40)
    print(f"Total Cost Basis:                ${total_cost_basis:,.2f}")
    print(f"Total Unrealized P&L:            ${total_unrealized_pl:,.2f} ({total_unrealized_pl_pct:.2f}%)")
    print("-" * 40)


if __name__ == "__main__":
    track_performance()
