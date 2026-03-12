"""
Risk-Adjusted Momentum 3M with Stops Strategy
Same as Risk-Adj Mom 3M but includes stop loss and take profit logic.
- Stop loss: 5% loss from entry price
- Take profit: 15% gain from entry price
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime


def select_risk_adj_mom_3m_with_stops_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
) -> List[str]:
    """Select stocks using Risk-Adj Mom 3M scoring (delegates to shared function)."""
    from shared_strategies import select_risk_adj_mom_stocks
    
    return select_risk_adj_mom_stocks(
        all_tickers=all_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        lookback_days=90,
        strategy_name="Risk-Adj Mom 3M with Stops"
    )


def check_risk_adj_mom_3m_stops(
    ticker: str,
    data: pd.DataFrame,
    entry_price: float,
    current_price: float,
    position_days: int,
) -> tuple[bool, str]:
    """
    Check if position should be closed based on stop loss or take profit.
    
    Returns:
        (should_close, reason)
    """
    STOP_LOSS_PCT = 5.0  # 5% stop loss
    TAKE_PROFIT_PCT = 15.0  # 15% take profit
    
    if current_price <= 0 or entry_price <= 0:
        return False, "Invalid prices"
    
    pnl_pct = (current_price - entry_price) / entry_price * 100
    
    # Stop loss: close if down 5% or more
    if pnl_pct <= -STOP_LOSS_PCT:
        return True, f"Stop loss triggered: {pnl_pct:.1f}% loss"
    
    # Take profit: close if up 15% or more
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, f"Take profit triggered: {pnl_pct:.1f}% gain"
    
    return False, "Hold"


def update_risk_adj_mom_3m_with_stops_positions(
    positions: Dict,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    transaction_cost: float,
) -> tuple[Dict, float, List[str]]:
    """
    Check and apply custom stops for Risk-Adj Mom 3M with Stops strategy.
    
    Returns:
        (updated_positions, transaction_costs, sold_tickers)
    """
    from backtesting import _last_valid_close_up_to
    
    total_costs = 0.0
    sold_tickers = []
    
    for ticker, pos_info in list(positions.items()):
        if pos_info.get('shares', 0) <= 0:
            continue
            
        # Get current price
        ticker_df = ticker_data_grouped.get(ticker)
        if ticker_df is None:
            continue
            
        current_price = _last_valid_close_up_to(ticker_df, current_date)
        if current_price is None:
            continue
        
        # Get entry price (stored as entry_price or avg_price)
        entry_price = pos_info.get('entry_price', pos_info.get('avg_price', 0))
        if entry_price <= 0:
            # Store entry price if not set
            pos_info['entry_price'] = current_price
            continue
        
        # Check if stop should be triggered
        should_close, reason = check_risk_adj_mom_3m_stops(
            ticker, ticker_df, entry_price, current_price, 
            pos_info.get('days_held', 0)
        )
        
        if should_close:
            # Sell the position
            shares = pos_info['shares']
            sale_value = shares * current_price
            cost = sale_value * transaction_cost
            net_value = sale_value - cost
            
            # Update position
            pos_info['shares'] = 0
            pos_info['value'] = 0
            pos_info['exit_price'] = current_price
            pos_info['exit_reason'] = reason
            pos_info['exit_date'] = current_date
            
            total_costs += cost
            sold_tickers.append((ticker, reason, net_value))
            
            print(f"   💰 RiskAdj 3M Stop: Selling {ticker}: {reason}")
    
    return positions, total_costs, sold_tickers
