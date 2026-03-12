"""
AI Elite Market-Up Only Strategy
Same as AI Elite but only rebalances when market is up.
Uses SPY (or equal-weighted average) market return over last 5 days.
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd


def select_ai_elite_market_up_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    per_ticker_models: Dict[str, any] = None
) -> List[str]:
    """Select stocks using AI Elite scoring, but only when market is up."""
    from ai_elite_strategy import _calculate_market_return, select_ai_elite_stocks

    # Check market direction (5-day return)
    market_return = _calculate_market_return(ticker_data_grouped, current_date, 5)
    
    if market_return is None:
        # On first day or when market data unavailable, assume market is up to allow initial investment
        print(f"   📊 AI Elite Market-Up: Market data unavailable, allowing initial investment")
        market_return = 1.0  # Assume slightly positive to proceed
    
    if market_return <= 0:
        print(f"   📊 AI Elite Market-Up: Market is down ({market_return:.1f}%), skipping rebalance")
        return []  # Don't rebalance when market is down
    
    print(f"   📊 AI Elite Market-Up: Market is up ({market_return:.1f}%), proceeding with selection")

    # Delegate to AI Elite stock selection
    return select_ai_elite_stocks(
        all_tickers=all_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        per_ticker_models=per_ticker_models
    )
