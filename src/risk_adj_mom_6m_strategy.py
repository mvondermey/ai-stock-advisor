"""
Risk-Adjusted Momentum 6M Strategy
Identical to Risk-Adj Mom but uses 6-month (180-day) performance window instead of 1-year.
score = return_6m / sqrt(volatility)
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd


def select_risk_adj_mom_6m_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    lookback_days: int = 180,  # Allow override but default to 6M
) -> List[str]:
    """
    Select stocks using Risk-Adjusted Momentum with 6-month lookback.
    Delegates to shared parallel implementation for performance.
    """
    from shared_strategies import select_risk_adj_mom_stocks
    
    return select_risk_adj_mom_stocks(
        all_tickers=all_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        lookback_days=lookback_days,
        strategy_name="Risk-Adj Mom 6M"
    )
