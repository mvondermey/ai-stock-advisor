"""
Risk-Adjusted Momentum 1M Strategy
Identical to Risk-Adj Mom but uses 1-month (30-day) performance window instead of 1-year.
score = return_1m / sqrt(volatility)
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd


def select_risk_adj_mom_1m_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    lookback_days: int = 30,  # Allow override but default to 1M
) -> List[str]:
    """
    Select stocks using Risk-Adjusted Momentum with 1-month lookback.
    Delegates to shared parallel implementation for performance.
    """
    from shared_strategies import select_risk_adj_mom_stocks
    
    return select_risk_adj_mom_stocks(
        all_tickers=all_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        lookback_days=lookback_days
    )
