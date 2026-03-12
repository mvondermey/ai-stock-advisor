"""
Inverse ETF Hedge Strategy
Adds inverse ETFs to portfolios during market downturns instead of using stop losses
Uses calendar days for all calculations.
"""

from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta
from config import INVERSE_ETFS, PORTFOLIO_SIZE

def add_inverse_etf_hedge(
    current_stocks: List[str],
    market_conditions: Dict[str, float],
    hedge_percentage: float = 0.2,  # 20% of portfolio in hedge
    inverse_etf_preference: List[str] = ['SOXS', 'SQQQ', 'SPXU', 'FAZ', 'SH', 'PSQ']
) -> List[str]:
    """
    Add inverse ETFs to portfolio based on market conditions
    
    Args:
        current_stocks: Current selected stocks
        market_conditions: Dict with market metrics (e.g., {'sp500_3m': -0.15, 'nasdaq_3m': -0.20})
        hedge_percentage: What % of portfolio to allocate to inverse ETFs
        inverse_etf_preference: Preferred inverse ETFs in order
    
    Returns:
        Updated stock list with inverse ETFs
    """
    # Check if market is down significantly
    market_down = False
    market_decline = 0
    
    if 'sp500_3m' in market_conditions:
        market_decline = max(market_decline, abs(market_conditions['sp500_3m']))
    if 'nasdaq_3m' in market_conditions:
        market_decline = max(market_decline, abs(market_conditions['nasdaq_3m']))
    
    # Add hedge if market is down more than 10%
    if market_decline > 0.10:  # 10% market decline
        market_down = True
    
    if not market_down:
        return current_stocks
    
    # Calculate how many inverse ETFs to add
    num_hedge_positions = max(1, int(PORTFOLIO_SIZE * hedge_percentage))
    
    # Remove worst performing stocks to make room for hedge
    updated_stocks = current_stocks[:-num_hedge_positions] if len(current_stocks) > PORTFOLIO_SIZE - num_hedge_positions else current_stocks
    
    # Add preferred inverse ETFs that aren't already in portfolio
    for etf in inverse_etf_preference:
        if etf not in updated_stocks and len(updated_stocks) < PORTFOLIO_SIZE:
            updated_stocks.append(etf)
            print(f"   🛡️ Adding hedge {etf} (market down {market_decline:.1%})")
    
    return updated_stocks


def calculate_market_conditions(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None
) -> Dict[str, float]:
    """
    Calculate current market conditions using calendar days.
    
    Args:
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for calculation
        
    Returns:
        Dict with market condition metrics
    """
    conditions = {}
    
    if current_date is None:
        # Use latest date from data
        latest_dates = [data.index.max() for data in ticker_data_grouped.values() if len(data) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return conditions
    
    indices = {
        'SPY': 'sp500',
        'QQQ': 'nasdaq',
        'IWM': 'russell2000'
    }
    
    for ticker, name in indices.items():
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker]
            data = data[data.index <= current_date]
            
            # 3-month performance (90 calendar days)
            start_3m = current_date - timedelta(days=90)
            data_3m = data[data.index >= start_3m]
            if len(data_3m) >= 10:
                perf_3m = (data_3m['Close'].iloc[-1] / data_3m['Close'].iloc[0] - 1)
                conditions[f'{name}_3m'] = perf_3m
            
            # 1-month performance (30 calendar days)
            start_1m = current_date - timedelta(days=30)
            data_1m = data[data.index >= start_1m]
            if len(data_1m) >= 5:
                perf_1m = (data_1m['Close'].iloc[-1] / data_1m['Close'].iloc[0] - 1)
                conditions[f'{name}_1m'] = perf_1m
    
    return conditions
