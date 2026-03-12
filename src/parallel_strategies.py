"""
Parallel strategy execution module for hybrid parallelization.
Runs strategy stock selection in parallel while keeping daily loop sequential.
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
from multiprocessing import Pool, cpu_count


def calculate_momentum_scores(args):
    """Worker: Calculate momentum scores for a strategy."""
    ticker, ticker_df, current_date, lookback_days = args
    try:
        if ticker_df is None or len(ticker_df) < lookback_days:
            return None
        
        ticker_history = ticker_df[ticker_df.index <= current_date].tail(lookback_days + 10)
        if len(ticker_history) >= lookback_days:
            lookback_data = ticker_history.tail(lookback_days)
            start_price = lookback_data.iloc[0]['Close']
            end_price = lookback_data.iloc[-1]['Close']
            
            if start_price > 0:
                momentum_return = (end_price - start_price) / start_price
                return (ticker, momentum_return)
    except Exception:
        pass
    return None


def calculate_volatility_adjusted_momentum(args):
    """Worker: Calculate volatility-adjusted momentum score using calendar days."""
    ticker, ticker_df, current_date, lookback_days, vol_window = args
    try:
        from datetime import timedelta
        
        if ticker_df is None or len(ticker_df) < 30:
            return None
        
        # Filter data up to current_date
        ticker_history = ticker_df[ticker_df.index <= current_date]
        
        # Calculate momentum using calendar days
        start_date = current_date - timedelta(days=lookback_days)
        momentum_data = ticker_history[ticker_history.index >= start_date]
        
        if len(momentum_data) < 10:  # Need at least 10 trading days
            return None
        
        close_prices = momentum_data['Close'].dropna()
        if len(close_prices) < 2:
            return None
        
        start_price = close_prices.iloc[0]
        end_price = close_prices.iloc[-1]
        
        if start_price <= 0:
            return None
        
        momentum = (end_price - start_price) / start_price
        
        # Calculate volatility using calendar days
        vol_start_date = current_date - timedelta(days=vol_window)
        vol_data = ticker_history[ticker_history.index >= vol_start_date]
        returns = vol_data['Close'].pct_change().dropna()
        
        if len(returns) < 5:
            return None
        
        volatility = returns.std()
        
        if volatility > 0:
            vol_adj_score = momentum / volatility
            return (ticker, vol_adj_score, momentum, volatility)
    except Exception:
        pass
    return None


def calculate_performance_for_period(args):
    """Worker: Calculate performance over a specific period."""
    ticker, ticker_df, current_date, period_days = args
    try:
        if ticker_df is None:
            return None
        
        from datetime import timedelta
        perf_start_date = current_date - timedelta(days=period_days)
        perf_data = ticker_df.loc[perf_start_date:current_date]
        
        if len(perf_data) >= 50:
            valid_close = perf_data['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                
                if start_price > 0:
                    perf_pct = ((end_price - start_price) / start_price) * 100
                    return (ticker, perf_pct)
    except Exception:
        pass
    return None


def parallel_calculate_scores(tickers: List[str], 
                              ticker_data_grouped: Dict,
                              current_date: datetime,
                              score_function: str,
                              **kwargs) -> List[Tuple]:
    """
    Calculate scores for multiple tickers in parallel.
    
    Args:
        tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for calculations
        score_function: Name of scoring function ('momentum', 'vol_adj_momentum', 'performance')
        **kwargs: Additional parameters for scoring function
    
    Returns:
        List of (ticker, score, ...) tuples
    """
    # Prepare worker arguments
    worker_args = []
    for ticker in tickers:
        if ticker in ticker_data_grouped:
            ticker_df = ticker_data_grouped[ticker]
            
            if score_function == 'momentum':
                lookback_days = kwargs.get('lookback_days', 90)
                worker_args.append((ticker, ticker_df, current_date, lookback_days))
            
            elif score_function == 'vol_adj_momentum':
                lookback_days = kwargs.get('lookback_days', 90)
                vol_window = kwargs.get('vol_window', 60)
                worker_args.append((ticker, ticker_df, current_date, lookback_days, vol_window))
            
            elif score_function == 'performance':
                period_days = kwargs.get('period_days', 365)
                worker_args.append((ticker, ticker_df, current_date, period_days))
    
    if not worker_args:
        return []
    
    # Select worker function
    if score_function == 'momentum':
        worker_func = calculate_momentum_scores
    elif score_function == 'vol_adj_momentum':
        worker_func = calculate_volatility_adjusted_momentum
    elif score_function == 'performance':
        worker_func = calculate_performance_for_period
    else:
        raise ValueError(f"Unknown score function: {score_function}")
    
    # Run in parallel
    num_workers = min(cpu_count(), len(worker_args))
    with Pool(processes=num_workers) as pool:
        results = pool.map(worker_func, worker_args)
    
    # Filter out None results
    return [r for r in results if r is not None]
