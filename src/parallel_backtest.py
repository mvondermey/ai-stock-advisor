"""
Parallel performance calculations for backtesting optimization.
This module provides parallelized versions of performance calculation functions.
"""

from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime, timedelta


def calculate_single_ticker_performance(args):
    """Calculate performance for a single ticker - used for parallel processing."""
    ticker, ticker_data, current_date, train_start_date, period_days = args
    
    try:
        if ticker_data.empty:
            return None
            
        perf_start_date = max(train_start_date, current_date - timedelta(days=period_days))
        perf_data = ticker_data.loc[perf_start_date:current_date]
        
        if len(perf_data) >= 50:
            valid_close = perf_data['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                
                if not pd.isna(start_price) and not pd.isna(end_price) and start_price > 0:
                    perf_pct = ((end_price - start_price) / start_price) * 100
                    return (ticker, perf_pct)
    except Exception:
        pass
    
    return None


def calculate_parallel_performance(tickers: List[str], 
                                 ticker_data_grouped: Dict[str, pd.DataFrame],
                                 current_date: datetime,
                                 train_start_date: datetime,
                                 period_days: int = 365,
                                 num_processes: int = None) -> List[Tuple[str, float]]:
    """
    Calculate performance for multiple tickers in parallel.
    
    Args:
        tickers: List of ticker symbols
        ticker_data_grouped: Dict mapping ticker -> price data
        current_date: Current date for calculation
        train_start_date: Training start date constraint
        period_days: Performance window in days
        num_processes: Number of parallel processes (default: from config)
    
    Returns:
        List of (ticker, performance_pct) tuples
    """
    if num_processes is None:
        from config import NUM_PROCESSES
        num_processes = max(1, NUM_PROCESSES)
    
    # Prepare arguments for parallel processing
    args_list = []
    for ticker in tickers:
        ticker_data = ticker_data_grouped.get(ticker, pd.DataFrame())
        args_list.append((ticker, ticker_data, current_date, train_start_date, period_days))
    
    # Process in parallel
    import time
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_single_ticker_performance, args_list)
    elapsed = time.time() - start_time
    print(f"   ⏱️ Parallel processing: {len(tickers)} tickers in {elapsed:.2f}s ({num_processes} processes)")
    
    # Filter out None results and return valid performances
    performances = [r for r in results if r is not None]
    return performances


def calculate_parallel_risk_adjusted_scores(tickers: List[str],
                                           ticker_data_grouped: Dict[str, pd.DataFrame],
                                           current_date: datetime,
                                           train_start_date: datetime,
                                           num_processes: int = None) -> List[Tuple[str, float, float, float]]:
    """
    Calculate risk-adjusted momentum scores in parallel.
    
    Returns:
        List of (ticker, score, return_pct, volatility_pct) tuples
    """
    if num_processes is None:
        from config import NUM_PROCESSES
        num_processes = max(1, NUM_PROCESSES)
    
    def calculate_single_risk_adj(args):
        ticker, ticker_data, current_date, train_start_date = args
        
        try:
            if len(ticker_data) < 100:
                return None
                
            end_date = current_date or ticker_data.index.max()
            start_date = end_date - timedelta(days=365)  # Risk-adj uses 1-year
            if train_start_date:
                start_date = max(train_start_date, start_date)
            
            perf_data = ticker_data.loc[start_date:end_date]
            if len(perf_data) < 50:
                return None
            
            valid_close = perf_data['Close'].dropna()
            if len(valid_close) < 10:
                return None
            
            start_price = valid_close.iloc[0]
            end_price = valid_close.iloc[-1]
            
            if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
                return None
            
            # Calculate return and volatility
            basic_return = ((end_price - start_price) / start_price) * 100
            daily_returns = valid_close.pct_change().dropna()
            
            if len(daily_returns) <= 5:
                return None
            
            volatility = daily_returns.std() * 100
            risk_adj_score = basic_return / (volatility**0.5 + 0.001)
            
            return (ticker, risk_adj_score, basic_return, volatility)
            
        except Exception:
            return None
    
    # Prepare arguments
    args_list = []
    for ticker in tickers:
        ticker_data = ticker_data_grouped.get(ticker, pd.DataFrame())
        args_list.append((ticker, ticker_data, current_date, train_start_date))
    
    # Process in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_single_risk_adj, args_list)
    elapsed = time.time() - start_time
    print(f"   ⏱️ Parallel risk-adjusted: {len(tickers)} tickers in {elapsed:.2f}s ({num_processes} processes)")
    
    # Filter and return
    scores = [r for r in results if r is not None]
    return scores
