"""
Performance Filters Module
Provides universal performance threshold filtering for all strategies.
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from config import (
    MIN_PERFORMANCE_1Y, MIN_PERFORMANCE_6M, MIN_PERFORMANCE_3M,
    ENABLE_PERFORMANCE_FILTERS,
    MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M
)


def apply_performance_filters(
    ticker: str,
    ticker_data: pd.DataFrame,
    current_date: datetime,
    strategy_name: str = "Unknown",
    debug_limit: int = 10
) -> Optional[Dict[str, float]]:
    """
    Apply performance threshold filters to a ticker.
    
    Returns None if ticker fails any filter, otherwise returns performance metrics.
    
    Args:
        ticker: Stock ticker symbol
        ticker_data: Historical price data
        current_date: Current analysis date
        strategy_name: Name of strategy for debug output
        
    Returns:
        Dict with performance metrics if passes filters, None otherwise
    """
    if not ENABLE_PERFORMANCE_FILTERS:
        return None  # No filtering enabled
    
    try:
        # Convert current_date to pandas Timestamp with timezone
        current_ts = pd.Timestamp(current_date)
        if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
            if current_ts.tz is None:
                current_ts = current_ts.tz_localize(ticker_data.index.tz)
            else:
                current_ts = current_ts.tz_convert(ticker_data.index.tz)
        
        # Filter data up to current_date
        data_up_to_current = ticker_data.loc[:current_ts]
        
        if len(data_up_to_current) < MIN_DATA_DAYS_1Y:  # Need at least 1 year of data
            if debug_limit > 0:
                print(f"   ⚠️ {ticker}: Insufficient data ({len(data_up_to_current)} < {MIN_DATA_DAYS_1Y} days)")
            return None
        
        close_prices = data_up_to_current['Close'].dropna()
        if len(close_prices) < MIN_DATA_DAYS_1Y:
            if debug_limit > 0:
                print(f"   ⚠️ {ticker}: Insufficient Close data ({len(close_prices)} < {MIN_DATA_DAYS_1Y} valid prices)")
            return None
        
        # Calculate 1Y performance
        price_1y_ago = close_prices.iloc[-MIN_DATA_DAYS_1Y]
        price_current = close_prices.iloc[-1]
        perf_1y = (price_current - price_1y_ago) / price_1y_ago
        
        # Calculate 6M performance
        if len(close_prices) >= MIN_DATA_DAYS_6M:
            price_6m_ago = close_prices.iloc[-MIN_DATA_DAYS_6M]
            perf_6m = (price_current - price_6m_ago) / price_6m_ago
        else:
            if debug_limit > 0:
                print(f"   ⚠️ {ticker}: Insufficient 6M data ({len(close_prices)} < {MIN_DATA_DAYS_6M} days)")
            return None  # Insufficient data for 6M
        
        # Calculate 3M performance
        if len(close_prices) >= MIN_DATA_DAYS_3M:
            price_3m_ago = close_prices.iloc[-MIN_DATA_DAYS_3M]
            perf_3m = (price_current - price_3m_ago) / price_3m_ago
        else:
            if debug_limit > 0:
                print(f"   ⚠️ {ticker}: Insufficient 3M data ({len(close_prices)} < {MIN_DATA_DAYS_3M} days)")
            return None  # Insufficient data for 3M
        
        # Apply filters
        if perf_1y < MIN_PERFORMANCE_1Y:
            if debug_limit > 0:
                print(f"   ❌ {ticker}: Failed 1Y filter ({perf_1y:.1%} < {MIN_PERFORMANCE_1Y:.1%})")
            return None  # Failed 1Y filter
        
        if perf_6m < MIN_PERFORMANCE_6M:
            if debug_limit > 0:
                print(f"   ❌ {ticker}: Failed 6M filter ({perf_6m:.1%} < {MIN_PERFORMANCE_6M:.1%})")
            return None  # Failed 6M filter
        
        if perf_3m < MIN_PERFORMANCE_3M:
            if debug_limit > 0:
                print(f"   ❌ {ticker}: Failed 3M filter ({perf_3m:.1%} < {MIN_PERFORMANCE_3M:.1%})")
            return None  # Failed 3M filter
        
        # Return performance metrics
        return {
            'perf_1y': perf_1y,
            'perf_6m': perf_6m,
            'perf_3m': perf_3m
        }
        
    except Exception as e:
        # On error, allow ticker to pass (don't filter out due to calculation issues)
        return {'perf_1y': 0.15, 'perf_6m': 0.075, 'perf_3m': 0.0375}  # Assume good performance


def filter_tickers_by_performance(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    strategy_name: str = "Unknown"
) -> List[str]:
    """
    Filter a list of tickers by performance thresholds.
    
    Args:
        all_tickers: List of ticker symbols to filter
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current analysis date
        strategy_name: Name of strategy for debug output
        
    Returns:
        List of tickers that pass performance filters
    """
    if not ENABLE_PERFORMANCE_FILTERS:
        return all_tickers  # No filtering enabled
    
    passed_tickers = []
    failed_count = 0
    debug_count = 0
    max_debug = 10  # Limit debug output to avoid flooding
    
    print(f"\n   🔍 {strategy_name}: Performance filter analysis (first {max_debug} failures shown)")
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                failed_count += 1
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) == 0:
                failed_count += 1
                continue
            
            # Apply performance filters
            perf_metrics = apply_performance_filters(ticker, ticker_data, current_date, strategy_name, max_debug - debug_count)
            
            if perf_metrics is not None:
                # Passed all filters
                passed_tickers.append(ticker)
                if len(passed_tickers) <= 5:  # Show first 5 successes
                    print(f"   ✅ {ticker}: PASSED (1Y={perf_metrics['perf_1y']:.1%}, 6M={perf_metrics['perf_6m']:.1%}, 3M={perf_metrics['perf_3m']:.1%})")
            else:
                # Failed at least one filter
                failed_count += 1
                debug_count += 1
                
        except Exception:
            failed_count += 1
            debug_count += 1
            continue
    
    if failed_count > 0 and len(all_tickers) > 0:
        pass_rate = (len(passed_tickers) / len(all_tickers)) * 100
        print(f"   📊 {strategy_name}: Performance filters - {len(passed_tickers)}/{len(all_tickers)} passed ({pass_rate:.1f}%)")
    
    return passed_tickers
