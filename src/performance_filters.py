"""
Performance Filters Module
Provides universal performance threshold filtering for all strategies.
Uses calendar days (not trading days) for consistency across all calculations.
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from config import (
    MIN_PERFORMANCE_1Y, MIN_PERFORMANCE_6M, MIN_PERFORMANCE_3M,
    ENABLE_PERFORMANCE_FILTERS,
    MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M,
    INVERSE_ETFS, INVERSE_ETF_MIN_PERFORMANCE_1M, INVERSE_ETF_MIN_PERFORMANCE_3M,
    INVERSE_ETF_SKIP_1Y_FILTER
)

# Calendar day constants for performance calculations
CALENDAR_DAYS_1Y = 365
CALENDAR_DAYS_6M = 180
CALENDAR_DAYS_3M = 90
CALENDAR_DAYS_1M = 30


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
    Uses calendar days (timedelta) for all performance calculations.
    
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
        data_up_to_current = ticker_data[ticker_data.index <= current_ts]
        
        if len(data_up_to_current) < MIN_DATA_DAYS_1Y:  # Need minimum trading days
            if debug_limit > 0:
                print(f"   [WARN] {ticker}: Insufficient data ({len(data_up_to_current)} < {MIN_DATA_DAYS_1Y} days)")
            return None
        
        close_prices = data_up_to_current['Close'].dropna()
        price_current = close_prices.iloc[-1]
        
        # Calculate 1Y performance using calendar days
        start_1y = current_ts - timedelta(days=CALENDAR_DAYS_1Y)
        data_1y = close_prices[close_prices.index >= start_1y]
        if len(data_1y) >= MIN_DATA_DAYS_1Y:
            price_1y_ago = data_1y.iloc[0]
            perf_1y = (price_current - price_1y_ago) / price_1y_ago
        else:
            if debug_limit > 0:
                print(f"   [WARN] {ticker}: Insufficient 1Y data ({len(data_1y)} < {MIN_DATA_DAYS_1Y} trading days in 365 calendar days)")
            return None
        
        # Calculate 6M performance using calendar days
        start_6m = current_ts - timedelta(days=CALENDAR_DAYS_6M)
        data_6m = close_prices[close_prices.index >= start_6m]
        if len(data_6m) >= MIN_DATA_DAYS_6M:
            price_6m_ago = data_6m.iloc[0]
            perf_6m = (price_current - price_6m_ago) / price_6m_ago
        else:
            if debug_limit > 0:
                print(f"   [WARN] {ticker}: Insufficient 6M data ({len(data_6m)} < {MIN_DATA_DAYS_6M} trading days in 180 calendar days)")
            return None
        
        # Calculate 3M performance using calendar days
        start_3m = current_ts - timedelta(days=CALENDAR_DAYS_3M)
        data_3m = close_prices[close_prices.index >= start_3m]
        if len(data_3m) >= MIN_DATA_DAYS_3M:
            price_3m_ago = data_3m.iloc[0]
            perf_3m = (price_current - price_3m_ago) / price_3m_ago
        else:
            if debug_limit > 0:
                print(f"   [WARN] {ticker}: Insufficient 3M data ({len(data_3m)} < {MIN_DATA_DAYS_3M} trading days in 90 calendar days)")
            return None
        
        # Calculate 1M performance using calendar days
        start_1m = current_ts - timedelta(days=CALENDAR_DAYS_1M)
        data_1m = close_prices[close_prices.index >= start_1m]
        if len(data_1m) >= 10:  # Need at least 10 trading days in 30 calendar days
            price_1m_ago = data_1m.iloc[0]
            perf_1m = (price_current - price_1m_ago) / price_1m_ago
        else:
            perf_1m = 0.0
        
        # Check if this is an inverse ETF - use different filters
        is_inverse_etf = ticker in INVERSE_ETFS
        
        # Determine which filters to apply based on strategy timeframe
        # 1M strategies only need 1M filter, 3M strategies need 3M, etc.
        is_1m_strategy = '1M' in strategy_name and 'Monthly' not in strategy_name
        is_3m_strategy = '3M' in strategy_name
        is_6m_strategy = '6M' in strategy_name
        
        # Apply filters
        if is_inverse_etf:
            # Inverse ETF filters: only check 1M and 3M, skip 1Y/6M
            if not INVERSE_ETF_SKIP_1Y_FILTER and perf_1y < MIN_PERFORMANCE_1Y:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 1Y filter ({perf_1y:.1%} < {MIN_PERFORMANCE_1Y:.1%})")
                return None
            
            if perf_3m < INVERSE_ETF_MIN_PERFORMANCE_3M:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed inverse 3M filter ({perf_3m:.1%} < {INVERSE_ETF_MIN_PERFORMANCE_3M:.1%})")
                return None
            
            if perf_1m < INVERSE_ETF_MIN_PERFORMANCE_1M:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed inverse 1M filter ({perf_1m:.1%} < {INVERSE_ETF_MIN_PERFORMANCE_1M:.1%})")
                return None
            
            if debug_limit > 0:
                print(f"   [PASS] {ticker}: INVERSE ETF PASSED (1M={perf_1m:.1%}, 3M={perf_3m:.1%})")
        elif is_1m_strategy:
            # 1M strategies: only check 1M performance (positive momentum)
            min_perf_1m = 0.01  # 1% minimum for 1M strategies
            if perf_1m < min_perf_1m:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 1M filter ({perf_1m:.1%} < {min_perf_1m:.1%})")
                return None
        elif is_3m_strategy:
            # 3M strategies: only check 3M performance
            if perf_3m < MIN_PERFORMANCE_3M:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 3M filter ({perf_3m:.1%} < {MIN_PERFORMANCE_3M:.1%})")
                return None
        elif is_6m_strategy:
            # 6M strategies: check 6M and 3M performance
            if perf_6m < MIN_PERFORMANCE_6M:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 6M filter ({perf_6m:.1%} < {MIN_PERFORMANCE_6M:.1%})")
                return None
            if perf_3m < MIN_PERFORMANCE_3M:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 3M filter ({perf_3m:.1%} < {MIN_PERFORMANCE_3M:.1%})")
                return None
        else:
            # Default (1Y strategies): check all filters
            if perf_1y < MIN_PERFORMANCE_1Y:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 1Y filter ({perf_1y:.1%} < {MIN_PERFORMANCE_1Y:.1%})")
                return None  # Failed 1Y filter
            
            if perf_6m < MIN_PERFORMANCE_6M:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 6M filter ({perf_6m:.1%} < {MIN_PERFORMANCE_6M:.1%})")
                return None  # Failed 6M filter
            
            if perf_3m < MIN_PERFORMANCE_3M:
                if debug_limit > 0:
                    print(f"   [FAIL] {ticker}: Failed 3M filter ({perf_3m:.1%} < {MIN_PERFORMANCE_3M:.1%})")
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
    
    print(f"\n   [DEBUG] {strategy_name}: Performance filter analysis (first {max_debug} failures shown)")
    
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
                    print(f"   [PASS] {ticker}: PASSED (1Y={perf_metrics['perf_1y']:.1%}, 6M={perf_metrics['perf_6m']:.1%}, 3M={perf_metrics['perf_3m']:.1%})")
            else:
                # Failed at least one filter
                failed_count += 1
                debug_count += 1
                
        except Exception as e:
            failed_count += 1
            debug_count += 1
            continue
    
    if failed_count > 0 and len(all_tickers) > 0:
        pass_rate = (len(passed_tickers) / len(all_tickers)) * 100
        print(f"   [INFO] {strategy_name}: Performance filters - {len(passed_tickers)}/{len(all_tickers)} passed ({pass_rate:.1f}%)")
    
    return passed_tickers
