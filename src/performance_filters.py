"""
Performance Filters Module
Provides universal performance threshold filtering for all strategies.
Uses calendar days (not trading days) for consistency across all calculations.
"""

from typing import List
from datetime import datetime
from config import (
    MIN_PERFORMANCE_1M, MIN_PERFORMANCE_1Y, MIN_PERFORMANCE_6M, MIN_PERFORMANCE_3M,
    ENABLE_PERFORMANCE_FILTERS,
    MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M,
    INVERSE_ETFS, INVERSE_ETF_MIN_PERFORMANCE_1M, INVERSE_ETF_MIN_PERFORMANCE_3M,
    INVERSE_ETF_SKIP_1Y_FILTER
)

def _passes_inverse_etf_thresholds(perf_1y: float, perf_3m: float, perf_1m: float) -> bool:
    if not INVERSE_ETF_SKIP_1Y_FILTER and perf_1y < MIN_PERFORMANCE_1Y:
        return False
    if perf_3m < INVERSE_ETF_MIN_PERFORMANCE_3M:
        return False
    if perf_1m < INVERSE_ETF_MIN_PERFORMANCE_1M:
        return False
    return True


def _passes_normal_thresholds(perf_1y: float, perf_6m: float, perf_3m: float, perf_1m: float) -> bool:
    if perf_1y < MIN_PERFORMANCE_1Y:
        return False
    if perf_6m < MIN_PERFORMANCE_6M:
        return False
    if perf_3m < MIN_PERFORMANCE_3M:
        return False
    if perf_1m < MIN_PERFORMANCE_1M:
        return False
    return True


def filter_tickers_by_performance(
    all_tickers: List[str],
    current_date: datetime,
    strategy_name: str = "Unknown",
    price_history_cache=None
) -> List[str]:
    """
    Filter a list of tickers by performance thresholds.
    
    Args:
        all_tickers: List of ticker symbols to filter
        current_date: Current analysis date
        strategy_name: Name of strategy for debug output
        price_history_cache: Required PriceHistoryCache for fast lookups
        
    Returns:
        List of tickers that pass performance filters
    """
    if not ENABLE_PERFORMANCE_FILTERS:
        return all_tickers  # No filtering enabled
    if current_date is None:
        raise ValueError(f"{strategy_name}: current_date is required for cache-backed performance filtering")
    if price_history_cache is None:
        raise ValueError(f"{strategy_name}: price_history_cache is required; DataFrame fallback was removed")

    passed_tickers = []
    failed_count = 0
    max_debug = 10  # Limit debug output to avoid flooding
    
    print(f"\n   [DEBUG] {strategy_name}: Performance filter analysis (first {max_debug} failures shown)")

    from parallel_backtest import get_cached_performance_map
    import time as _time
    _filter_start = _time.time()

    perf_1y_map = get_cached_performance_map(price_history_cache, current_date, 365)
    perf_6m_map = get_cached_performance_map(price_history_cache, current_date, 180)
    perf_3m_map = get_cached_performance_map(price_history_cache, current_date, 90)
    perf_1m_map = get_cached_performance_map(price_history_cache, current_date, 30)

    for ticker in all_tickers:
        perf_1y_pct = perf_1y_map.get(ticker)
        perf_6m_pct = perf_6m_map.get(ticker)
        perf_3m_pct = perf_3m_map.get(ticker)
        perf_1m_pct = perf_1m_map.get(ticker, 0.0)

        if perf_1y_pct is None or perf_6m_pct is None or perf_3m_pct is None:
            failed_count += 1
            continue

        perf_1y = perf_1y_pct / 100.0
        perf_6m = perf_6m_pct / 100.0
        perf_3m = perf_3m_pct / 100.0
        perf_1m = perf_1m_pct / 100.0

        if ticker in INVERSE_ETFS:
            passed = _passes_inverse_etf_thresholds(perf_1y, perf_3m, perf_1m)
        else:
            passed = _passes_normal_thresholds(perf_1y, perf_6m, perf_3m, perf_1m)

        if passed:
            passed_tickers.append(ticker)
            if len(passed_tickers) <= 5:
                print(f"   [PASS] {ticker}: PASSED (1Y={perf_1y:.1%}, 6M={perf_6m:.1%}, 3M={perf_3m:.1%}, 1M={perf_1m:.1%})")
        else:
            failed_count += 1

    _elapsed = _time.time() - _filter_start
    if len(all_tickers) > 0:
        pass_rate = (len(passed_tickers) / len(all_tickers)) * 100
        print(f"   ⚡ {strategy_name}: Cached perf filter - {len(passed_tickers)}/{len(all_tickers)} passed ({pass_rate:.1f}%) in {_elapsed:.2f}s")

    return passed_tickers
