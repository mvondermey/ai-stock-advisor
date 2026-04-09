"""
Parallel performance calculations for backtesting optimization.
This module provides parallelized versions of performance calculation functions.
"""

from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime, timedelta

from strategy_disk_cache import (
    load_joblib_cache,
    save_joblib_cache,
    universe_signature_from_frames,
)


@dataclass
class PriceHistoryCache:
    """Precomputed valid close arrays plus rolling window caches for ranking."""
    date_ns_by_ticker: Dict[str, np.ndarray]
    close_by_ticker: Dict[str, np.ndarray]
    volume_by_ticker: Dict[str, np.ndarray] = field(default_factory=dict)
    high_by_ticker: Dict[str, np.ndarray] = field(default_factory=dict)
    low_by_ticker: Dict[str, np.ndarray] = field(default_factory=dict)
    performance_cache: Dict[Tuple[int, int], Dict[str, float]] = field(default_factory=dict)
    risk_adjusted_cache: Dict[Tuple[int, int], Dict[str, Tuple[float, float, float]]] = field(default_factory=dict)
    volatility_cache: Dict[Tuple[int, int], Dict[str, float]] = field(default_factory=dict)


def _timestamp_ns(value: datetime) -> int:
    return int(pd.Timestamp(value).value)


def _min_rows_for_period(period_days: int) -> int:
    if period_days <= 30:
        return 10
    if period_days <= 100:
        return 20
    if period_days <= 200:
        return 30
    return 50


def _window_slice(
    date_ns: np.ndarray,
    closes: np.ndarray,
    current_date: datetime,
    period_days: int,
) -> Optional[np.ndarray]:
    end_ns = _timestamp_ns(current_date)
    start_ns = _timestamp_ns(current_date - timedelta(days=period_days))
    start_idx = int(np.searchsorted(date_ns, start_ns, side='left'))
    end_idx = int(np.searchsorted(date_ns, end_ns, side='right'))
    if end_idx - start_idx < _min_rows_for_period(period_days):
        return None
    window = closes[start_idx:end_idx]
    if window.size < 2:
        return None
    return window


def build_price_history_cache(ticker_data_grouped: Dict[str, pd.DataFrame]) -> PriceHistoryCache:
    """Precompute valid close arrays for fast window lookups."""
    cache_key_parts = {
        "universe_signature": universe_signature_from_frames(ticker_data_grouped),
        "ticker_count": len(ticker_data_grouped),
    }
    cached_result = load_joblib_cache(
        "parallel_backtest/base_price_history",
        cache_key_parts,
        filename="price_history_cache.joblib",
    )
    if isinstance(cached_result, PriceHistoryCache):
        return cached_result

    date_ns_by_ticker: Dict[str, np.ndarray] = {}
    close_by_ticker: Dict[str, np.ndarray] = {}
    volume_by_ticker: Dict[str, np.ndarray] = {}
    high_by_ticker: Dict[str, np.ndarray] = {}
    low_by_ticker: Dict[str, np.ndarray] = {}

    for ticker, ticker_data in ticker_data_grouped.items():
        if ticker_data is None or ticker_data.empty or 'Close' not in ticker_data.columns:
            continue
        valid_close = pd.to_numeric(ticker_data['Close'], errors='coerce').dropna()
        if len(valid_close) < 2:
            continue
        date_index = pd.DatetimeIndex(valid_close.index)
        if date_index.tz is not None:
            date_index = date_index.tz_convert("UTC").tz_localize(None)
        date_ns_by_ticker[ticker] = date_index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=True)
        close_by_ticker[ticker] = valid_close.to_numpy(dtype=float, copy=True)

        if 'Volume' in ticker_data.columns:
            volume_values = pd.to_numeric(ticker_data.loc[valid_close.index, 'Volume'], errors='coerce').to_numpy(dtype=float, copy=True)
            volume_by_ticker[ticker] = volume_values
        if 'High' in ticker_data.columns:
            high_values = pd.to_numeric(ticker_data.loc[valid_close.index, 'High'], errors='coerce').to_numpy(dtype=float, copy=True)
            high_by_ticker[ticker] = high_values
        if 'Low' in ticker_data.columns:
            low_values = pd.to_numeric(ticker_data.loc[valid_close.index, 'Low'], errors='coerce').to_numpy(dtype=float, copy=True)
            low_by_ticker[ticker] = low_values

    cache = PriceHistoryCache(
        date_ns_by_ticker=date_ns_by_ticker,
        close_by_ticker=close_by_ticker,
        volume_by_ticker=volume_by_ticker,
        high_by_ticker=high_by_ticker,
        low_by_ticker=low_by_ticker,
    )
    save_joblib_cache(
        "parallel_backtest/base_price_history",
        cache_key_parts,
        cache,
        filename="price_history_cache.joblib",
    )
    return cache


def calculate_cached_performance(
    tickers: List[str],
    price_history_cache: PriceHistoryCache,
    current_date: datetime,
    period_days: int = 365,
) -> List[Tuple[str, float]]:
    """Return cached lookback performance using precomputed close arrays."""
    cache_key = (_timestamp_ns(current_date), int(period_days))
    if cache_key not in price_history_cache.performance_cache:
        start_time = time.time()
        performance_map: Dict[str, float] = {}
        for ticker, closes in price_history_cache.close_by_ticker.items():
            date_ns = price_history_cache.date_ns_by_ticker[ticker]
            window = _window_slice(date_ns, closes, current_date, period_days)
            if window is None:
                continue
            start_price = float(window[0])
            end_price = float(window[-1])
            if start_price > 0 and not np.isnan(start_price) and not np.isnan(end_price):
                performance_map[ticker] = ((end_price - start_price) / start_price) * 100.0
        elapsed = time.time() - start_time
        print(f"   ⏱️ Cached performance: {len(price_history_cache.close_by_ticker)} tickers in {elapsed:.2f}s (window={period_days}d)")
        price_history_cache.performance_cache[cache_key] = performance_map

    performance_map = price_history_cache.performance_cache[cache_key]
    return [(ticker, performance_map[ticker]) for ticker in tickers if ticker in performance_map]


def calculate_cached_volatility(
    tickers: List[str],
    price_history_cache: PriceHistoryCache,
    current_date: datetime,
    period_days: int = 365,
) -> Dict[str, float]:
    """Return cached annualized volatility using the same lookback window arrays."""
    cache_key = (_timestamp_ns(current_date), int(period_days))
    if cache_key not in price_history_cache.volatility_cache:
        start_time = time.time()
        volatility_map: Dict[str, float] = {}
        for ticker, closes in price_history_cache.close_by_ticker.items():
            date_ns = price_history_cache.date_ns_by_ticker[ticker]
            window = _window_slice(date_ns, closes, current_date, period_days)
            if window is None or window.size < 3:
                continue
            daily_returns = np.diff(window) / window[:-1]
            if daily_returns.size > 10:
                volatility_map[ticker] = float(np.std(daily_returns, ddof=1) * np.sqrt(252) * 100.0)
        elapsed = time.time() - start_time
        print(f"   ⏱️ Cached volatility: {len(price_history_cache.close_by_ticker)} tickers in {elapsed:.2f}s (window={period_days}d)")
        price_history_cache.volatility_cache[cache_key] = volatility_map

    volatility_map = price_history_cache.volatility_cache[cache_key]
    return {ticker: volatility_map[ticker] for ticker in tickers if ticker in volatility_map}


def calculate_cached_risk_adjusted_scores(
    tickers: List[str],
    price_history_cache: PriceHistoryCache,
    current_date: datetime,
    lookback_days: int = 365,
) -> List[Tuple[str, float, float, float]]:
    """Return cached risk-adjusted momentum scores from precomputed arrays."""
    cache_key = (_timestamp_ns(current_date), int(lookback_days))
    if cache_key not in price_history_cache.risk_adjusted_cache:
        start_time = time.time()
        risk_adjusted_map: Dict[str, Tuple[float, float, float]] = {}
        min_perf_days = max(10, lookback_days // 5)
        for ticker, closes in price_history_cache.close_by_ticker.items():
            date_ns = price_history_cache.date_ns_by_ticker[ticker]
            window = _window_slice(date_ns, closes, current_date, lookback_days)
            if window is None or window.size < min_perf_days:
                continue
            start_price = float(window[0])
            end_price = float(window[-1])
            if start_price <= 0 or np.isnan(start_price) or np.isnan(end_price):
                continue
            daily_returns = np.diff(window) / window[:-1]
            if daily_returns.size <= 5:
                continue
            basic_return = ((end_price - start_price) / start_price) * 100.0
            volatility = float(np.std(daily_returns, ddof=1) * 100.0)
            risk_adjusted_score = float(basic_return / (volatility**0.5 + 0.001))
            risk_adjusted_map[ticker] = (risk_adjusted_score, basic_return, volatility)
        elapsed = time.time() - start_time
        print(f"   ⏱️ Cached risk-adjusted: {len(price_history_cache.close_by_ticker)} tickers in {elapsed:.2f}s (window={lookback_days}d)")
        price_history_cache.risk_adjusted_cache[cache_key] = risk_adjusted_map

    risk_adjusted_map = price_history_cache.risk_adjusted_cache[cache_key]
    return [
        (ticker, *risk_adjusted_map[ticker])
        for ticker in tickers
        if ticker in risk_adjusted_map
    ]


def calculate_single_ticker_performance(args):
    """Calculate performance for a single ticker - used for parallel processing."""
    ticker, ticker_data, current_date, period_days = args

    try:
        if ticker_data.empty:
            return None

        perf_start_date = current_date - timedelta(days=period_days)
        perf_data = ticker_data.loc[perf_start_date:current_date]

        # Adaptive min rows based on period length
        if period_days <= 30:
            min_rows = 10
        elif period_days <= 100:
            min_rows = 20
        elif period_days <= 200:
            min_rows = 30
        else:
            min_rows = 50

        if len(perf_data) >= min_rows:
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
                                 period_days: int = 365,
                                 num_processes: int = None) -> List[Tuple[str, float]]:
    """
    Calculate performance for multiple tickers in parallel.

    Args:
        tickers: List of ticker symbols
        ticker_data_grouped: Dict mapping ticker -> price data
        current_date: Current date for calculation
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
        args_list.append((ticker, ticker_data, current_date, period_days))

    # Process in parallel
    start_time = time.time()
    with Pool(processes=num_processes, maxtasksperchild=50) as pool:
        results = pool.map(calculate_single_ticker_performance, args_list)
    gc.collect()  # Release semaphores after Pool closes (WSL fix)
    elapsed = time.time() - start_time
    print(f"   ⏱️ Parallel processing: {len(tickers)} tickers in {elapsed:.2f}s ({num_processes} processes)")

    # Filter out None results and return valid performances
    performances = [r for r in results if r is not None]
    return performances


def _calculate_single_risk_adj(args):
    """Module-level worker for parallel risk-adjusted score calculation."""
    ticker, ticker_data, current_date, lookback_days = args

    try:
        # Use calendar days with timedelta
        end_date = current_date or ticker_data.index.max()
        start_date = end_date - timedelta(days=lookback_days)

        # Filter data to date range
        perf_data = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= end_date)]

        # Need at least 1/3 of lookback period in trading days
        min_perf_days = max(10, lookback_days // 5)  # ~1/5 for calendar->trading conversion
        if len(perf_data) < min_perf_days:
            return None

        valid_close = perf_data['Close'].dropna()
        if len(valid_close) < min_perf_days:
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


def calculate_parallel_risk_adjusted_scores(tickers: List[str],
                                           ticker_data_grouped: Dict[str, pd.DataFrame],
                                           current_date: datetime,
                                           lookback_days: int = 365,
                                           num_processes: int = None) -> List[Tuple[str, float, float, float]]:
    """
    Calculate risk-adjusted momentum scores in parallel.

    Args:
        lookback_days: Performance window in days (365=1Y, 180=6M, 90=3M, 30=1M)

    Returns:
        List of (ticker, score, return_pct, volatility_pct) tuples
    """
    if num_processes is None:
        from config import NUM_PROCESSES
        num_processes = max(1, NUM_PROCESSES)

    start_time = time.time()

    # Prepare arguments
    args_list = []
    for ticker in tickers:
        ticker_data = ticker_data_grouped.get(ticker, pd.DataFrame())
        args_list.append((ticker, ticker_data, current_date, lookback_days))

    # Process in parallel
    with Pool(processes=num_processes, maxtasksperchild=50) as pool:
        results = pool.map(_calculate_single_risk_adj, args_list)
    gc.collect()  # Release semaphores after Pool closes (WSL fix)
    elapsed = time.time() - start_time
    print(f"   ⏱️ Parallel risk-adjusted: {len(tickers)} tickers in {elapsed:.2f}s ({num_processes} processes)")

    # Filter and return
    scores = [r for r in results if r is not None]
    return scores
