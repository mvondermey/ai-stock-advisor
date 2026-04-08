"""
Multi-Horizon Ensemble Strategy
Combines signals from different analysis horizons (long_term, medium_term, short_term)
for better entry/exit timing.
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from strategy_cache_adapter import ensure_price_history_cache, resolve_cache_current_date
from config import (
    MULTI_TIMEFRAMES, MULTI_TIMEFRAME_LOOKBACK,
    MULTI_TIMEFRAME_WEIGHTS, MULTI_TIMEFRAME_MIN_CONSENSUS,
    PORTFOLIO_SIZE, N_TOP_TICKERS, NUM_PROCESSES
)

_MULTI_TIMEFRAME_SELECTION_CONTEXT: Dict[str, object] = {}


def _timestamp_ns(value: datetime) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value)


def build_compact_cache(
    ticker_data: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    close_series = pd.to_numeric(ticker_data["Close"], errors="coerce")
    valid_mask = close_series.notna()
    if int(valid_mask.sum()) < 2:
        return {}

    filtered_close = close_series.loc[valid_mask]
    date_index = pd.DatetimeIndex(filtered_close.index)
    if date_index.tz is not None:
        date_index = date_index.tz_convert("UTC").tz_localize(None)

    volume_values = None
    if "Volume" in ticker_data.columns:
        volume_series = pd.to_numeric(ticker_data.loc[valid_mask, "Volume"], errors="coerce").fillna(0.0)
        volume_values = volume_series.to_numpy(dtype=float, copy=True)

    return {
        "date_ns": date_index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=True),
        "close": filtered_close.to_numpy(dtype=float, copy=True),
        "volume": volume_values,
    }


def _build_multi_timeframe_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    tickers: List[str],
    prebuilt_cache: Dict[str, Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = prebuilt_cache or {}
    iterator = tqdm(
        tickers,
        total=len(tickers),
        desc="   Multi-Horizon cache build",
        ncols=100,
        unit="ticker",
    )
    for ticker in iterator:
        ticker_data = ticker_data_grouped.get(ticker)
        if ticker_data is None or ticker_data.empty or "Close" not in ticker_data.columns:
            continue

        close_series = pd.to_numeric(ticker_data["Close"], errors="coerce")
        valid_mask = close_series.notna()
        if int(valid_mask.sum()) < 2:
            continue

        filtered_close = close_series.loc[valid_mask]
        date_index = pd.DatetimeIndex(filtered_close.index)
        if date_index.tz is not None:
            date_index = date_index.tz_convert("UTC").tz_localize(None)

        volume_values = None
        if "Volume" in ticker_data.columns:
            volume_series = pd.to_numeric(ticker_data.loc[valid_mask, "Volume"], errors="coerce").fillna(0.0)
            volume_values = volume_series.to_numpy(dtype=float, copy=True)

        cache[ticker] = {
            "date_ns": date_index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=True),
            "close": filtered_close.to_numpy(dtype=float, copy=True),
            "volume": volume_values,
        }

    return cache


def _build_multi_timeframe_cache_from_price_history(
    price_history_cache,
    tickers: List[str],
    prebuilt_cache: Dict[str, Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = prebuilt_cache or {}
    iterator = tqdm(
        tickers,
        total=len(tickers),
        desc="   Multi-Horizon cache build",
        ncols=100,
        unit="ticker",
    )
    for ticker in iterator:
        date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
        close_values = price_history_cache.close_by_ticker.get(ticker)
        if date_ns is None or close_values is None or date_ns.size < 2 or close_values.size < 2:
            continue

        cache[ticker] = {
            "date_ns": date_ns,
            "close": close_values,
            "volume": price_history_cache.volume_by_ticker.get(ticker),
        }

    return cache


def build_multi_timeframe_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    tickers: List[str],
    price_history_cache=None,
) -> Dict[str, Dict[str, np.ndarray]]:
    if price_history_cache is not None:
        return _build_multi_timeframe_cache_from_price_history(price_history_cache, tickers)
    return _build_multi_timeframe_cache(ticker_data_grouped, tickers)


def _get_cache_entry_from_price_history(price_history_cache, ticker: str) -> Optional[Dict[str, np.ndarray]]:
    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    close_values = price_history_cache.close_by_ticker.get(ticker)
    if date_ns is None or close_values is None or date_ns.size < 2 or close_values.size < 2:
        return None
    return {
        "date_ns": date_ns,
        "close": close_values,
        "volume": price_history_cache.volume_by_ticker.get(ticker),
    }


def _window_from_cache(
    cache_entry: Dict[str, np.ndarray],
    current_date: datetime,
    period_days: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    date_ns = cache_entry["date_ns"]
    end_ns = _timestamp_ns(current_date)
    start_ns = _timestamp_ns(current_date - timedelta(days=period_days))
    start_idx = int(np.searchsorted(date_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(date_ns, end_ns, side="right"))
    if end_idx <= start_idx:
        return np.array([], dtype=float), None

    close_window = cache_entry["close"][start_idx:end_idx]
    volume_values = cache_entry.get("volume")
    volume_window = None if volume_values is None else volume_values[start_idx:end_idx]
    return close_window, volume_window


def _calculate_daily_momentum_from_cache(
    cache_entry: Dict[str, np.ndarray],
    current_date: datetime,
) -> float:
    date_ns = cache_entry["date_ns"]
    end_ns = _timestamp_ns(current_date)
    end_idx = int(np.searchsorted(date_ns, end_ns, side="right"))
    if end_idx < 50:
        return 0.0

    close_values = cache_entry["close"][:end_idx]
    if close_values.size < 2:
        return 0.0

    momentum_1y = (close_values[-1] / close_values[0] - 1) * 100
    close_1y, _ = _window_from_cache(cache_entry, current_date, 365)
    if close_1y.size >= 50:
        momentum_1y = (close_1y[-1] / close_1y[0] - 1) * 100

    momentum_6m = momentum_1y
    close_6m, _ = _window_from_cache(cache_entry, current_date, 180)
    if close_6m.size >= 25:
        momentum_6m = (close_6m[-1] / close_6m[0] - 1) * 100

    momentum_3m = momentum_1y
    close_3m, _ = _window_from_cache(cache_entry, current_date, 90)
    if close_3m.size >= 10:
        momentum_3m = (close_3m[-1] / close_3m[0] - 1) * 100

    return momentum_1y * 0.5 + momentum_6m * 0.3 + momentum_3m * 0.2


def _calculate_medium_term_momentum_from_cache(close_values: np.ndarray) -> float:
    if close_values.size < 20:
        return 0.0

    momentum_30d = (close_values[-1] / close_values[0] - 1) * 100
    recent_10d = close_values[-10:]
    prev_10d = close_values[-20:-10]
    recent_avg = float(np.mean(recent_10d))
    prev_avg = float(np.mean(prev_10d))
    if prev_avg == 0:
        return 0.0

    trend_signal = ((recent_avg / prev_avg) - 1) * 100
    returns = np.diff(close_values) / close_values[:-1]
    if returns.size == 0:
        return 0.0
    volatility = float(np.std(returns) * np.sqrt(252) * 100)
    vol_adjusted = momentum_30d / (volatility + 1)
    return vol_adjusted * 0.7 + trend_signal * 0.3


def _calculate_short_term_momentum_from_cache(
    close_values: np.ndarray,
    volume_values: Optional[np.ndarray],
) -> float:
    if close_values.size < 5:
        return 0.0

    momentum_7d = (close_values[-1] / close_values[0] - 1) * 100
    recent_3d = close_values[-3:]
    price_change = (recent_3d[-1] / recent_3d[0] - 1) * 100

    volume_factor = 1.0
    if volume_values is not None and volume_values.size >= 5:
        recent_vol = float(np.mean(volume_values[-5:]))
        avg_vol = float(np.mean(volume_values))
        volume_factor = min(recent_vol / (avg_vol + 1), 2.0)

    return (momentum_7d * 0.5 + price_change * 0.3) * volume_factor


def _calculate_multi_timeframe_signals_from_cache(
    ticker: str,
    cache_entry: Dict[str, np.ndarray],
    current_date: datetime,
    timeframes: List[str] = None,
) -> Dict[str, float]:
    if timeframes is None:
        timeframes = MULTI_TIMEFRAMES

    signals = {}
    for timeframe in timeframes:
        if timeframe == "long_term":
            signals[timeframe] = _calculate_daily_momentum_from_cache(cache_entry, current_date)
            continue

        lookback_days = MULTI_TIMEFRAME_LOOKBACK[timeframe]
        close_window, volume_window = _window_from_cache(cache_entry, current_date, lookback_days)
        if timeframe == "medium_term":
            signals[timeframe] = _calculate_medium_term_momentum_from_cache(close_window)
        elif timeframe == "short_term":
            signals[timeframe] = _calculate_short_term_momentum_from_cache(close_window, volume_window)
        else:
            signals[timeframe] = 0.0

    return signals


def calculate_multi_timeframe_signals(
    ticker: str,
    ticker_data: pd.DataFrame,
    current_date: datetime,
    timeframes: List[str] = None
) -> Dict[str, float]:
    """
    Calculate momentum signals for multiple timeframes

    Args:
        ticker: Stock symbol
        ticker_data: Historical price data (daily)
        current_date: Current date for analysis
        timeframes: List of timeframes to analyze

    Returns:
        Dictionary of signals per timeframe
    """
    if timeframes is None:
        timeframes = MULTI_TIMEFRAMES

    signals = {}

    for timeframe in timeframes:
        lookback_days = MULTI_TIMEFRAME_LOOKBACK[timeframe]

        # Calculate start date for this timeframe
        start_date = current_date - timedelta(days=lookback_days)

        # Filter data for lookback period
        tf_data = ticker_data[ticker_data.index >= start_date]

        if len(tf_data) < 10:  # Need minimum data points
            signals[timeframe] = 0.0
            continue

        # Calculate momentum signal based on analysis horizon
        if timeframe == "long_term":
            # Long-term momentum (1-year return)
            signal = calculate_daily_momentum(tf_data)
        elif timeframe == "medium_term":
            # Medium-term momentum (30-day return with volatility adjustment)
            signal = calculate_medium_term_momentum(tf_data)
        elif timeframe == "short_term":
            # Short-term momentum (7-day return with recent trend)
            signal = calculate_short_term_momentum(tf_data)
        else:
            signal = 0.0

        signals[timeframe] = signal

    return signals

def calculate_daily_momentum(data: pd.DataFrame, current_date: datetime = None) -> float:
    """Calculate long-term daily momentum signal using calendar days"""
    if len(data) < 50:
        return 0.0

    # Use calendar days for calculations
    if current_date is None:
        current_date = data.index.max()

    # Filter data up to current_date
    data = data[data.index <= current_date]

    # 1-year momentum (365 calendar days)
    start_1y = current_date - timedelta(days=365)
    data_1y = data[data.index >= start_1y]
    if len(data_1y) >= 50:
        momentum_1y = (data_1y['Close'].iloc[-1] / data_1y['Close'].iloc[0] - 1) * 100
    else:
        momentum_1y = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100

    # 6-month momentum (180 calendar days)
    start_6m = current_date - timedelta(days=180)
    data_6m = data[data.index >= start_6m]
    if len(data_6m) >= 25:
        momentum_6m = (data_6m['Close'].iloc[-1] / data_6m['Close'].iloc[0] - 1) * 100
    else:
        momentum_6m = momentum_1y

    # 3-month momentum (90 calendar days)
    start_3m = current_date - timedelta(days=90)
    data_3m = data[data.index >= start_3m]
    if len(data_3m) >= 10:
        momentum_3m = (data_3m['Close'].iloc[-1] / data_3m['Close'].iloc[0] - 1) * 100
    else:
        momentum_3m = momentum_1y

    # Weighted combination
    signal = (momentum_1y * 0.5 + momentum_6m * 0.3 + momentum_3m * 0.2)

    return signal

def calculate_medium_term_momentum(data: pd.DataFrame) -> float:
    """Calculate medium-term momentum signal

    Note: data is already filtered by lookback period from calculate_multi_timeframe_signals
    """
    if len(data) < 20:
        return 0.0

    # 30-day momentum (data is already filtered to lookback period)
    if len(data) >= 2:
        momentum_30d = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    else:
        return 0.0

    # Recent trend (last 10 days vs previous 10 days)
    if len(data) >= 20:
        recent_10d = data['Close'].iloc[-10:]
        prev_10d = data['Close'].iloc[-20:-10]
        recent_avg = recent_10d.mean()
        prev_avg = prev_10d.mean()
        trend_signal = ((recent_avg / prev_avg) - 1) * 100
    else:
        trend_signal = momentum_30d

    # Volatility adjustment
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    vol_adjusted = momentum_30d / (volatility + 1)  # Avoid division by zero

    # Combine momentum and trend
    signal = (vol_adjusted * 0.7 + trend_signal * 0.3)

    return signal

def calculate_short_term_momentum(data: pd.DataFrame) -> float:
    """Calculate short-term momentum signal

    Note: data is already filtered by lookback period from calculate_multi_timeframe_signals
    """
    if len(data) < 5:
        return 0.0

    # 7-day momentum (data is already filtered to lookback period)
    if len(data) >= 2:
        momentum_7d = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    else:
        return 0.0

    # Recent price action (last 3 days)
    if len(data) >= 3:
        recent_3d = data['Close'].iloc[-3:]
        price_change = (recent_3d.iloc[-1] / recent_3d.iloc[0] - 1) * 100
    else:
        price_change = momentum_7d

    # Volume confirmation (if available)
    if 'Volume' in data.columns and len(data) >= 5:
        recent_vol = data['Volume'].iloc[-5:].mean()
        avg_vol = data['Volume'].mean()
        volume_factor = min(recent_vol / (avg_vol + 1), 2.0)  # Cap at 2x
    else:
        volume_factor = 1.0

    # Combine signals
    signal = (momentum_7d * 0.5 + price_change * 0.3) * volume_factor

    return signal

def calculate_ensemble_score(signals: Dict[str, float]) -> Tuple[float, bool]:
    """
    Calculate weighted ensemble score and consensus

    Args:
        signals: Dictionary of signals per timeframe

    Returns:
        Tuple of (ensemble_score, has_consensus)
    """
    ensemble_score = 0.0
    consensus_count = 0

    for timeframe, signal in signals.items():
        weight = MULTI_TIMEFRAME_WEIGHTS[timeframe]
        ensemble_score += signal * weight

        # Count positive signals for consensus
        if signal > 0:
            consensus_count += 1

    has_consensus = consensus_count >= MULTI_TIMEFRAME_MIN_CONSENSUS

    return ensemble_score, has_consensus


def _init_multi_timeframe_worker(
    selection_cache: Optional[Dict[str, Dict[str, np.ndarray]]],
    current_date: datetime,
    price_history_cache=None,
) -> None:
    global _MULTI_TIMEFRAME_SELECTION_CONTEXT
    _MULTI_TIMEFRAME_SELECTION_CONTEXT = {
        "selection_cache": selection_cache,
        "current_date": current_date,
        "price_history_cache": price_history_cache,
    }


def _score_multi_timeframe_ticker_worker(
    ticker: str,
) -> Optional[Tuple[str, float, Dict[str, float]]]:
    context = _MULTI_TIMEFRAME_SELECTION_CONTEXT
    selection_cache = context.get("selection_cache") or {}
    current_date = context.get("current_date")
    price_history_cache = context.get("price_history_cache")
    cache_entry = selection_cache.get(ticker)
    if cache_entry is None and price_history_cache is not None:
        cache_entry = _get_cache_entry_from_price_history(price_history_cache, ticker)
    if cache_entry is None or current_date is None:
        return None

    signals = _calculate_multi_timeframe_signals_from_cache(ticker, cache_entry, current_date)
    ensemble_score, has_consensus = calculate_ensemble_score(signals)
    if has_consensus:
        return (ticker, ensemble_score, signals)
    return None

def select_multi_timeframe_stocks(
    initial_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = PORTFOLIO_SIZE,
    verbose: bool = True,
    selection_cache: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    price_history_cache=None,
) -> List[str]:
    """
    Select stocks using multi-timeframe ensemble strategy

    Args:
        initial_tickers: List of candidate tickers
        ticker_data_grouped: Dictionary of ticker data
        current_date: Current date for analysis
        top_n: Number of stocks to select
        verbose: Whether to print detailed output

    Returns:
        List of selected tickers
    """
    # Filter out inverse ETFs - they should only be in inverse_etf_hedge strategy
    from config import INVERSE_ETFS
    tickers_to_use = [t for t in initial_tickers if t not in INVERSE_ETFS]
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    current_date = resolve_cache_current_date(price_history_cache, current_date, tickers_to_use)
    if current_date is None:
        return []
    if selection_cache is None:
        cached_tickers = [
            ticker for ticker in tickers_to_use
            if _get_cache_entry_from_price_history(price_history_cache, ticker) is not None
        ]
    else:
        cached_tickers = [ticker for ticker in tickers_to_use if ticker in selection_cache]

    stock_scores = []
    n_workers = max(1, min(NUM_PROCESSES, len(cached_tickers))) if cached_tickers else 1

    if n_workers > 1 and len(cached_tickers) >= max(32, n_workers * 2):
        if verbose:
            print(f"   🚀 Multi-Horizon Ensemble: Scoring {len(cached_tickers)} tickers with {n_workers} threads")
        _init_multi_timeframe_worker(selection_cache, current_date, price_history_cache=price_history_cache)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(_score_multi_timeframe_ticker_worker, cached_tickers),
                total=len(cached_tickers),
                desc="   Multi-Horizon Ensemble scoring",
                ncols=100,
                unit="ticker",
            ))
        stock_scores = [result for result in results if result is not None]
    else:
        for ticker in cached_tickers:
            cache_entry = (
                selection_cache.get(ticker)
                if selection_cache is not None
                else _get_cache_entry_from_price_history(price_history_cache, ticker)
            )
            if cache_entry is None:
                continue

            signals = _calculate_multi_timeframe_signals_from_cache(ticker, cache_entry, current_date)
            ensemble_score, has_consensus = calculate_ensemble_score(signals)
            if has_consensus:
                stock_scores.append((ticker, ensemble_score, signals))

    # Sort by ensemble score
    stock_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top N stocks
    selected_stocks = [ticker for ticker, score, signals in stock_scores[:top_n]]

    if verbose:
        print(f"   📊 Multi-Horizon Ensemble: {len(stock_scores)} candidates with consensus")
        print(f"   🎯 Selected: {selected_stocks[:5]}..." if len(selected_stocks) > 5 else f"   🎯 Selected: {selected_stocks}")

    return selected_stocks

def print_multi_timeframe_analysis(selected_stocks: List[str], stock_scores: List[Tuple]):
    """Print detailed multi-timeframe analysis"""
    print(f"\n🎯 Multi-Horizon Ensemble Analysis:")
    print(f"   Selected {len(selected_stocks)} stocks from {len(stock_scores)} candidates")
    print(f"   Consensus requirement: {MULTI_TIMEFRAME_MIN_CONSENSUS}/{len(MULTI_TIMEFRAMES)} timeframes")

    print(f"\n   Top 5 selections:")
    for i, (ticker, score, signals) in enumerate(stock_scores[:5]):
        print(f"   {i+1}. {ticker}: Score={score:.2f}")
        for tf, signal in signals.items():
            weight = MULTI_TIMEFRAME_WEIGHTS[tf]
            print(f"      {tf}: {signal:+.2f} (weight: {weight:.1f})")

    print(f"\n   Timeframe weights: {MULTI_TIMEFRAME_WEIGHTS}")
    print(f"   Lookback periods: {MULTI_TIMEFRAME_LOOKBACK}")
