"""
Multi-Timeframe Ensemble Strategy
Combines signals from different analysis horizons (long_term, medium_term, short_term)
for better entry/exit timing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from config import (
    MULTI_TIMEFRAMES, MULTI_TIMEFRAME_LOOKBACK,
    MULTI_TIMEFRAME_WEIGHTS, MULTI_TIMEFRAME_MIN_CONSENSUS,
    PORTFOLIO_SIZE, N_TOP_TICKERS
)

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

def select_multi_timeframe_stocks(
    initial_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = PORTFOLIO_SIZE,
    verbose: bool = True
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

    stock_scores = []

    for ticker in tickers_to_use:
        if ticker not in ticker_data_grouped:
            continue

        ticker_data = ticker_data_grouped[ticker]

        # Calculate multi-timeframe signals
        signals = calculate_multi_timeframe_signals(ticker, ticker_data, current_date)

        # Calculate ensemble score and consensus
        ensemble_score, has_consensus = calculate_ensemble_score(signals)

        # Only consider stocks with consensus
        if has_consensus:
            stock_scores.append((ticker, ensemble_score, signals))

    # Sort by ensemble score
    stock_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top N stocks
    selected_stocks = [ticker for ticker, score, signals in stock_scores[:top_n]]

    if verbose:
        print(f"   📊 Multi-TF Ensemble: {len(stock_scores)} candidates with consensus")
        print(f"   🎯 Selected: {selected_stocks[:5]}..." if len(selected_stocks) > 5 else f"   🎯 Selected: {selected_stocks}")

    return selected_stocks

def print_multi_timeframe_analysis(selected_stocks: List[str], stock_scores: List[Tuple]):
    """Print detailed multi-timeframe analysis"""
    print(f"\n🎯 Multi-Timeframe Ensemble Analysis:")
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
