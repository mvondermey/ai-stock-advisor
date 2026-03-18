#!/usr/bin/env python3
"""Enhanced Static BH 1Y strategies with smart rebalancing triggers."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

from shared_strategies import select_top_performers
from config import (
    STATIC_BH_1Y_VOLUME_MIN_DAILY,
    STATIC_BH_1Y_SECTOR_MAX_PER_SECTOR,
    STATIC_BH_1Y_PERFORMANCE_MIN_IMPROVEMENT,
    STATIC_BH_1Y_MARKET_REGIME_BASE_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_HIGH_VOL_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_LOW_VOL_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_VOL_THRESHOLD
)


def select_volume_filtered_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    min_volume: float = STATIC_BH_1Y_VOLUME_MIN_DAILY  # Minimum $1M daily volume
) -> List[str]:
    """
    Static BH 1Y with volume confirmation filter.
    Only selects stocks with sufficient liquidity.
    """
    # Get top performers first
    candidates = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n * 2  # Get more to filter
    )
    
    # Apply volume filter
    filtered_stocks = []
    for ticker in candidates:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if not data.empty and 'Volume' in data.columns:
                # Calculate average daily volume over last 20 days
                recent_volume = data['Volume'].tail(20).mean()
                # Convert to dollar volume (volume * price)
                recent_price = data['Close'].tail(20).mean()
                if recent_price > 0:
                    avg_dollar_volume = recent_volume * recent_price
                    if avg_dollar_volume >= min_volume:
                        filtered_stocks.append(ticker)
    
    return filtered_stocks[:top_n]


def select_sector_rotated_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    max_per_sector: int = 3
) -> List[str]:
    """
    Static BH 1Y with sector rotation enhancement.
    Limits to max 3 stocks per sector for diversification.
    """
    # Simple sector mapping (you can expand this)
    sector_mapping = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'NFLX'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'MDT', 'ISRG'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'SCHW'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR'],
        'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'LOW', 'TGT', 'COST'],
        'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT'],
        'Materials': ['DOW', 'DD', 'NUE', 'FCX', 'BHP'],
        'Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'XEL'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
        'Communication': ['VZ', 'T', 'TMUS', 'CMCSA', 'CHTR']
    }
    
    # Reverse mapping: ticker -> sector
    ticker_to_sector = {}
    for sector, tickers_in_sector in sector_mapping.items():
        for ticker in tickers_in_sector:
            ticker_to_sector[ticker] = sector
    
    # Get top performers
    candidates = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n * 3  # Get more to ensure diversity
    )
    
    # Apply sector filter
    selected = []
    sector_counts = {}
    
    for ticker in candidates:
        sector = ticker_to_sector.get(ticker, 'Other')
        current_count = sector_counts.get(sector, 0)
        
        if current_count < max_per_sector:
            selected.append(ticker)
            sector_counts[sector] = current_count + 1
            
        if len(selected) >= top_n:
            break
    
    return selected


def select_performance_threshold_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    current_stocks: List[str],
    top_n: int = 10,
    min_improvement: float = 0.02  # 2% minimum improvement
) -> Tuple[List[str], bool]:
    """
    Static BH 1Y with performance threshold trigger.
    Only rebalances if new selection offers >2% improvement.
    
    Returns:
        (selected_stocks, should_rebalance)
    """
    # Get current portfolio performance
    if not current_stocks:
        # No current stocks, always rebalance
        new_stocks = select_top_performers(
            tickers, ticker_data_grouped, current_date,
            lookback_days=365, top_n=top_n
        )
        return new_stocks, True
    
    # Calculate current portfolio momentum
    current_momentum = 0
    valid_count = 0
    
    for ticker in current_stocks:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if len(data) >= 365:
                start_date = current_date - timedelta(days=365)
                period_data = data[data.index >= start_date]
                if len(period_data) >= 2:
                    start_price = period_data['Close'].iloc[0]
                    end_price = period_data['Close'].iloc[-1]
                    momentum = (end_price / start_price - 1)
                    current_momentum += momentum
                    valid_count += 1
    
    if valid_count > 0:
        current_momentum /= valid_count
    
    # Get new candidates
    new_candidates = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # Calculate new portfolio momentum
    new_momentum = 0
    valid_count = 0
    
    for ticker in new_candidates:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if len(data) >= 365:
                start_date = current_date - timedelta(days=365)
                period_data = data[data.index >= start_date]
                if len(period_data) >= 2:
                    start_price = period_data['Close'].iloc[0]
                    end_price = period_data['Close'].iloc[-1]
                    momentum = (end_price / start_price - 1)
                    new_momentum += momentum
                    valid_count += 1
    
    if valid_count > 0:
        new_momentum /= valid_count
    
    # Check if improvement exceeds threshold
    improvement = new_momentum - current_momentum
    should_rebalance = improvement > min_improvement
    
    return new_candidates, should_rebalance


def select_market_regime_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    base_rebalance_days: int = 22,
    high_vol_days: int = 15,
    low_vol_days: int = 30,
    vol_threshold: float = 0.20  # 20% volatility threshold
) -> Tuple[List[str], int]:
    """
    Static BH 1Y with market-regime based rebalancing.
    Adjusts rebalancing frequency based on market volatility.
    
    Returns:
        (selected_stocks, next_rebalance_days)
    """
    # Calculate market volatility using SPY
    market_volatility = 0.20  # Default
    spy_data = ticker_data_grouped.get('SPY')
    if spy_data is not None:
        spy_recent = spy_data.loc[:current_date].tail(60)  # Last 60 days
        if len(spy_recent) >= 20:
            spy_returns = spy_recent['Close'].pct_change().dropna()
            if len(spy_returns) > 0:
                market_volatility = spy_returns.std() * np.sqrt(252)  # Annualized
    
    # Determine rebalancing frequency based on volatility
    if market_volatility > vol_threshold:
        # High volatility - rebalance more frequently
        rebalance_days = high_vol_days
    elif market_volatility < vol_threshold / 2:
        # Low volatility - rebalance less frequently
        rebalance_days = low_vol_days
    else:
        # Normal volatility - use base frequency
        rebalance_days = base_rebalance_days
    
    # Get top performers
    selected_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    return selected_stocks, rebalance_days


def check_momentum_persistence(
    previous_top_stocks_history: List[List[str]],
    current_top_stocks: List[str],
    min_stable_days: int = 5,
    min_overlap: float = 0.8
) -> Tuple[bool, List[List[str]]]:
    """
    Check if top stocks have been stable for N consecutive days.
    
    Args:
        previous_top_stocks_history: List of previous day's top stocks (most recent last)
        current_top_stocks: Today's top stocks
        min_stable_days: Minimum consecutive days of stability required
        min_overlap: Minimum overlap ratio to consider "stable"
    
    Returns:
        (is_stable, updated_history)
    """
    if not current_top_stocks:
        return False, []
    
    # Add current to history
    updated_history = previous_top_stocks_history.copy()
    updated_history.append(current_top_stocks)
    
    # Keep only last N+1 days (we need N comparisons)
    if len(updated_history) > min_stable_days + 1:
        updated_history = updated_history[-(min_stable_days + 1):]
    
    # Need at least min_stable_days + 1 entries to check stability
    if len(updated_history) < min_stable_days + 1:
        return False, updated_history
    
    # Check if all consecutive days have sufficient overlap
    for i in range(1, len(updated_history)):
        prev_set = set(updated_history[i-1])
        curr_set = set(updated_history[i])
        if len(prev_set) == 0:
            return False, updated_history
        overlap = len(prev_set & curr_set) / len(prev_set)
        if overlap < min_overlap:
            return False, updated_history
    
    return True, updated_history


def select_momentum_persistence_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    previous_top_stocks_history: List[List[str]],
    current_stocks: List[str],
    top_n: int = 10,
    min_stable_days: int = 5,
    min_overlap: float = 0.8
) -> Tuple[List[str], bool, List[List[str]]]:
    """
    Static BH 1Y with momentum persistence requirement.
    Only rebalances when top performers have been stable for N consecutive days.
    
    Args:
        tickers: List of tickers to consider
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        previous_top_stocks_history: History of previous top stocks
        current_stocks: Currently held stocks
        top_n: Number of stocks to select
        min_stable_days: Minimum consecutive days of stability
        min_overlap: Minimum overlap ratio for stability
    
    Returns:
        (selected_stocks, should_rebalance, updated_history)
    """
    # Get current top performers
    new_top_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # Check momentum persistence
    is_stable, updated_history = check_momentum_persistence(
        previous_top_stocks_history, new_top_stocks, min_stable_days, min_overlap
    )
    
    # Only rebalance if:
    # 1. No current stocks (initialization)
    # 2. Top stocks have been stable AND they differ from current holdings
    if not current_stocks:
        return new_top_stocks, True, updated_history
    
    if is_stable:
        current_set = set(current_stocks)
        new_set = set(new_top_stocks)
        # Check if the stable top stocks are different from what we hold
        overlap_with_current = len(current_set & new_set) / len(current_set) if current_set else 0
        if overlap_with_current < 0.9:  # At least 1 stock different
            return new_top_stocks, True, updated_history
    
    return current_stocks, False, updated_history


def select_overlap_based_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    current_stocks: List[str],
    top_n: int = 10,
    overlap_threshold: float = 0.7
) -> Tuple[List[str], bool]:
    """
    Static BH 1Y with overlap-based rebalancing.
    Only rebalances when overlap between current holdings and new top performers
    drops below threshold.
    
    Args:
        tickers: List of tickers to consider
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        current_stocks: Currently held stocks
        top_n: Number of stocks to select
        overlap_threshold: Minimum overlap to avoid rebalancing (default 70%)
    
    Returns:
        (selected_stocks, should_rebalance)
    """
    # Get current top performers
    new_top_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # No current stocks - always rebalance (initialization)
    if not current_stocks:
        return new_top_stocks, True
    
    # Calculate overlap
    current_set = set(current_stocks)
    new_set = set(new_top_stocks)
    overlap = len(current_set & new_set) / len(current_set) if current_set else 0
    
    # Rebalance if overlap drops below threshold
    should_rebalance = overlap < overlap_threshold
    
    if should_rebalance:
        return new_top_stocks, True
    else:
        return current_stocks, False
