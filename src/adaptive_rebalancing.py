#!/usr/bin/env python3
"""Adaptive rebalancing strategies for Static BH 1Y."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

def check_volatility_trigger(
    positions: Dict[str, Dict],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    threshold: float = 0.02
) -> bool:
    """
    Check if portfolio volatility exceeds threshold.
    
    Args:
        positions: Current portfolio positions
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        threshold: Volatility threshold (default 2%)
    
    Returns:
        True if portfolio 20D volatility > threshold
    """
    if not positions:
        return False
    
    returns = []
    
    for ticker in positions:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if len(data) >= 20:
                # Calculate daily returns for last 20 days
                prices = data['Close'].iloc[-20:]
                ticker_returns = prices.pct_change().dropna()
                if len(ticker_returns) > 0:
                    returns.append(ticker_returns)
    
    if not returns:
        return False
    
    # Calculate portfolio returns (equal weight)
    portfolio_returns = pd.concat(returns, axis=1).mean(axis=1)
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
    
    return portfolio_vol > threshold


def check_performance_trigger(
    positions: Dict[str, Dict],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    target_weight: float = 0.1,  # 10% for 10 positions
    threshold: float = 0.25
) -> bool:
    """
    Check if any position deviates from target weight by more than threshold.
    
    Args:
        positions: Current portfolio positions
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        target_weight: Target weight for each position
        threshold: Deviation threshold (default 25%)
    
    Returns:
        True if max deviation > threshold
    """
    if not positions:
        return False
    
    current_values = []
    total_value = 0
    
    # Calculate current values
    for ticker, pos in positions.items():
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                value = pos['shares'] * current_price
                current_values.append(value)
                total_value += value
    
    if total_value == 0:
        return False
    
    # Check weight deviations
    for value in current_values:
        current_weight = value / total_value
        deviation = abs(current_weight - target_weight) / target_weight
        if deviation > threshold:
            return True
    
    return False


def check_momentum_trigger(
    positions: Dict[str, Dict],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    lookback_days: int = 60
) -> bool:
    """
    Check if portfolio momentum has turned negative.
    
    Args:
        positions: Current portfolio positions
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        lookback_days: Days to look back for momentum calculation
    
    Returns:
        True if portfolio momentum < 0
    """
    if not positions:
        return False
    
    portfolio_values = []
    
    for ticker in positions:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if len(data) >= lookback_days:
                # Get prices at start and end of lookback period
                start_date = current_date - timedelta(days=lookback_days)
                period_data = data[data.index >= start_date]
                if len(period_data) >= 2:
                    start_price = period_data['Close'].iloc[0]
                    end_price = period_data['Close'].iloc[-1]
                    portfolio_values.append(end_price / start_price - 1)
    
    if not portfolio_values:
        return False
    
    # Average momentum across positions
    avg_momentum = np.mean(portfolio_values)
    return avg_momentum < 0


def check_atr_trigger(
    positions: Dict[str, Dict],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    threshold: float = 0.05
) -> bool:
    """
    Check if cumulative ATR change exceeds threshold.
    
    Args:
        positions: Current portfolio positions
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        threshold: ATR change threshold (default 5%)
    
    Returns:
        True if cumulative ATR change > threshold
    """
    if not positions:
        return False
    
    atr_changes = []
    
    for ticker in positions:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if len(data) >= 14:
                # Calculate 14-day ATR
                high = data['High'].iloc[-14:]
                low = data['Low'].iloc[-14:]
                close = data['Close'].iloc[-14:]
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.mean()
                
                # Calculate ATR as percentage of price
                current_price = data['Close'].iloc[-1]
                atr_pct = atr / current_price
                atr_changes.append(atr_pct)
    
    if not atr_changes:
        return False
    
    # Average ATR percentage across positions
    avg_atr_pct = np.mean(atr_changes)
    return avg_atr_pct > threshold


def check_hybrid_trigger(
    positions: Dict[str, Dict],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    days_since_rebalance: int,
    max_days: int = 15,
    vol_threshold: float = 0.02,
    perf_threshold: float = 0.25,
    momentum_lookback: int = 60,
    atr_threshold: float = 0.05
) -> bool:
    """
    Hybrid approach combining multiple triggers with safety net.
    
    Args:
        positions: Current portfolio positions
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        days_since_rebalance: Days since last rebalance
        max_days: Maximum days without rebalance (safety net)
        vol_threshold: Volatility threshold
        perf_threshold: Performance deviation threshold
        momentum_lookback: Momentum lookback period
        atr_threshold: ATR threshold
    
    Returns:
        True if any trigger is hit or safety net is exceeded
    """
    # Safety net: always rebalance if too many days have passed
    if days_since_rebalance >= max_days:
        return True
    
    # Check individual triggers
    triggers = [
        check_volatility_trigger(positions, ticker_data_grouped, current_date, vol_threshold),
        check_performance_trigger(positions, ticker_data_grouped, current_date, threshold=perf_threshold),
        check_momentum_trigger(positions, ticker_data_grouped, current_date, momentum_lookback),
        check_atr_trigger(positions, ticker_data_grouped, current_date, atr_threshold)
    ]
    
    return any(triggers)
