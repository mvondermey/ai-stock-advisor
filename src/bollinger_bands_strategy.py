"""
Bollinger Bands Strategies

4 distinct strategies using Bollinger Bands:
1. BB Mean Reversion - Buy at lower band, sell at upper band
2. BB Breakout - Buy when price breaks above upper band with volume
3. BB Squeeze + Breakout - Wait for squeeze, then trade the breakout
4. BB + RSI Combo - Buy when at lower band AND RSI < 30
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from config import PORTFOLIO_SIZE
from strategy_cache_adapter import (
    ensure_price_history_cache,
    get_cached_history_up_to,
    resolve_cache_current_date,
)


# ============================================
# CONFIGURATION
# ============================================

BB_PERIOD = 20  # Bollinger Bands period
BB_STD_DEV = 2.0  # Standard deviations for bands
RSI_PERIOD = 14  # RSI calculation period
RSI_OVERSOLD = 30  # RSI oversold threshold
RSI_OVERBOUGHT = 70  # RSI overbought threshold
SQUEEZE_THRESHOLD = 0.04  # Bandwidth threshold for squeeze (4%)
VOLUME_SURGE_MULT = 1.5  # Volume must be 1.5x average for breakout
MIN_DATA_DAYS = 50  # Minimum days of data required


# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_bollinger_bands(close: pd.Series, period: int = BB_PERIOD, std_dev: float = BB_STD_DEV) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_bandwidth(upper: pd.Series, lower: pd.Series, sma: pd.Series) -> pd.Series:
    """Calculate Bollinger Bandwidth (volatility indicator)."""
    return (upper - lower) / sma


def _get_cached_series(
    price_history_cache,
    ticker: str,
    current_date: datetime,
    field_name: str,
    min_rows: int,
) -> Optional[pd.Series]:
    values = get_cached_history_up_to(
        price_history_cache,
        ticker,
        current_date,
        field_name=field_name,
        min_rows=min_rows,
    )
    if values is None or len(values) < min_rows:
        return None
    return pd.Series(values)


def calculate_percent_b(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """Calculate %B (position within bands). 0 = lower, 1 = upper."""
    return (close - lower) / (upper - lower)


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# ============================================
# 1. BB MEAN REVERSION STRATEGY
# ============================================

def select_bb_mean_reversion_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE,
    price_history_cache=None,
) -> List[str]:
    """
    BB Mean Reversion Strategy:
    - Buy when price touches/crosses below lower band (oversold)
    - Rank by how far below the lower band (more oversold = higher priority)
    - Exit when price reaches middle band or upper band
    
    Returns:
        List of selected tickers
    """
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB Mean Reversion",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []
    
    candidates = []
    
    for ticker in filtered_tickers:
        try:
            close = _get_cached_series(
                price_history_cache, ticker, current_date, "close", MIN_DATA_DAYS
            )
            if close is None or len(close) < MIN_DATA_DAYS:
                continue

            upper, sma, lower = calculate_bollinger_bands(close)
            
            current_price = close.iloc[-1]
            current_lower = lower.iloc[-1]
            current_sma = sma.iloc[-1]
            
            if pd.isna(current_lower) or pd.isna(current_sma):
                continue
            
            # Check if price is at or below lower band
            percent_b = (current_price - current_lower) / (upper.iloc[-1] - current_lower) if (upper.iloc[-1] - current_lower) > 0 else 0.5
            
            # Only consider if %B < 0.2 (near or below lower band)
            if percent_b < 0.2:
                # Score by how oversold (lower %B = more oversold = higher score)
                oversold_score = 1 - percent_b
                candidates.append((ticker, oversold_score, percent_b, current_price))
        
        except Exception as e:
            print(f"   ⚠️ BB Mean Reversion error for {ticker}: {e}")
            continue
    
    # Sort by oversold score (most oversold first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected = [ticker for ticker, score, pct_b, price in candidates[:top_n]]
    
    print(f"   📊 BB Mean Reversion: {len(candidates)} oversold candidates, selected {len(selected)}")
    for ticker, score, pct_b, price in candidates[:min(5, len(candidates))]:
        print(f"      {ticker}: %B={pct_b:.2f}, Score={score:.2f}, Price=${price:.2f}")
    
    return selected


# ============================================
# 2. BB BREAKOUT STRATEGY
# ============================================

def select_bb_breakout_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE,
    price_history_cache=None,
) -> List[str]:
    """
    BB Breakout Strategy:
    - Buy when price breaks above upper band
    - Require volume confirmation (1.5x average)
    - Rank by breakout strength (how far above upper band)
    
    Returns:
        List of selected tickers
    """
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB Breakout",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []
    
    candidates = []
    
    for ticker in filtered_tickers:
        try:
            close = _get_cached_series(
                price_history_cache, ticker, current_date, "close", MIN_DATA_DAYS
            )
            volume = _get_cached_series(
                price_history_cache, ticker, current_date, "volume", MIN_DATA_DAYS
            )
            if close is None or len(close) < MIN_DATA_DAYS:
                continue

            upper, sma, lower = calculate_bollinger_bands(close)
            
            current_price = close.iloc[-1]
            current_upper = upper.iloc[-1]
            
            if pd.isna(current_upper):
                continue
            
            # Check if price is above upper band
            if current_price <= current_upper:
                continue
            
            # Check volume confirmation
            if volume is not None and len(volume) >= 20:
                avg_volume = volume.tail(20).mean()
                current_volume = volume.iloc[-1]
                if current_volume < avg_volume * VOLUME_SURGE_MULT:
                    continue  # No volume confirmation
            
            # Calculate breakout strength (% above upper band)
            breakout_pct = ((current_price - current_upper) / current_upper) * 100
            
            candidates.append((ticker, breakout_pct, current_price, current_upper))
        
        except Exception as e:
            print(f"   ⚠️ BB Breakout error for {ticker}: {e}")
            continue
    
    # Sort by breakout strength
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected = [ticker for ticker, bp, price, upper in candidates[:top_n]]
    
    print(f"   📊 BB Breakout: {len(candidates)} breakout candidates, selected {len(selected)}")
    for ticker, bp, price, upper in candidates[:min(5, len(candidates))]:
        print(f"      {ticker}: +{bp:.1f}% above upper band, Price=${price:.2f}")
    
    return selected


# ============================================
# 3. BB SQUEEZE + BREAKOUT STRATEGY
# ============================================

class BBSqueezeTracker:
    """Track squeeze states for BB Squeeze strategy."""
    
    def __init__(self):
        self.squeeze_states = {}  # ticker -> {'in_squeeze': bool, 'squeeze_days': int}
    
    def update_squeeze_state(self, ticker: str, bandwidth: float, avg_bandwidth: float) -> bool:
        """Update squeeze state. Returns True if currently in squeeze."""
        is_squeeze = bandwidth < avg_bandwidth * SQUEEZE_THRESHOLD / 0.04  # Normalize
        
        if ticker not in self.squeeze_states:
            self.squeeze_states[ticker] = {'in_squeeze': False, 'squeeze_days': 0}
        
        if is_squeeze:
            self.squeeze_states[ticker]['in_squeeze'] = True
            self.squeeze_states[ticker]['squeeze_days'] += 1
        else:
            # Reset if no longer in squeeze
            if self.squeeze_states[ticker]['in_squeeze']:
                self.squeeze_states[ticker]['in_squeeze'] = False
            self.squeeze_states[ticker]['squeeze_days'] = 0
        
        return is_squeeze
    
    def was_in_squeeze(self, ticker: str) -> Tuple[bool, int]:
        """Check if ticker was recently in squeeze."""
        if ticker not in self.squeeze_states:
            return False, 0
        state = self.squeeze_states[ticker]
        return state['in_squeeze'], state['squeeze_days']


_squeeze_tracker = None

def get_squeeze_tracker() -> BBSqueezeTracker:
    global _squeeze_tracker
    if _squeeze_tracker is None:
        _squeeze_tracker = BBSqueezeTracker()
    return _squeeze_tracker

def reset_squeeze_tracker():
    global _squeeze_tracker
    _squeeze_tracker = None


def select_bb_squeeze_breakout_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE,
    price_history_cache=None,
) -> List[str]:
    """
    BB Squeeze + Breakout Strategy:
    - Detect squeeze: Bandwidth < threshold (low volatility)
    - Wait for breakout: Price closes outside bands after squeeze
    - Enter in breakout direction
    - Rank by squeeze duration + breakout strength
    
    Returns:
        List of selected tickers
    """
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB Squeeze Breakout",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []
    
    tracker = get_squeeze_tracker()
    candidates = []
    
    for ticker in filtered_tickers:
        try:
            close = _get_cached_series(
                price_history_cache, ticker, current_date, "close", MIN_DATA_DAYS
            )
            if close is None or len(close) < MIN_DATA_DAYS:
                continue

            upper, sma, lower = calculate_bollinger_bands(close)
            bandwidth = calculate_bandwidth(upper, lower, sma)
            
            current_price = close.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            current_bw = bandwidth.iloc[-1]
            avg_bw = bandwidth.tail(50).mean() if len(bandwidth) >= 50 else bandwidth.mean()
            
            if pd.isna(current_bw) or pd.isna(avg_bw):
                continue
            
            # Check previous day's squeeze state
            prev_bw = bandwidth.iloc[-2] if len(bandwidth) >= 2 else current_bw
            was_squeeze = prev_bw < SQUEEZE_THRESHOLD
            
            # Update current squeeze state
            is_squeeze = tracker.update_squeeze_state(ticker, current_bw, avg_bw)
            
            # Look for breakout AFTER squeeze
            if was_squeeze and not is_squeeze:
                # Squeeze just ended - check for breakout
                if current_price > current_upper:
                    # Bullish breakout
                    breakout_strength = ((current_price - current_upper) / current_upper) * 100
                    _, squeeze_days = tracker.was_in_squeeze(ticker)
                    score = breakout_strength * (1 + squeeze_days * 0.1)  # Bonus for longer squeeze
                    candidates.append((ticker, score, breakout_strength, squeeze_days, 'BULL'))
                elif current_price < current_lower:
                    # Bearish breakout (skip for long-only)
                    pass
        
        except Exception as e:
            print(f"   ⚠️ BB Squeeze error for {ticker}: {e}")
            continue
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected = [ticker for ticker, score, bp, sq_days, direction in candidates[:top_n]]
    
    print(f"   📊 BB Squeeze Breakout: {len(candidates)} squeeze breakouts, selected {len(selected)}")
    for ticker, score, bp, sq_days, direction in candidates[:min(5, len(candidates))]:
        print(f"      {ticker}: {direction} +{bp:.1f}%, Squeeze={sq_days}d, Score={score:.1f}")
    
    return selected


# ============================================
# 4. BB + RSI COMBO STRATEGY
# ============================================

def select_bb_rsi_combo_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE,
    price_history_cache=None,
) -> List[str]:
    """
    BB + RSI Combo Strategy:
    - Buy when price at lower band AND RSI < 30 (double oversold confirmation)
    - Rank by combined oversold score
    
    Returns:
        List of selected tickers
    """
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB RSI Combo",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []
    
    candidates = []
    
    for ticker in filtered_tickers:
        try:
            close = _get_cached_series(
                price_history_cache, ticker, current_date, "close", MIN_DATA_DAYS
            )
            if close is None or len(close) < MIN_DATA_DAYS:
                continue

            upper, sma, lower = calculate_bollinger_bands(close)
            rsi = calculate_rsi(close)
            
            current_price = close.iloc[-1]
            current_lower = lower.iloc[-1]
            current_upper = upper.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_lower) or pd.isna(current_rsi):
                continue
            
            # Calculate %B
            percent_b = (current_price - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) > 0 else 0.5
            
            # Check both conditions: near lower band AND RSI oversold
            if percent_b < 0.2 and current_rsi < RSI_OVERSOLD:
                # Combined oversold score
                bb_score = 1 - percent_b  # Higher when more oversold
                rsi_score = (RSI_OVERSOLD - current_rsi) / RSI_OVERSOLD  # Higher when RSI lower
                combo_score = bb_score * 0.5 + rsi_score * 0.5
                
                candidates.append((ticker, combo_score, percent_b, current_rsi, current_price))
        
        except Exception as e:
            print(f"   ⚠️ BB RSI Combo error for {ticker}: {e}")
            continue
    
    # Sort by combo score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected = [ticker for ticker, score, pct_b, rsi, price in candidates[:top_n]]
    
    print(f"   📊 BB RSI Combo: {len(candidates)} double-oversold candidates, selected {len(selected)}")
    for ticker, score, pct_b, rsi, price in candidates[:min(5, len(candidates))]:
        print(f"      {ticker}: %B={pct_b:.2f}, RSI={rsi:.1f}, Score={score:.2f}")
    
    return selected


# ============================================
# RESET FUNCTIONS
# ============================================

def reset_bb_strategy_states():
    """Reset all BB strategy states."""
    reset_squeeze_tracker()
