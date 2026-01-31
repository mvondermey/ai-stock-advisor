"""
New Advanced Trading Strategies

1. Momentum Acceleration - 3M momentum with acceleration filter
2. Concentrated 3M - Fewer positions with volatility filter
3. Dual Momentum - Absolute + relative momentum (Antonacci style)
4. Trend Following ATR - Trend following with ATR trailing stops
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from config import (
    PORTFOLIO_SIZE,
    MOM_ACCEL_LOOKBACK_DAYS, MOM_ACCEL_SHORT_LOOKBACK, MOM_ACCEL_MIN_ACCELERATION,
    CONCENTRATED_3M_POSITIONS, CONCENTRATED_3M_MAX_VOLATILITY, CONCENTRATED_3M_REBALANCE_DAYS,
    DUAL_MOM_LOOKBACK_DAYS, DUAL_MOM_ABSOLUTE_THRESHOLD, DUAL_MOM_POSITIONS, DUAL_MOM_RISK_OFF_TICKER,
    TREND_ATR_LOOKBACK_DAYS, TREND_ATR_PERIOD, TREND_ATR_TRAILING_MULT, TREND_ATR_ENTRY_BREAKOUT,
)


# ============================================
# 1. MOMENTUM ACCELERATION STRATEGY
# ============================================

def select_momentum_acceleration_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE
) -> List[str]:
    """
    Momentum Acceleration Strategy:
    - Require positive 3M momentum
    - Require momentum acceleration (current 1M > previous 1M)
    - Rank by acceleration-adjusted momentum score
    
    Returns:
        List of selected tickers
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    candidates = []
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            # Need enough data for 3M + 1M lookback
            if len(ticker_data) < MOM_ACCEL_LOOKBACK_DAYS + MOM_ACCEL_SHORT_LOOKBACK:
                continue
            
            # Convert current_date to pandas Timestamp with timezone
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
            
            # Calculate 3M momentum
            start_3m = current_ts - timedelta(days=MOM_ACCEL_LOOKBACK_DAYS)
            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]
            
            if len(data_3m) < 50:
                continue
            
            valid_close = data_3m['Close'].dropna()
            if len(valid_close) < 2:
                continue
            
            momentum_3m = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100
            
            # Skip if 3M momentum is negative
            if momentum_3m <= 0:
                continue
            
            # Calculate current 1M momentum (last 21 days)
            start_1m_current = current_ts - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK)
            data_1m_current = ticker_data[(ticker_data.index >= start_1m_current) & (ticker_data.index <= current_ts)]
            
            if len(data_1m_current) < 10:
                continue
            
            valid_1m = data_1m_current['Close'].dropna()
            if len(valid_1m) < 2:
                continue
            
            momentum_1m_current = (valid_1m.iloc[-1] / valid_1m.iloc[0] - 1) * 100
            
            # Calculate previous 1M momentum (21-42 days ago)
            start_1m_prev = current_ts - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK * 2)
            end_1m_prev = current_ts - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK)
            data_1m_prev = ticker_data[(ticker_data.index >= start_1m_prev) & (ticker_data.index <= end_1m_prev)]
            
            if len(data_1m_prev) < 10:
                continue
            
            valid_1m_prev = data_1m_prev['Close'].dropna()
            if len(valid_1m_prev) < 2:
                continue
            
            momentum_1m_prev = (valid_1m_prev.iloc[-1] / valid_1m_prev.iloc[0] - 1) * 100
            
            # Calculate acceleration
            acceleration = momentum_1m_current - momentum_1m_prev
            
            # Skip if acceleration is below threshold
            if acceleration < MOM_ACCEL_MIN_ACCELERATION:
                continue
            
            # Score: 3M momentum weighted by acceleration
            score = momentum_3m * (1 + acceleration / 100)
            
            candidates.append((ticker, score, momentum_3m, acceleration))
            
        except Exception:
            continue
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _, _, _ in candidates[:top_n]]
        
        print(f"   📈 Momentum Acceleration: Selected {len(selected)} stocks")
        for t, score, mom3m, accel in candidates[:min(5, len(candidates))]:
            print(f"      {t}: score={score:.1f}, 3M={mom3m:+.1f}%, accel={accel:+.1f}%")
        
        return selected
    
    print(f"   ❌ Momentum Acceleration: No candidates found")
    return []


# ============================================
# 2. CONCENTRATED 3M + VOL FILTER STRATEGY
# ============================================

def select_concentrated_3m_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = None
) -> List[str]:
    """
    Concentrated 3M Strategy:
    - Top performers by 3M momentum
    - Volatility filter (max 40% annualized)
    - Fewer positions (5 instead of 10)
    
    Returns:
        List of selected tickers
    """
    if top_n is None:
        top_n = CONCENTRATED_3M_POSITIONS
    
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    candidates = []
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            if len(ticker_data) < 90:
                continue
            
            # Convert current_date to pandas Timestamp with timezone
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
            
            # Calculate 3M momentum
            start_3m = current_ts - timedelta(days=90)
            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]
            
            if len(data_3m) < 50:
                continue
            
            valid_close = data_3m['Close'].dropna()
            if len(valid_close) < 2:
                continue
            
            momentum_3m = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100
            
            # Skip if 3M momentum is negative
            if momentum_3m <= 0:
                continue
            
            # Calculate volatility (30-day)
            returns = valid_close.pct_change().dropna()
            if len(returns) < 20:
                continue
            
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Skip if volatility exceeds threshold
            if volatility > CONCENTRATED_3M_MAX_VOLATILITY:
                continue
            
            candidates.append((ticker, momentum_3m, volatility))
            
        except Exception:
            continue
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _, _ in candidates[:top_n]]
        
        print(f"   🎯 Concentrated 3M: Selected {len(selected)} stocks (max {top_n})")
        for t, mom3m, vol in candidates[:min(5, len(candidates))]:
            print(f"      {t}: 3M={mom3m:+.1f}%, vol={vol*100:.1f}%")
        
        return selected
    
    print(f"   ❌ Concentrated 3M: No candidates found")
    return []


# ============================================
# 3. DUAL MOMENTUM STRATEGY
# ============================================

def select_dual_momentum_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = None
) -> Tuple[List[str], bool]:
    """
    Dual Momentum Strategy (Antonacci style):
    - Absolute momentum: Only buy if 3M return > threshold
    - Relative momentum: Pick top N by 3M return
    - Risk-off: Return empty list if market momentum negative (caller handles cash)
    
    Returns:
        Tuple of (selected tickers, is_risk_on)
    """
    if top_n is None:
        top_n = DUAL_MOM_POSITIONS
    
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return [], False
    
    candidates = []
    total_momentum = 0.0
    valid_count = 0
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            if len(ticker_data) < DUAL_MOM_LOOKBACK_DAYS:
                continue
            
            # Convert current_date to pandas Timestamp with timezone
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
            
            # Calculate momentum
            start_date = current_ts - timedelta(days=DUAL_MOM_LOOKBACK_DAYS)
            data = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= current_ts)]
            
            if len(data) < 50:
                continue
            
            valid_close = data['Close'].dropna()
            if len(valid_close) < 2:
                continue
            
            momentum = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100
            
            total_momentum += momentum
            valid_count += 1
            
            # Absolute momentum filter
            if momentum > DUAL_MOM_ABSOLUTE_THRESHOLD:
                candidates.append((ticker, momentum))
            
        except Exception:
            continue
    
    # Calculate market momentum (average of all stocks)
    market_momentum = total_momentum / valid_count if valid_count > 0 else 0
    
    # Risk-off if market momentum is negative
    is_risk_on = market_momentum > 0
    
    if not is_risk_on:
        print(f"   ⚠️ Dual Momentum: RISK-OFF mode (market momentum: {market_momentum:.1f}%)")
        return [], False
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in candidates[:top_n]]
        
        print(f"   📊 Dual Momentum: RISK-ON, selected {len(selected)} stocks")
        print(f"      Market momentum: {market_momentum:+.1f}%")
        for t, mom in candidates[:min(5, len(candidates))]:
            print(f"      {t}: {mom:+.1f}%")
        
        return selected, True
    
    print(f"   ❌ Dual Momentum: No candidates with positive absolute momentum")
    return [], True


# ============================================
# 4. TREND FOLLOWING WITH ATR TRAILING STOP
# ============================================

class TrendFollowingATR:
    """
    Trend Following Strategy with ATR-based trailing stops.
    Tracks positions and their trailing stops.
    """
    
    def __init__(self):
        self.positions = {}  # ticker -> {'entry_price': float, 'peak_price': float, 'atr': float}
    
    def calculate_atr(self, ticker_data: pd.DataFrame, period: int = None) -> float:
        """Calculate Average True Range."""
        if period is None:
            period = TREND_ATR_PERIOD
        
        if len(ticker_data) < period + 5:
            return 0.0
        
        high_low = ticker_data['High'] - ticker_data['Low']
        high_close = abs(ticker_data['High'] - ticker_data['Close'].shift(1))
        low_close = abs(ticker_data['Low'] - ticker_data['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def is_breakout(self, ticker_data: pd.DataFrame, current_price: float) -> bool:
        """Check if current price is a breakout above N-day high."""
        if len(ticker_data) < TREND_ATR_ENTRY_BREAKOUT:
            return False
        
        high_n_days = ticker_data['High'].iloc[-TREND_ATR_ENTRY_BREAKOUT:-1].max()
        return current_price > high_n_days
    
    def check_trailing_stop(self, ticker: str, current_price: float) -> bool:
        """Check if trailing stop is hit. Returns True if should sell."""
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        
        # Update peak price
        if current_price > pos['peak_price']:
            pos['peak_price'] = current_price
        
        # Calculate trailing stop level
        stop_level = pos['peak_price'] - (pos['atr'] * TREND_ATR_TRAILING_MULT)
        
        return current_price < stop_level
    
    def add_position(self, ticker: str, entry_price: float, atr: float):
        """Add a new position."""
        self.positions[ticker] = {
            'entry_price': entry_price,
            'peak_price': entry_price,
            'atr': atr
        }
    
    def remove_position(self, ticker: str):
        """Remove a position."""
        if ticker in self.positions:
            del self.positions[ticker]
    
    def get_positions(self) -> Dict:
        """Get current positions."""
        return self.positions.copy()


# Global instance for state persistence
_trend_atr_instance = None

def get_trend_atr_instance() -> TrendFollowingATR:
    """Get or create the global trend ATR instance."""
    global _trend_atr_instance
    if _trend_atr_instance is None:
        _trend_atr_instance = TrendFollowingATR()
    return _trend_atr_instance

def reset_trend_atr_state():
    """Reset the global trend ATR instance."""
    global _trend_atr_instance
    _trend_atr_instance = None


def select_trend_following_atr_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE
) -> Tuple[List[str], List[str]]:
    """
    Trend Following ATR Strategy:
    - Enter on breakout above N-day high
    - Exit on ATR trailing stop
    - Require positive 3M momentum for entry
    
    Returns:
        Tuple of (stocks_to_buy, stocks_to_sell)
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return [], []
    
    trend_tracker = get_trend_atr_instance()
    
    stocks_to_buy = []
    stocks_to_sell = []
    breakout_candidates = []
    
    # Check existing positions for trailing stop exits
    for ticker in list(trend_tracker.positions.keys()):
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
            
            data = ticker_data[ticker_data.index <= current_ts]
            if len(data) < 1:
                continue
            
            current_price = data['Close'].iloc[-1]
            
            if trend_tracker.check_trailing_stop(ticker, current_price):
                stocks_to_sell.append(ticker)
                print(f"      🛑 {ticker}: Trailing stop hit @ ${current_price:.2f}")
        except Exception:
            continue
    
    # Remove sold positions
    for ticker in stocks_to_sell:
        trend_tracker.remove_position(ticker)
    
    # Find new breakout candidates
    for ticker in all_tickers:
        try:
            # Skip if already in position
            if ticker in trend_tracker.positions:
                continue
            
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            if len(ticker_data) < TREND_ATR_LOOKBACK_DAYS:
                continue
            
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
            
            data = ticker_data[ticker_data.index <= current_ts]
            if len(data) < TREND_ATR_ENTRY_BREAKOUT + 10:
                continue
            
            current_price = data['Close'].iloc[-1]
            
            # Check 3M momentum first
            start_3m = current_ts - timedelta(days=TREND_ATR_LOOKBACK_DAYS)
            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]
            
            if len(data_3m) < 50:
                continue
            
            valid_close = data_3m['Close'].dropna()
            if len(valid_close) < 2:
                continue
            
            momentum_3m = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100
            
            # Skip if momentum is negative
            if momentum_3m <= 0:
                continue
            
            # Check for breakout
            if trend_tracker.is_breakout(data, current_price):
                atr = trend_tracker.calculate_atr(data)
                if atr > 0:
                    breakout_candidates.append((ticker, momentum_3m, current_price, atr))
        
        except Exception:
            continue
    
    # Select top breakouts by momentum
    if breakout_candidates:
        breakout_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Limit new entries based on available slots
        current_positions = len(trend_tracker.positions)
        available_slots = max(0, top_n - current_positions)
        
        for ticker, momentum, price, atr in breakout_candidates[:available_slots]:
            stocks_to_buy.append(ticker)
            trend_tracker.add_position(ticker, price, atr)
            print(f"      🚀 {ticker}: Breakout entry @ ${price:.2f}, ATR=${atr:.2f}, 3M={momentum:+.1f}%")
    
    print(f"   📈 Trend Following ATR: {len(stocks_to_buy)} buys, {len(stocks_to_sell)} sells, {len(trend_tracker.positions)} positions")
    
    return stocks_to_buy, stocks_to_sell


def get_trend_following_current_stocks() -> List[str]:
    """Get list of current trend following positions."""
    trend_tracker = get_trend_atr_instance()
    return list(trend_tracker.positions.keys())
