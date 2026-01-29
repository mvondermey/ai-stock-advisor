"""
Momentum Breakout Strategy

Buys stocks breaking above 52-week highs with volume confirmation.
Features:
- 52-week high breakout detection
- Volume surge confirmation (2x average)
- Exit when price drops below 20-day MA
- Trend strength filtering using ADX
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from config import (
    TRANSACTION_COST,
    PORTFOLIO_SIZE,
)

# ============================================
# Configuration Parameters
# ============================================

# Breakout parameters
LOOKBACK_DAYS_52W = 252  # Trading days in a year
VOLUME_SURGE_MULTIPLIER = 1.5  # Volume must be 1.5x average
VOLUME_LOOKBACK_DAYS = 20  # Days to calculate average volume

# Exit parameters
EXIT_MA_PERIOD = 20  # Exit when price drops below 20-day MA
TRAILING_STOP_PCT = 0.10  # 10% trailing stop

# Trend filter
MIN_ADX = 20  # Minimum ADX for trend strength
ADX_PERIOD = 14

# Position sizing
MAX_POSITION_PCT = 0.20  # Max 20% in single stock


class MomentumBreakout:
    """Momentum Breakout Strategy Implementation."""
    
    def __init__(self):
        self.positions = {}  # ticker -> {'entry_price': float, 'high_since_entry': float}
        self.last_signals = {}
    
    def calculate_adx(self, ticker_data: pd.DataFrame, period: int = ADX_PERIOD) -> float:
        """Calculate Average Directional Index (ADX) for trend strength."""
        try:
            if len(ticker_data) < period * 2:
                return 0.0
            
            high = ticker_data['High'] if 'High' in ticker_data.columns else ticker_data['Close']
            low = ticker_data['Low'] if 'Low' in ticker_data.columns else ticker_data['Close']
            close = ticker_data['Close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            # Smoothed averages
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            return 0.0
    
    def is_52w_high_breakout(self, ticker_data: pd.DataFrame, current_date: datetime) -> Tuple[bool, float]:
        """Check if stock is breaking above 52-week high."""
        try:
            # Get data up to current date
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < LOOKBACK_DAYS_52W:
                return False, 0.0
            
            current_price = data['Close'].iloc[-1]
            high_52w = data['Close'].iloc[-LOOKBACK_DAYS_52W:-1].max()  # Exclude today
            
            # Breakout if current price > 52-week high
            is_breakout = current_price > high_52w
            breakout_pct = ((current_price - high_52w) / high_52w) * 100 if high_52w > 0 else 0
            
            return is_breakout, breakout_pct
            
        except Exception as e:
            return False, 0.0
    
    def has_volume_surge(self, ticker_data: pd.DataFrame, current_date: datetime) -> bool:
        """Check if current volume is significantly above average."""
        try:
            if 'Volume' not in ticker_data.columns:
                return True  # Assume valid if no volume data
            
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < VOLUME_LOOKBACK_DAYS + 1:
                return True
            
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].iloc[-VOLUME_LOOKBACK_DAYS-1:-1].mean()
            
            return current_volume >= avg_volume * VOLUME_SURGE_MULTIPLIER
            
        except Exception as e:
            return True
    
    def should_exit(self, ticker: str, ticker_data: pd.DataFrame, current_date: datetime) -> bool:
        """Check if position should be exited."""
        try:
            if ticker not in self.positions:
                return False
            
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < EXIT_MA_PERIOD:
                return False
            
            current_price = data['Close'].iloc[-1]
            ma_20 = data['Close'].iloc[-EXIT_MA_PERIOD:].mean()
            
            # Exit if price drops below 20-day MA
            if current_price < ma_20:
                return True
            
            # Trailing stop
            entry_price = self.positions[ticker]['entry_price']
            high_since_entry = self.positions[ticker].get('high_since_entry', entry_price)
            
            # Update high since entry
            if current_price > high_since_entry:
                self.positions[ticker]['high_since_entry'] = current_price
                high_since_entry = current_price
            
            # Check trailing stop
            if current_price < high_since_entry * (1 - TRAILING_STOP_PCT):
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Select stocks with momentum breakout signals."""
        print(f"\n   🎯 Momentum Breakout Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        breakout_candidates = []
        
        for ticker in all_tickers:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) < LOOKBACK_DAYS_52W:
                continue
            
            # Check for 52-week high breakout
            is_breakout, breakout_pct = self.is_52w_high_breakout(ticker_data, current_date)
            if not is_breakout:
                continue
            
            # Check volume confirmation
            if not self.has_volume_surge(ticker_data, current_date):
                continue
            
            # Check trend strength (ADX)
            recent_data = ticker_data[ticker_data.index <= current_date].tail(50)
            adx = self.calculate_adx(recent_data)
            if adx < MIN_ADX:
                continue
            
            # Score by breakout strength and ADX
            score = breakout_pct * (adx / 100)
            breakout_candidates.append((ticker, score, breakout_pct, adx))
        
        # Sort by score
        breakout_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = [ticker for ticker, score, bp, adx in breakout_candidates[:top_n]]
        
        print(f"   ✅ Found {len(breakout_candidates)} breakout candidates")
        print(f"   ✅ Selected {len(selected)} stocks:")
        for ticker, score, bp, adx in breakout_candidates[:top_n]:
            print(f"      {ticker}: Breakout +{bp:.1f}%, ADX={adx:.1f}")
        
        return selected


# Global instance
_breakout_instance = None

def get_breakout_instance() -> MomentumBreakout:
    """Get or create the global momentum breakout instance."""
    global _breakout_instance
    if _breakout_instance is None:
        _breakout_instance = MomentumBreakout()
    return _breakout_instance


def select_momentum_breakout_stocks(all_tickers: List[str],
                                    ticker_data_grouped: Dict[str, pd.DataFrame],
                                    current_date: datetime = None,
                                    train_start_date: datetime = None,
                                    top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Momentum Breakout stock selection strategy.
    
    Selects stocks breaking above 52-week highs with volume confirmation.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    instance = get_breakout_instance()
    return instance.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_breakout_state():
    """Reset the global momentum breakout instance."""
    global _breakout_instance
    _breakout_instance = None
