"""
Earnings Momentum (PEAD) Strategy

Post-Earnings Announcement Drift - buys stocks with positive earnings surprises.
Features:
- Earnings surprise detection
- SUE (Standardized Unexpected Earnings) scoring
- 60-90 day holding period for drift capture
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

# Earnings parameters
EARNINGS_LOOKBACK_DAYS = 30  # Look for earnings in last 30 days
MIN_SURPRISE_PCT = 5.0  # Minimum earnings surprise percentage
HOLDING_PERIOD_DAYS = 60  # Hold for 60 days after earnings

# Price reaction parameters
MIN_POST_EARNINGS_RETURN = 0.02  # 2% minimum post-earnings return
MAX_POST_EARNINGS_DAYS = 5  # Days after earnings to measure reaction

# Volume parameters
VOLUME_SURGE_MULTIPLIER = 1.5  # Volume surge on earnings day


class EarningsMomentum:
    """Earnings Momentum (PEAD) Strategy Implementation."""
    
    def __init__(self):
        self.earnings_dates = {}  # ticker -> last earnings date
        self.positions = {}  # ticker -> {'entry_date': date, 'entry_price': float}
    
    def detect_earnings_event(self, ticker_data: pd.DataFrame, 
                              current_date: datetime) -> Tuple[bool, datetime, float]:
        """
        Detect if there was a recent earnings event based on price/volume patterns.
        Returns: (has_earnings, earnings_date, estimated_surprise)
        """
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < EARNINGS_LOOKBACK_DAYS:
                return False, None, 0.0
            
            recent_data = data.tail(EARNINGS_LOOKBACK_DAYS)
            
            # Look for earnings signature:
            # 1. Large price gap (>3%)
            # 2. Volume spike (>2x average)
            
            if 'Volume' not in recent_data.columns:
                # Use price gaps only
                daily_returns = recent_data['Close'].pct_change()
                large_moves = daily_returns[abs(daily_returns) > 0.03]
                
                if len(large_moves) == 0:
                    return False, None, 0.0
                
                # Most recent large move
                earnings_idx = large_moves.index[-1]
                surprise = large_moves.iloc[-1] * 100  # Convert to percentage
                
                return True, earnings_idx, surprise
            
            # With volume data
            avg_volume = recent_data['Volume'].mean()
            daily_returns = recent_data['Close'].pct_change()
            
            # Find days with both price gap and volume spike
            for i in range(len(recent_data) - 1, -1, -1):
                date = recent_data.index[i]
                ret = daily_returns.iloc[i] if i < len(daily_returns) else 0
                vol = recent_data['Volume'].iloc[i]
                
                if abs(ret) > 0.03 and vol > avg_volume * VOLUME_SURGE_MULTIPLIER:
                    return True, date, ret * 100
            
            return False, None, 0.0
            
        except Exception as e:
            return False, None, 0.0
    
    def calculate_post_earnings_drift(self, ticker_data: pd.DataFrame,
                                      earnings_date: datetime,
                                      current_date: datetime) -> float:
        """Calculate the post-earnings price drift."""
        try:
            data = ticker_data[(ticker_data.index >= earnings_date) & 
                              (ticker_data.index <= current_date)]
            
            if len(data) < 2:
                return 0.0
            
            earnings_price = data['Close'].iloc[0]
            current_price = data['Close'].iloc[-1]
            
            drift = (current_price - earnings_price) / earnings_price
            return drift
            
        except Exception as e:
            return 0.0
    
    def calculate_sue_score(self, ticker_data: pd.DataFrame,
                           current_date: datetime) -> float:
        """
        Calculate Standardized Unexpected Earnings (SUE) proxy.
        Uses price momentum as proxy for earnings surprise.
        """
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < 90:
                return 0.0
            
            # Use 3-month return as earnings momentum proxy
            price_now = data['Close'].iloc[-1]
            price_3m = data['Close'].iloc[-63]
            
            return_3m = (price_now - price_3m) / price_3m
            
            # Standardize by volatility
            returns = data['Close'].pct_change().dropna()
            vol = returns.tail(63).std() * np.sqrt(252)
            
            if vol == 0:
                return 0.0
            
            sue = return_3m / vol
            return sue
            
        except Exception as e:
            return 0.0
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Select stocks with positive earnings momentum."""
        print(f"\n   🎯 Earnings Momentum (PEAD) Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        candidates = []
        
        for ticker in all_tickers:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) < 90:
                continue
            
            # Detect recent earnings
            has_earnings, earnings_date, surprise = self.detect_earnings_event(
                ticker_data, current_date
            )
            
            if has_earnings and surprise > MIN_SURPRISE_PCT:
                # Calculate post-earnings drift
                drift = self.calculate_post_earnings_drift(
                    ticker_data, earnings_date, current_date
                )
                
                # Calculate SUE score
                sue = self.calculate_sue_score(ticker_data, current_date)
                
                # Combined score
                score = surprise + (drift * 100) + (sue * 10)
                
                candidates.append((ticker, score, surprise, drift, sue))
        
        # Also include stocks with high SUE even without detected earnings
        for ticker in all_tickers:
            if any(c[0] == ticker for c in candidates):
                continue
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) < 90:
                continue
            
            sue = self.calculate_sue_score(ticker_data, current_date)
            
            if sue > 1.5:  # High SUE threshold
                candidates.append((ticker, sue * 10, 0, 0, sue))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = [ticker for ticker, score, surprise, drift, sue in candidates[:top_n]]
        
        print(f"   ✅ Found {len(candidates)} earnings momentum candidates")
        print(f"   ✅ Selected {len(selected)} stocks:")
        for ticker, score, surprise, drift, sue in candidates[:top_n]:
            if surprise > 0:
                print(f"      {ticker}: Surprise={surprise:.1f}%, Drift={drift*100:.1f}%, SUE={sue:.2f}")
            else:
                print(f"      {ticker}: SUE={sue:.2f}")
        
        return selected


# Global instance
_earnings_instance = None

def get_earnings_instance() -> EarningsMomentum:
    """Get or create the global earnings momentum instance."""
    global _earnings_instance
    if _earnings_instance is None:
        _earnings_instance = EarningsMomentum()
    return _earnings_instance


def select_earnings_momentum_stocks(all_tickers: List[str],
                                    ticker_data_grouped: Dict[str, pd.DataFrame],
                                    current_date: datetime = None,
                                    train_start_date: datetime = None,
                                    top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Earnings Momentum (PEAD) stock selection strategy.
    
    Selects stocks with positive earnings surprises.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    instance = get_earnings_instance()
    return instance.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_earnings_state():
    """Reset the global earnings momentum instance."""
    global _earnings_instance
    _earnings_instance = None
