"""
Options-Based Sentiment Strategy

Uses put/call ratios and unusual options activity as trading signals.
Features:
- Put/Call ratio analysis
- Unusual options volume detection
- Implied volatility skew analysis
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

# Put/Call ratio thresholds
PCR_BULLISH_THRESHOLD = 0.7  # Below this = bullish (more calls than puts)
PCR_BEARISH_THRESHOLD = 1.3  # Above this = bearish (more puts than calls)
PCR_EXTREME_LOW = 0.5  # Extreme bullish
PCR_EXTREME_HIGH = 1.5  # Extreme bearish (contrarian buy signal)

# Unusual activity parameters
VOLUME_SPIKE_MULTIPLIER = 3.0  # Options volume 3x average = unusual
OI_CHANGE_THRESHOLD = 0.20  # 20% change in open interest

# IV parameters
IV_PERCENTILE_LOW = 20  # Low IV = cheap options
IV_PERCENTILE_HIGH = 80  # High IV = expensive options


class OptionsSentiment:
    """Options-Based Sentiment Strategy Implementation."""
    
    def __init__(self):
        # Simulated options data (in production, fetch from options data provider)
        self.options_data = self._generate_mock_options_data()
    
    def _generate_mock_options_data(self) -> Dict[str, Dict]:
        """
        Generate mock options data.
        In production, this would fetch from CBOE, options data API, etc.
        """
        mock_data = {
            'AAPL': {
                'put_volume': 150000,
                'call_volume': 250000,
                'avg_put_volume': 100000,
                'avg_call_volume': 120000,
                'iv_current': 25,
                'iv_percentile': 35,
                'unusual_calls': True,
            },
            'MSFT': {
                'put_volume': 80000,
                'call_volume': 120000,
                'avg_put_volume': 70000,
                'avg_call_volume': 90000,
                'iv_current': 22,
                'iv_percentile': 28,
                'unusual_calls': False,
            },
            'NVDA': {
                'put_volume': 200000,
                'call_volume': 350000,
                'avg_put_volume': 100000,
                'avg_call_volume': 150000,
                'iv_current': 45,
                'iv_percentile': 65,
                'unusual_calls': True,
            },
            'GOOGL': {
                'put_volume': 60000,
                'call_volume': 80000,
                'avg_put_volume': 55000,
                'avg_call_volume': 70000,
                'iv_current': 28,
                'iv_percentile': 42,
                'unusual_calls': False,
            },
            'META': {
                'put_volume': 120000,
                'call_volume': 90000,
                'avg_put_volume': 80000,
                'avg_call_volume': 100000,
                'iv_current': 35,
                'iv_percentile': 55,
                'unusual_calls': False,
            },
            'TSLA': {
                'put_volume': 300000,
                'call_volume': 500000,
                'avg_put_volume': 200000,
                'avg_call_volume': 250000,
                'iv_current': 55,
                'iv_percentile': 70,
                'unusual_calls': True,
            },
            'AMD': {
                'put_volume': 100000,
                'call_volume': 180000,
                'avg_put_volume': 80000,
                'avg_call_volume': 100000,
                'iv_current': 42,
                'iv_percentile': 48,
                'unusual_calls': True,
            },
        }
        return mock_data
    
    def calculate_put_call_ratio(self, ticker: str) -> Tuple[float, str]:
        """
        Calculate put/call ratio and interpret signal.
        Returns: (pcr, signal)
        """
        if ticker not in self.options_data:
            return 1.0, 'neutral'
        
        data = self.options_data[ticker]
        put_vol = data['put_volume']
        call_vol = data['call_volume']
        
        if call_vol == 0:
            return 999.0, 'bearish'
        
        pcr = put_vol / call_vol
        
        # Interpret signal
        if pcr < PCR_EXTREME_LOW:
            signal = 'extreme_bullish'
        elif pcr < PCR_BULLISH_THRESHOLD:
            signal = 'bullish'
        elif pcr > PCR_EXTREME_HIGH:
            signal = 'contrarian_bullish'  # Extreme fear = buy opportunity
        elif pcr > PCR_BEARISH_THRESHOLD:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        return pcr, signal
    
    def detect_unusual_activity(self, ticker: str) -> Tuple[bool, str]:
        """
        Detect unusual options activity.
        Returns: (has_unusual, activity_type)
        """
        if ticker not in self.options_data:
            return False, 'none'
        
        data = self.options_data[ticker]
        
        # Check for volume spikes
        call_spike = data['call_volume'] > data['avg_call_volume'] * VOLUME_SPIKE_MULTIPLIER
        put_spike = data['put_volume'] > data['avg_put_volume'] * VOLUME_SPIKE_MULTIPLIER
        
        if call_spike and not put_spike:
            return True, 'unusual_calls'
        elif put_spike and not call_spike:
            return True, 'unusual_puts'
        elif call_spike and put_spike:
            return True, 'unusual_both'
        
        # Check explicit unusual flag
        if data.get('unusual_calls', False):
            return True, 'unusual_calls'
        
        return False, 'none'
    
    def get_iv_signal(self, ticker: str) -> Tuple[float, str]:
        """
        Get implied volatility signal.
        Returns: (iv_percentile, signal)
        """
        if ticker not in self.options_data:
            return 50, 'neutral'
        
        data = self.options_data[ticker]
        iv_pct = data['iv_percentile']
        
        if iv_pct < IV_PERCENTILE_LOW:
            signal = 'low_iv'  # Good for buying options/stock
        elif iv_pct > IV_PERCENTILE_HIGH:
            signal = 'high_iv'  # Expensive, potential mean reversion
        else:
            signal = 'neutral'
        
        return iv_pct, signal
    
    def calculate_sentiment_score(self, ticker: str, ticker_data: pd.DataFrame = None,
                                  current_date: datetime = None) -> float:
        """Calculate overall options sentiment score."""
        score = 0.0
        
        # Put/Call ratio component
        pcr, pcr_signal = self.calculate_put_call_ratio(ticker)
        if pcr_signal == 'bullish':
            score += 2.0
        elif pcr_signal == 'extreme_bullish':
            score += 3.0
        elif pcr_signal == 'contrarian_bullish':
            score += 2.5  # Extreme fear = opportunity
        elif pcr_signal == 'bearish':
            score -= 1.0
        
        # Unusual activity component
        has_unusual, activity_type = self.detect_unusual_activity(ticker)
        if has_unusual:
            if activity_type == 'unusual_calls':
                score += 2.0  # Smart money buying calls
            elif activity_type == 'unusual_puts':
                score -= 0.5  # Could be hedging or bearish
        
        # IV component
        iv_pct, iv_signal = self.get_iv_signal(ticker)
        if iv_signal == 'low_iv':
            score += 1.0  # Cheap options = good entry
        elif iv_signal == 'high_iv':
            score += 0.5  # High IV can mean big move coming
        
        # If no options data, estimate from price action
        if ticker not in self.options_data and ticker_data is not None:
            score = self._estimate_sentiment_from_price(ticker_data, current_date)
        
        return score
    
    def _estimate_sentiment_from_price(self, ticker_data: pd.DataFrame,
                                       current_date: datetime) -> float:
        """Estimate sentiment from price patterns when no options data available."""
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < 20:
                return 0.0
            
            recent = data.tail(20)
            
            # Bullish patterns
            price_trend = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
            
            # Volume trend
            if 'Volume' in recent.columns:
                vol_trend = recent['Volume'].tail(5).mean() / recent['Volume'].head(5).mean()
            else:
                vol_trend = 1.0
            
            # Score based on patterns
            score = 0.0
            if price_trend > 0.05 and vol_trend > 1.2:
                score = 1.5  # Bullish with volume confirmation
            elif price_trend > 0.02:
                score = 0.5
            elif price_trend < -0.05:
                score = -0.5
            
            return score
            
        except Exception as e:
            return 0.0
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Select stocks with bullish options sentiment."""
        print(f"\n   🎯 Options-Based Sentiment Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        candidates = []
        
        for ticker in all_tickers:
            ticker_data = ticker_data_grouped.get(ticker)
            score = self.calculate_sentiment_score(ticker, ticker_data, current_date)
            
            if score > 0:
                pcr, pcr_signal = self.calculate_put_call_ratio(ticker)
                has_unusual, activity_type = self.detect_unusual_activity(ticker)
                candidates.append((ticker, score, pcr, pcr_signal, has_unusual, activity_type))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = [ticker for ticker, score, pcr, pcr_signal, has_unusual, activity_type in candidates[:top_n]]
        
        print(f"   ✅ Found {len(candidates)} bullish sentiment candidates")
        print(f"   ✅ Selected {len(selected)} stocks:")
        for ticker, score, pcr, pcr_signal, has_unusual, activity_type in candidates[:top_n]:
            unusual_str = f" [{activity_type.upper()}]" if has_unusual else ""
            print(f"      {ticker}: Score={score:.2f}, PCR={pcr:.2f} ({pcr_signal}){unusual_str}")
        
        return selected


# Global instance
_options_instance = None

def get_options_instance() -> OptionsSentiment:
    """Get or create the global options sentiment instance."""
    global _options_instance
    if _options_instance is None:
        _options_instance = OptionsSentiment()
    return _options_instance


def select_options_sentiment_stocks(all_tickers: List[str],
                                    ticker_data_grouped: Dict[str, pd.DataFrame],
                                    current_date: datetime = None,
                                    train_start_date: datetime = None,
                                    top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Options-Based Sentiment stock selection strategy.
    
    Selects stocks with bullish options sentiment.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    instance = get_options_instance()
    return instance.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_options_state():
    """Reset the global options sentiment instance."""
    global _options_instance
    _options_instance = None
