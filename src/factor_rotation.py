"""
Factor Rotation Strategy

Rotates between factors (Value, Growth, Momentum, Quality) based on market regime.
Features:
- Market regime detection using VIX and yield curve
- Dynamic factor allocation
- Automatic rotation based on economic indicators
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

# Market regime thresholds
VIX_LOW_THRESHOLD = 15  # Below this = low volatility regime
VIX_HIGH_THRESHOLD = 25  # Above this = high volatility regime

# Factor weights by regime
FACTOR_WEIGHTS = {
    'low_vol': {  # Bull market, low fear
        'momentum': 0.40,
        'growth': 0.30,
        'quality': 0.20,
        'value': 0.10,
    },
    'normal': {  # Normal market
        'momentum': 0.25,
        'growth': 0.25,
        'quality': 0.25,
        'value': 0.25,
    },
    'high_vol': {  # Bear market, high fear
        'momentum': 0.10,
        'growth': 0.10,
        'quality': 0.40,
        'value': 0.40,
    },
}

# Factor calculation parameters
MOMENTUM_LOOKBACK = 252  # 1 year for momentum
VALUE_METRICS = ['pe_ratio', 'pb_ratio', 'dividend_yield']
QUALITY_METRICS = ['roe', 'debt_to_equity', 'profit_margin']
GROWTH_LOOKBACK = 63  # 3 months for growth


class FactorRotation:
    """Factor Rotation Strategy Implementation."""
    
    def __init__(self):
        self.current_regime = 'normal'
        self.last_rotation_date = None
        self.factor_scores = {}
    
    def detect_market_regime(self, ticker_data_grouped: Dict[str, pd.DataFrame],
                            current_date: datetime) -> str:
        """Detect current market regime based on volatility."""
        try:
            # Try to get VIX data (^VIX or VIX)
            vix_tickers = ['^VIX', 'VIX', 'VIXY']
            vix_value = None
            
            for vix_ticker in vix_tickers:
                if vix_ticker in ticker_data_grouped:
                    vix_data = ticker_data_grouped[vix_ticker]
                    vix_data = vix_data[vix_data.index <= current_date]
                    if len(vix_data) > 0:
                        vix_value = vix_data['Close'].iloc[-1]
                        break
            
            # If no VIX, estimate from market volatility
            if vix_value is None:
                # Use SPY or market proxy volatility
                market_tickers = ['SPY', 'QQQ', 'IWM']
                for market_ticker in market_tickers:
                    if market_ticker in ticker_data_grouped:
                        market_data = ticker_data_grouped[market_ticker]
                        market_data = market_data[market_data.index <= current_date]
                        if len(market_data) >= 20:
                            returns = market_data['Close'].pct_change().dropna()
                            vol = returns.tail(20).std() * np.sqrt(252) * 100
                            vix_value = vol
                            break
            
            if vix_value is None:
                return 'normal'
            
            # Determine regime
            if vix_value < VIX_LOW_THRESHOLD:
                return 'low_vol'
            elif vix_value > VIX_HIGH_THRESHOLD:
                return 'high_vol'
            else:
                return 'normal'
                
        except Exception as e:
            return 'normal'
    
    def calculate_momentum_score(self, ticker_data: pd.DataFrame, 
                                 current_date: datetime) -> float:
        """Calculate momentum factor score (12-month return minus last month)."""
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < MOMENTUM_LOOKBACK:
                return 0.0
            
            # 12-month return
            price_now = data['Close'].iloc[-1]
            price_12m = data['Close'].iloc[-MOMENTUM_LOOKBACK]
            price_1m = data['Close'].iloc[-21] if len(data) >= 21 else price_now
            
            # Momentum = 12M return - 1M return (to avoid short-term reversal)
            return_12m = (price_now - price_12m) / price_12m
            return_1m = (price_now - price_1m) / price_1m
            
            momentum = return_12m - return_1m
            return momentum
            
        except Exception as e:
            return 0.0
    
    def calculate_value_score(self, ticker_data: pd.DataFrame,
                             current_date: datetime) -> float:
        """Calculate value factor score based on price trends (proxy for valuation)."""
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < 252:
                return 0.0
            
            # Use price-to-52-week-high as value proxy
            # Lower ratio = more "value" (beaten down)
            price_now = data['Close'].iloc[-1]
            high_52w = data['Close'].iloc[-252:].max()
            
            # Invert so lower price relative to high = higher value score
            value_score = 1 - (price_now / high_52w)
            return value_score
            
        except Exception as e:
            return 0.0
    
    def calculate_quality_score(self, ticker_data: pd.DataFrame,
                               current_date: datetime) -> float:
        """Calculate quality factor score based on stability and consistency."""
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < 252:
                return 0.0
            
            returns = data['Close'].pct_change().dropna()
            
            # Quality metrics from price data:
            # 1. Lower volatility = higher quality
            vol = returns.tail(252).std() * np.sqrt(252)
            vol_score = max(0, 1 - vol)  # Invert: lower vol = higher score
            
            # 2. Positive return consistency
            positive_days = (returns.tail(252) > 0).sum() / 252
            
            # 3. Drawdown resilience
            rolling_max = data['Close'].tail(252).cummax()
            drawdown = (data['Close'].tail(252) - rolling_max) / rolling_max
            max_dd = abs(drawdown.min())
            dd_score = max(0, 1 - max_dd)
            
            quality_score = (vol_score + positive_days + dd_score) / 3
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def calculate_growth_score(self, ticker_data: pd.DataFrame,
                              current_date: datetime) -> float:
        """Calculate growth factor score based on recent price acceleration."""
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < GROWTH_LOOKBACK * 2:
                return 0.0
            
            # Growth = acceleration in returns
            price_now = data['Close'].iloc[-1]
            price_3m = data['Close'].iloc[-GROWTH_LOOKBACK]
            price_6m = data['Close'].iloc[-GROWTH_LOOKBACK * 2]
            
            return_3m = (price_now - price_3m) / price_3m
            return_prev_3m = (price_3m - price_6m) / price_6m
            
            # Growth score = recent return acceleration
            growth_score = return_3m - return_prev_3m
            return growth_score
            
        except Exception as e:
            return 0.0
    
    def calculate_combined_score(self, ticker: str, ticker_data: pd.DataFrame,
                                current_date: datetime, regime: str) -> float:
        """Calculate combined factor score based on current regime."""
        weights = FACTOR_WEIGHTS.get(regime, FACTOR_WEIGHTS['normal'])
        
        momentum = self.calculate_momentum_score(ticker_data, current_date)
        value = self.calculate_value_score(ticker_data, current_date)
        quality = self.calculate_quality_score(ticker_data, current_date)
        growth = self.calculate_growth_score(ticker_data, current_date)
        
        combined = (
            weights['momentum'] * momentum +
            weights['value'] * value +
            weights['quality'] * quality +
            weights['growth'] * growth
        )
        
        return combined
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Select stocks based on factor rotation."""
        print(f"\n   🎯 Factor Rotation Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        # Detect market regime
        regime = self.detect_market_regime(ticker_data_grouped, current_date)
        self.current_regime = regime
        print(f"   📊 Market Regime: {regime.upper()}")
        print(f"   📊 Factor Weights: {FACTOR_WEIGHTS[regime]}")
        
        # Calculate scores for all tickers
        scored_tickers = []
        
        for ticker in all_tickers:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) < 252:
                continue
            
            score = self.calculate_combined_score(ticker, ticker_data, current_date, regime)
            scored_tickers.append((ticker, score))
        
        # Sort by score
        scored_tickers.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = [ticker for ticker, score in scored_tickers[:top_n]]
        
        print(f"   ✅ Analyzed {len(scored_tickers)} tickers")
        print(f"   ✅ Selected {len(selected)} stocks:")
        for ticker, score in scored_tickers[:top_n]:
            print(f"      {ticker}: Score={score:.4f}")
        
        return selected


# Global instance
_factor_rotation_instance = None

def get_factor_rotation_instance() -> FactorRotation:
    """Get or create the global factor rotation instance."""
    global _factor_rotation_instance
    if _factor_rotation_instance is None:
        _factor_rotation_instance = FactorRotation()
    return _factor_rotation_instance


def select_factor_rotation_stocks(all_tickers: List[str],
                                  ticker_data_grouped: Dict[str, pd.DataFrame],
                                  current_date: datetime = None,
                                  train_start_date: datetime = None,
                                  top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Factor Rotation stock selection strategy.
    
    Rotates between factors based on market regime.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    instance = get_factor_rotation_instance()
    return instance.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_factor_rotation_state():
    """Reset the global factor rotation instance."""
    global _factor_rotation_instance
    _factor_rotation_instance = None
