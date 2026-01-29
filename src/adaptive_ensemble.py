"""
Adaptive Meta-Ensemble Strategy

Dynamically combines multiple strategies based on:
1. Recent strategy performance (rolling window)
2. Market regime detection (trending, volatile, mean-reverting)
3. Consensus filtering (only trade when multiple strategies agree)

This strategy adapts to changing market conditions instead of using fixed allocations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Import config
from config import (
    TRANSACTION_COST,
    PORTFOLIO_SIZE,
    AI_REBALANCE_FREQUENCY_DAYS,
)

# Import existing strategies
from shared_strategies import (
    select_dynamic_bh_stocks,
    select_risk_adj_mom_stocks,
    select_quality_momentum_stocks,
    select_volatility_adj_mom_stocks,
)

# ============================================
# Configuration Parameters
# ============================================

# Strategies to include in the ensemble
ENSEMBLE_STRATEGIES = [
    'static_bh_3m',      # Best raw returns
    'dyn_bh_1y_vol',     # Best risk-adjusted
    'risk_adj_mom',      # Consistent performer
    'quality_mom',       # Strong momentum + quality
]

# Rolling window for strategy performance evaluation (days)
STRATEGY_PERFORMANCE_WINDOW = 20

# Minimum number of strategies that must agree on a stock
MIN_CONSENSUS_AGREEMENT = 2

# Regime detection parameters
REGIME_VOLATILITY_THRESHOLD_HIGH = 0.025  # Daily volatility > 2.5% = high volatility regime
REGIME_VOLATILITY_THRESHOLD_LOW = 0.010   # Daily volatility < 1.0% = low volatility regime
REGIME_TREND_THRESHOLD = 0.05             # 5% monthly return = trending regime

# Strategy weights by regime (will be dynamically adjusted)
REGIME_STRATEGY_WEIGHTS = {
    'trending': {
        'static_bh_3m': 0.35,
        'dyn_bh_1y_vol': 0.25,
        'risk_adj_mom': 0.25,
        'quality_mom': 0.15,
    },
    'volatile': {
        'static_bh_3m': 0.15,
        'dyn_bh_1y_vol': 0.35,
        'risk_adj_mom': 0.30,
        'quality_mom': 0.20,
    },
    'mean_reverting': {
        'static_bh_3m': 0.20,
        'dyn_bh_1y_vol': 0.30,
        'risk_adj_mom': 0.20,
        'quality_mom': 0.30,
    },
    'neutral': {
        'static_bh_3m': 0.25,
        'dyn_bh_1y_vol': 0.25,
        'risk_adj_mom': 0.25,
        'quality_mom': 0.25,
    },
}


class AdaptiveMetaEnsemble:
    """
    Adaptive Meta-Ensemble that dynamically weights strategies based on:
    1. Recent performance of each strategy
    2. Current market regime
    3. Consensus among strategies
    """
    
    def __init__(self):
        self.strategy_history = defaultdict(list)  # Track strategy picks over time
        self.performance_history = defaultdict(list)  # Track strategy returns
        self.current_regime = 'neutral'
        self.last_rebalance_date = None
        self.current_holdings = []
        
    def detect_market_regime(self, ticker_data_grouped: Dict[str, pd.DataFrame], 
                            current_date: datetime) -> str:
        """
        Detect current market regime based on SPY or market-wide metrics.
        
        Returns:
            str: 'trending', 'volatile', 'mean_reverting', or 'neutral'
        """
        # Try to use SPY as market proxy
        market_proxy = None
        for proxy_ticker in ['SPY', 'QQQ', 'IWM']:
            if proxy_ticker in ticker_data_grouped:
                market_proxy = ticker_data_grouped[proxy_ticker]
                break
        
        if market_proxy is None or len(market_proxy) < 30:
            # Fallback: use average of all tickers
            return 'neutral'
        
        try:
            # Get recent data (last 30 days)
            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(market_proxy.index, 'tz') and market_proxy.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(market_proxy.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(market_proxy.index.tz)
            
            lookback_start = current_date_tz - timedelta(days=30)
            recent_data = market_proxy[(market_proxy.index >= lookback_start) & 
                                       (market_proxy.index <= current_date_tz)]
            
            if len(recent_data) < 10:
                return 'neutral'
            
            close_prices = recent_data['Close'].dropna()
            if len(close_prices) < 10:
                return 'neutral'
            
            # Calculate metrics
            daily_returns = close_prices.pct_change().dropna()
            
            # 1. Volatility (annualized)
            volatility = daily_returns.std()
            
            # 2. Trend (monthly return)
            monthly_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
            
            # 3. Mean reversion indicator (autocorrelation of returns)
            if len(daily_returns) >= 5:
                autocorr = daily_returns.autocorr(lag=1)
            else:
                autocorr = 0
            
            # Determine regime
            if volatility > REGIME_VOLATILITY_THRESHOLD_HIGH:
                regime = 'volatile'
            elif abs(monthly_return) > REGIME_TREND_THRESHOLD:
                regime = 'trending'
            elif autocorr < -0.1:  # Negative autocorrelation suggests mean reversion
                regime = 'mean_reverting'
            else:
                regime = 'neutral'
            
            print(f"   📊 Market Regime: {regime} (vol={volatility:.3f}, trend={monthly_return:.2%}, autocorr={autocorr:.2f})")
            return regime
            
        except Exception as e:
            print(f"   ⚠️ Regime detection error: {e}")
            return 'neutral'
    
    def get_strategy_picks(self, strategy_name: str, all_tickers: List[str],
                          ticker_data_grouped: Dict[str, pd.DataFrame],
                          current_date: datetime, train_start_date: datetime = None,
                          top_n: int = 20) -> List[str]:
        """
        Get stock picks from a specific strategy.
        """
        try:
            if strategy_name == 'static_bh_3m':
                return select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, 
                                               period='3m', current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'dyn_bh_1y_vol':
                # Dynamic BH 1Y with volatility filter
                picks = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                                period='1y', current_date=current_date, top_n=top_n * 2)
                # Apply volatility filter
                filtered_picks = []
                for ticker in picks:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        if len(ticker_data) >= 20:
                            daily_returns = ticker_data['Close'].pct_change().dropna()
                            vol = daily_returns.std() * np.sqrt(252) * 100  # Annualized %
                            if vol <= 120:  # Max 120% annualized volatility
                                filtered_picks.append(ticker)
                return filtered_picks[:top_n]
            
            elif strategy_name == 'risk_adj_mom':
                return select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped,
                                                 current_date=current_date, 
                                                 train_start_date=train_start_date,
                                                 top_n=top_n)
            
            elif strategy_name == 'quality_mom':
                return select_quality_momentum_stocks(all_tickers, ticker_data_grouped,
                                                     current_date=current_date, top_n=top_n)
            
            else:
                print(f"   ⚠️ Unknown strategy: {strategy_name}")
                return []
                
        except Exception as e:
            print(f"   ⚠️ Strategy {strategy_name} error: {e}")
            return []
    
    def calculate_strategy_weights(self, regime: str, 
                                   recent_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate dynamic strategy weights based on regime and recent performance.
        
        Combines:
        1. Base weights from regime
        2. Performance-based adjustment (inverse volatility weighting)
        """
        # Get base weights for current regime
        base_weights = REGIME_STRATEGY_WEIGHTS.get(regime, REGIME_STRATEGY_WEIGHTS['neutral']).copy()
        
        # Adjust weights based on recent performance
        if recent_performance:
            # Calculate performance scores (higher is better)
            perf_scores = {}
            for strategy, perf in recent_performance.items():
                if strategy in base_weights:
                    # Use softmax-like transformation
                    perf_scores[strategy] = np.exp(perf * 10)  # Scale factor
            
            # Normalize performance scores
            total_score = sum(perf_scores.values())
            if total_score > 0:
                perf_weights = {s: score / total_score for s, score in perf_scores.items()}
                
                # Blend base weights with performance weights (50/50)
                for strategy in base_weights:
                    if strategy in perf_weights:
                        base_weights[strategy] = 0.5 * base_weights[strategy] + 0.5 * perf_weights[strategy]
        
        # Normalize final weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            base_weights = {s: w / total_weight for s, w in base_weights.items()}
        
        return base_weights
    
    def get_consensus_picks(self, strategy_picks: Dict[str, List[str]],
                           weights: Dict[str, float],
                           min_agreement: int = MIN_CONSENSUS_AGREEMENT) -> List[Tuple[str, float]]:
        """
        Get stocks that appear in multiple strategies, weighted by strategy weights.
        
        Returns:
            List of (ticker, weighted_score) tuples, sorted by score
        """
        # Count how many strategies picked each stock and calculate weighted score
        stock_scores = defaultdict(float)
        stock_counts = defaultdict(int)
        
        for strategy, picks in strategy_picks.items():
            weight = weights.get(strategy, 0.25)
            for rank, ticker in enumerate(picks):
                # Higher rank (earlier in list) gets higher score
                rank_score = 1.0 / (rank + 1)  # 1st place = 1.0, 2nd = 0.5, etc.
                stock_scores[ticker] += weight * rank_score
                stock_counts[ticker] += 1
        
        # Filter by minimum agreement
        consensus_picks = [
            (ticker, score) 
            for ticker, score in stock_scores.items()
            if stock_counts[ticker] >= min_agreement
        ]
        
        # Sort by weighted score (descending)
        consensus_picks.sort(key=lambda x: x[1], reverse=True)
        
        return consensus_picks
    
    def select_stocks(self, all_tickers: List[str], 
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """
        Main entry point: Select stocks using adaptive ensemble.
        
        Steps:
        1. Detect market regime
        2. Get picks from each strategy
        3. Calculate dynamic weights
        4. Apply consensus filter
        5. Return top N stocks
        """
        print(f"\n   🎯 Adaptive Meta-Ensemble Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        # 1. Detect market regime
        self.current_regime = self.detect_market_regime(ticker_data_grouped, current_date)
        
        # 2. Get picks from each strategy
        strategy_picks = {}
        for strategy in ENSEMBLE_STRATEGIES:
            print(f"   🔍 Getting picks from {strategy}...")
            picks = self.get_strategy_picks(
                strategy, all_tickers, ticker_data_grouped,
                current_date, train_start_date, top_n=top_n * 2
            )
            strategy_picks[strategy] = picks
            print(f"      → {len(picks)} picks")
        
        # ✅ SAFETY CHECK: If all strategies returned empty, likely due to stale data
        total_picks = sum(len(picks) for picks in strategy_picks.values())
        if total_picks == 0:
            from config import DATA_FRESHNESS_MAX_DAYS
            print(f"\n   ⚠️ WARNING: All strategies returned 0 picks!")
            print(f"   ⚠️ This likely means your data is stale (>{DATA_FRESHNESS_MAX_DAYS} days old)")
            print(f"   ⚠️ ACTION REQUIRED: Download fresh price data before trading")
            print(f"   ❌ TRADING ABORTED: No valid recommendations possible\n")
            return []
        
        # 3. Calculate dynamic weights
        # For now, use regime-based weights (performance tracking would need historical data)
        weights = self.calculate_strategy_weights(self.current_regime, {})
        
        print(f"   ⚖️ Strategy weights ({self.current_regime} regime):")
        for strategy, weight in weights.items():
            print(f"      {strategy}: {weight:.1%}")
        
        # 4. Apply consensus filter
        consensus_picks = self.get_consensus_picks(strategy_picks, weights)
        
        if not consensus_picks:
            print(f"   ⚠️ No consensus picks found, using top strategy picks")
            # Fallback: use picks from highest-weighted strategy
            best_strategy = max(weights.items(), key=lambda x: x[1])[0]
            return strategy_picks.get(best_strategy, [])[:top_n]
        
        # 5. Return top N stocks
        selected_tickers = [ticker for ticker, score in consensus_picks[:top_n]]
        
        print(f"   ✅ Selected {len(selected_tickers)} stocks via consensus:")
        for ticker, score in consensus_picks[:top_n]:
            # Count how many strategies picked this stock
            count = sum(1 for picks in strategy_picks.values() if ticker in picks)
            print(f"      {ticker}: score={score:.3f}, agreement={count}/{len(ENSEMBLE_STRATEGIES)}")
        
        return selected_tickers


# ============================================
# Module-level function for integration
# ============================================

# Global instance for state persistence across calls
_ensemble_instance = None

def get_ensemble_instance() -> AdaptiveMetaEnsemble:
    """Get or create the global ensemble instance."""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = AdaptiveMetaEnsemble()
    return _ensemble_instance


def select_adaptive_ensemble_stocks(all_tickers: List[str], 
                                    ticker_data_grouped: Dict[str, pd.DataFrame],
                                    current_date: datetime = None,
                                    train_start_date: datetime = None,
                                    top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Adaptive Meta-Ensemble stock selection strategy.
    
    This strategy dynamically combines multiple strategies based on:
    1. Current market regime (trending, volatile, mean-reverting)
    2. Recent strategy performance
    3. Consensus among strategies (only trade when multiple agree)
    
    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data (with date as index)
        current_date: Current date for analysis
        train_start_date: Start date for training (used by some sub-strategies)
        top_n: Number of stocks to select
        
    Returns:
        List[str]: Selected ticker symbols
    """
    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers 
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            print("   ❌ No data available for adaptive ensemble")
            return []
    
    # Get ensemble instance and select stocks
    ensemble = get_ensemble_instance()
    return ensemble.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_ensemble_state():
    """Reset the global ensemble instance (useful for backtesting)."""
    global _ensemble_instance
    _ensemble_instance = None
