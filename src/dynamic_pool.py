"""
Dynamic Strategy Pool Strategy

Rotates strategies in/out based on recent performance.
Features:
- Tracks rolling performance of all available strategies
- Keeps top N performing strategies in the pool
- Dynamic weight allocation based on performance
- Performance-based strategy rotation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Import config
from config import (
    TRANSACTION_COST,
    PORTFOLIO_SIZE,
)

# Import existing strategies
from shared_strategies import (
    select_dynamic_bh_stocks,
    select_risk_adj_mom_stocks,
    select_quality_momentum_stocks,
    select_volatility_adj_mom_stocks,
    select_mean_reversion_stocks,
    select_3m_1y_ratio_stocks,
    select_1y_3m_ratio_stocks,
    select_turnaround_stocks,
)

# ============================================
# Configuration Parameters
# ============================================

# All available strategies
ALL_AVAILABLE_STRATEGIES = [
    'static_bh_3m',
    'static_bh_1y',
    'dyn_bh_1y_vol',
    'dyn_bh_3m',
    'dyn_bh_1m',
    'risk_adj_mom',
    'quality_mom',
    'vol_adj_mom',
    'mean_reversion',
    '3m_1y_ratio',
    '1y_3m_ratio',
    'turnaround',
]

# Pool parameters
POOL_SIZE = 4  # Number of strategies to keep in active pool
PERFORMANCE_WINDOW_DAYS = 30  # Days to evaluate strategy performance
MIN_PERFORMANCE_SAMPLES = 10  # Minimum samples for performance evaluation

# Performance tracking
PERFORMANCE_UPDATE_FREQUENCY = 5  # Days between performance updates
STRATEGY_ROTATION_THRESHOLD = 0.15  # 15% performance difference triggers rotation

# Weighting parameters
MIN_STRATEGY_WEIGHT = 0.10  # 10% minimum weight per strategy
MAX_STRATEGY_WEIGHT = 0.40  # 40% maximum weight per strategy
PERFORMANCE_SCALING_FACTOR = 5.0  # How aggressively to weight by performance


class DynamicStrategyPool:
    """
    Dynamic Strategy Pool that:
    1. Tracks performance of all available strategies
    2. Keeps top N strategies in active pool
    3. Allocates weights based on recent performance
    4. Rotates strategies based on performance
    """
    
    def __init__(self):
        self.active_strategies = ['static_bh_3m', 'dyn_bh_1y_vol', 'risk_adj_mom', 'quality_mom']
        self.strategy_performance = defaultdict(list)  # strategy -> list of recent returns
        self.strategy_weights = {s: 0.25 for s in self.active_strategies}
        self.last_performance_update = None
        self.performance_history = defaultdict(deque)  # strategy -> rolling window
        
    def calculate_strategy_performance(self, strategy_name: str, all_tickers: List[str],
                                     ticker_data_grouped: Dict[str, pd.DataFrame],
                                     current_date: datetime, train_start_date: datetime = None) -> float:
        """Calculate recent performance for a strategy."""
        try:
            # Get strategy picks
            picks = self.get_strategy_picks(
                strategy_name, all_tickers, ticker_data_grouped,
                current_date, train_start_date, top_n=10
            )
            
            if not picks:
                return 0.0
            
            # Calculate equal-weight return over performance window
            returns = []
            lookback_start = current_date - timedelta(days=PERFORMANCE_WINDOW_DAYS)
            
            for ticker in picks[:5]:  # Top 5 picks for performance evaluation
                if ticker in ticker_data_grouped:
                    ticker_data = ticker_data_grouped[ticker]
                    window_data = ticker_data[(ticker_data.index >= lookback_start) & 
                                             (ticker_data.index <= current_date)]
                    
                    if len(window_data) >= 10:
                        start_price = window_data['Close'].iloc[0]
                        end_price = window_data['Close'].iloc[-1]
                        if start_price > 0:
                            stock_return = (end_price / start_price) - 1
                            # Subtract transaction costs (2 trades)
                            stock_return -= 2 * TRANSACTION_COST
                            returns.append(stock_return)
            
            if returns:
                return np.mean(returns)
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def update_strategy_performance(self, all_tickers: List[str],
                                   ticker_data_grouped: Dict[str, pd.DataFrame],
                                   current_date: datetime, train_start_date: datetime = None):
        """Update performance tracking for all strategies."""
        print(f"   📊 Updating strategy performance...")
        
        # Calculate performance for all available strategies
        current_performance = {}
        for strategy in ALL_AVAILABLE_STRATEGIES:
            perf = self.calculate_strategy_performance(
                strategy, all_tickers, ticker_data_grouped,
                current_date, train_start_date
            )
            current_performance[strategy] = perf
            
            # Add to rolling history
            self.performance_history[strategy].append(perf)
            if len(self.performance_history[strategy]) > PERFORMANCE_WINDOW_DAYS:
                self.performance_history[strategy].popleft()
        
        # Display current performance
        sorted_performance = sorted(current_performance.items(), key=lambda x: x[1], reverse=True)
        print(f"   📈 Strategy performance (last {PERFORMANCE_WINDOW_DAYS} days):")
        for strategy, perf in sorted_performance[:8]:  # Top 8
            print(f"      {strategy}: {perf:.2%}")
        
        # Check if rotation is needed
        self.check_strategy_rotation(current_performance)
        
        # Update weights based on performance
        self.update_strategy_weights(current_performance)
        
        self.last_performance_update = current_date
    
    def check_strategy_rotation(self, current_performance: Dict[str, float]):
        """Check if strategies should be rotated based on performance."""
        # Get current active strategy performance
        active_performance = {s: current_performance.get(s, 0) for s in self.active_strategies}
        
        # Get top performing strategies
        all_performance = [(s, p) for s, p in current_performance.items() 
                          if len(self.performance_history[s]) >= MIN_PERFORMANCE_SAMPLES]
        all_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Check if any inactive strategy is significantly better
        top_inactive = [s for s, p in all_performance if s not in self.active_strategies]
        worst_active = min(active_performance.items(), key=lambda x: x[1]) if active_performance else (None, 0)
        
        should_rotate = False
        if top_inactive and worst_active[0]:
            best_inactive_perf = current_performance[top_inactive[0]]
            worst_active_perf = worst_active[1]
            
            if best_inactive_perf > worst_active_perf + STRATEGY_ROTATION_THRESHOLD:
                should_rotate = True
                print(f"   🔄 Rotation triggered: {top_inactive[0]} ({best_inactive_perf:.2%}) > {worst_active[0]} ({worst_active_perf:.2%})")
        
        if should_rotate:
            # Replace worst active strategy with best inactive
            if worst_active[0] and top_inactive:
                self.active_strategies.remove(worst_active[0])
                self.active_strategies.append(top_inactive[0])
                print(f"   ✅ Rotated: {worst_active[0]} → {top_inactive[0]}")
    
    def update_strategy_weights(self, current_performance: Dict[str, float]):
        """Update strategy weights based on recent performance."""
        # Get performance for active strategies
        active_performance = {s: current_performance.get(s, 0) for s in self.active_strategies}
        
        # Apply softmax transformation for weights
        exp_scores = {}
        for strategy, perf in active_performance.items():
            # Scale and apply softmax
            exp_score = np.exp(perf * PERFORMANCE_SCALING_FACTOR)
            exp_scores[strategy] = exp_score
        
        # Normalize weights
        total_score = sum(exp_scores.values())
        if total_score > 0:
            raw_weights = {s: score / total_score for s, score in exp_scores.items()}
        else:
            raw_weights = {s: 1.0 / len(self.active_strategies) for s in self.active_strategies}
        
        # Apply weight constraints
        constrained_weights = {}
        for strategy in self.active_strategies:
            weight = raw_weights.get(strategy, 0.25)
            weight = max(MIN_STRATEGY_WEIGHT, min(MAX_STRATEGY_WEIGHT, weight))
            constrained_weights[strategy] = weight
        
        # Renormalize to sum to 1
        total_weight = sum(constrained_weights.values())
        self.strategy_weights = {s: w / total_weight for s, w in constrained_weights.items()}
        
        # Display weights
        print(f"   ⚖️ Strategy weights:")
        for strategy, weight in sorted(self.strategy_weights.items(), key=lambda x: x[1], reverse=True):
            perf = current_performance.get(strategy, 0)
            print(f"      {strategy}: {weight:.1%} (perf: {perf:.2%})")
    
    def get_strategy_picks(self, strategy_name: str, all_tickers: List[str],
                          ticker_data_grouped: Dict[str, pd.DataFrame],
                          current_date: datetime, train_start_date: datetime = None,
                          top_n: int = 10) -> List[str]:
        """Get stock picks from a specific strategy."""
        try:
            if strategy_name == 'static_bh_3m':
                return select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                               period='3m', current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'static_bh_1y':
                return select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                               period='1y', current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'dyn_bh_1y_vol':
                picks = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                                period='1y', current_date=current_date, top_n=top_n * 2)
                filtered_picks = []
                for ticker in picks:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        if len(ticker_data) >= 20:
                            daily_returns = ticker_data['Close'].pct_change().dropna()
                            vol = daily_returns.std() * np.sqrt(252) * 100
                            if vol <= 120:
                                filtered_picks.append(ticker)
                return filtered_picks[:top_n]
            
            elif strategy_name == 'dyn_bh_3m':
                return select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                               period='3m', current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'dyn_bh_1m':
                return select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                               period='1m', current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'risk_adj_mom':
                return select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped,
                                                 current_date=current_date,
                                                 train_start_date=train_start_date,
                                                 top_n=top_n)
            
            elif strategy_name == 'quality_mom':
                return select_quality_momentum_stocks(all_tickers, ticker_data_grouped,
                                                     current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'vol_adj_mom':
                return select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped,
                                                      current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'mean_reversion':
                return select_mean_reversion_stocks(all_tickers, ticker_data_grouped,
                                                   current_date=current_date, top_n=top_n)
            
            elif strategy_name == '3m_1y_ratio':
                return select_3m_1y_ratio_stocks(all_tickers, ticker_data_grouped,
                                                current_date=current_date, top_n=top_n)
            
            elif strategy_name == '1y_3m_ratio':
                return select_1y_3m_ratio_stocks(all_tickers, ticker_data_grouped,
                                                current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'turnaround':
                return select_turnaround_stocks(all_tickers, ticker_data_grouped,
                                              current_date=current_date, top_n=top_n)
            
            else:
                return []
                
        except Exception as e:
            return []
    
    def calculate_ensemble_scores(self, strategy_picks: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate ensemble scores with dynamic weights."""
        stock_scores = defaultdict(float)
        stock_counts = defaultdict(int)
        
        for strategy, picks in strategy_picks.items():
            weight = self.strategy_weights.get(strategy, 0.25)
            for rank, ticker in enumerate(picks):
                rank_score = 1.0 / (rank + 1)
                stock_scores[ticker] += weight * rank_score
                stock_counts[ticker] += 1
        
        # Apply consensus filter (at least 2 strategies)
        consensus_scores = {
            ticker: score 
            for ticker, score in stock_scores.items()
            if stock_counts[ticker] >= 2
        }
        
        return consensus_scores
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Main entry point: Select stocks using dynamic strategy pool."""
        print(f"\n   🎯 Dynamic Strategy Pool Strategy")
        print(f"   📅 Date: {current_date.date()}")
        print(f"   🏊 Active strategies: {', '.join(self.active_strategies)}")
        
        # Update performance if needed
        if (self.last_performance_update is None or 
            (current_date - self.last_performance_update).days >= PERFORMANCE_UPDATE_FREQUENCY):
            self.update_strategy_performance(all_tickers, ticker_data_grouped, current_date, train_start_date)
        
        # 1. Get picks from active strategies
        strategy_picks = {}
        for strategy in self.active_strategies:
            print(f"   🔍 Getting picks from {strategy}...")
            picks = self.get_strategy_picks(
                strategy, all_tickers, ticker_data_grouped,
                current_date, train_start_date, top_n=top_n * 2
            )
            strategy_picks[strategy] = picks
            print(f"      → {len(picks)} picks (weight: {self.strategy_weights.get(strategy, 0.25):.1%})")
        
        # 2. Calculate ensemble scores
        ensemble_scores = self.calculate_ensemble_scores(strategy_picks)
        
        if not ensemble_scores:
            print(f"   ⚠️ No consensus picks found")
            return []
        
        # 3. Sort by ensemble score
        sorted_candidates = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        final_selection = [ticker for ticker, score in sorted_candidates[:top_n]]
        
        # 4. Display results
        print(f"   ✅ Selected {len(final_selection)} stocks:")
        for ticker, score in sorted_candidates[:top_n]:
            # Count how many strategies picked this stock
            count = sum(1 for picks in strategy_picks.values() if ticker in picks)
            print(f"      {ticker}: score={score:.3f}, agreement={count}/{len(self.active_strategies)}")
        
        return final_selection


# ============================================
# Module-level function for integration
# ============================================

# Global instance for state persistence
_dynamic_pool_instance = None

def get_dynamic_pool_instance() -> DynamicStrategyPool:
    """Get or create the global dynamic pool instance."""
    global _dynamic_pool_instance
    if _dynamic_pool_instance is None:
        _dynamic_pool_instance = DynamicStrategyPool()
    return _dynamic_pool_instance


def select_dynamic_pool_stocks(all_tickers: List[str],
                                ticker_data_grouped: Dict[str, pd.DataFrame],
                                current_date: datetime = None,
                                train_start_date: datetime = None,
                                top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Dynamic Strategy Pool stock selection strategy.
    
    This strategy:
    1. Tracks performance of all available strategies
    2. Keeps top N strategies in active pool
    3. Allocates weights based on recent performance
    4. Rotates strategies based on performance
    
    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data
        current_date: Current date for analysis
        train_start_date: Start date for training
        top_n: Number of stocks to select
        
    Returns:
        List[str]: Selected ticker symbols
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    pool = get_dynamic_pool_instance()
    return pool.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_dynamic_pool_state():
    """Reset the global dynamic pool instance."""
    global _dynamic_pool_instance
    _dynamic_pool_instance = None
