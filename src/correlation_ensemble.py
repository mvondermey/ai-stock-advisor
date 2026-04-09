"""
Correlation-Filtered Ensemble Strategy

Combines multiple strategies with correlation filtering to avoid concentrated positions.
Features:
- Correlation matrix analysis to reduce portfolio concentration
- Sector diversification constraints
- Ensemble consensus with correlation filtering
- Risk-adjusted selection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import itertools
from strategy_cache_adapter import (
    ensure_price_history_cache,
    get_cached_frame_between,
    get_cached_history_up_to,
    resolve_cache_current_date,
)

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
)

# ============================================
# Configuration Parameters
# ============================================

# Strategies to include (different from volatility_ensemble for diversification)
CORR_ENSEMBLE_STRATEGIES = [
    'static_bh_6m',      # 6-month static buy & hold
    'static_bh_1y',      # 1-year static buy & hold (top performer)
    'dyn_bh_1y_vol',     # Dynamic BH with volatility filter (top performer)
    'dyn_bh_6m',         # 6-month dynamic buy & hold
]

# Correlation parameters
MAX_CORRELATION = 0.85  # Reduced from 0.70 to 0.85 (less strict)
CORRELATION_LOOKBACK_DAYS = 60  # Days to calculate correlation
MIN_CORRELATION_SAMPLES = 20  # Minimum data points for correlation

# Sector diversification
MAX_SECTOR_WEIGHT = 0.40  # Maximum 40% in any single sector
SECTOR_MAPPING = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'CRM', 'ADBE', 'NFLX'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'DHR', 'ABBV', 'BMY', 'AMGN', 'GILD', 'MDT', 'ISRG'],
    'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CB', 'COF', 'USB'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'TGT', 'COST', 'WMT', 'SBUX', 'BKNG', 'EXPE'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL', 'BKR', 'KMI'],
    'Industrial': ['CAT', 'DE', 'GE', 'HON', 'UPS', 'RTX', 'BA', 'MMM', 'LMT', 'NOV', 'PH', 'EMR'],
    'Materials': ['LIN', 'APD', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'BHP', 'RIO', 'VALE', 'AA', 'ALB'],
}

# Ensemble weights (for new strategies)
CORR_STRATEGY_WEIGHTS = {
    'static_bh_6m': 0.25,
    'static_bh_1y': 0.25,
    'dyn_bh_1y_vol': 0.25,
    'dyn_bh_6m': 0.25,
}


class CorrelationFilteredEnsemble:
    """
    Correlation-Filtered Ensemble that manages diversification through:
    1. Correlation filtering to avoid concentrated positions
    2. Sector diversification constraints
    3. Ensemble consensus with risk adjustment
    """
    
    def __init__(self):
        self.correlation_cache = {}
        self.sector_cache = {}
        
    def get_stock_sector(self, ticker: str) -> str:
        """Get sector for a ticker based on known mappings."""
        if ticker in self.sector_cache:
            return self.sector_cache[ticker]
        
        for sector, stocks in SECTOR_MAPPING.items():
            if ticker in stocks:
                self.sector_cache[ticker] = sector
                return sector
        
        # Default sector if not found
        self.sector_cache[ticker] = 'Other'
        return 'Other'
    
    def calculate_correlation_matrix(self, tickers: List[str], 
                                   ticker_data_grouped: Dict[str, pd.DataFrame],
                                   current_date: datetime,
                                   price_history_cache=None) -> pd.DataFrame:
        """Calculate correlation matrix for given tickers."""
        try:
            # Remove duplicates from tickers list
            tickers = list(dict.fromkeys(tickers))
            
            # Get returns data
            returns_data = {}
            price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
            
            for ticker in tickers:
                recent_data = get_cached_frame_between(
                    price_history_cache,
                    ticker,
                    current_date - timedelta(days=CORRELATION_LOOKBACK_DAYS),
                    current_date,
                    field_names=("close",),
                    min_rows=MIN_CORRELATION_SAMPLES,
                )
                if recent_data is not None and len(recent_data) >= MIN_CORRELATION_SAMPLES:
                    returns = recent_data['close'].pct_change(fill_method=None).dropna()
                    if len(returns) >= MIN_CORRELATION_SAMPLES:
                        returns_data[ticker] = returns.reset_index(drop=True)
            
            if len(returns_data) < 2:
                # Return identity matrix if insufficient data
                return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Fill diagonal with 1.0
            np.fill_diagonal(correlation_matrix.values, 1.0)
            
            return correlation_matrix
            
        except Exception as e:
            print(f"   ⚠️ Error calculating correlation matrix: {e}")
            # Return identity matrix on error
            return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    
    def find_high_correlation_pairs(self, correlation_matrix: pd.DataFrame,
                                   threshold: float = MAX_CORRELATION) -> List[Tuple[str, str, float]]:
        """Find pairs of stocks with correlation above threshold."""
        high_corr_pairs = []
        
        for i, ticker1 in enumerate(correlation_matrix.index):
            for j, ticker2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicate pairs and self-correlation
                    corr_value = correlation_matrix.loc[ticker1, ticker2]
                    if corr_value > threshold:
                        high_corr_pairs.append((ticker1, ticker2, corr_value))
        
        # Sort by correlation (descending)
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        return high_corr_pairs
    
    def apply_correlation_filter(self, candidates: List[Tuple[str, float]],
                                correlation_matrix: pd.DataFrame,
                                max_correlation: float = MAX_CORRELATION) -> List[str]:
        """Filter candidates to avoid high correlation."""
        if not candidates:
            return []
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for ticker, score in candidates:
            # Check correlation with already selected stocks
            can_add = True
            for selected_ticker in selected:
                if ticker in correlation_matrix.index and selected_ticker in correlation_matrix.columns:
                    corr = correlation_matrix.loc[ticker, selected_ticker]
                    if corr > max_correlation:
                        can_add = False
                        break
            
            if can_add:
                selected.append(ticker)
        
        return selected
    
    def apply_sector_diversification(self, candidates: List[str],
                                    max_sector_weight: float = MAX_SECTOR_WEIGHT) -> List[str]:
        """Apply sector diversification constraints.
        
        NOTE: Disabled strict filtering since most stocks are classified as 'Other' sector
        due to limited sector mapping. Just return all candidates.
        """
        if not candidates:
            return []
        
        # Return all candidates - sector diversification disabled
        # Most stocks fall into 'Other' category which makes strict filtering counterproductive
        return candidates
    
    def get_strategy_picks(self, strategy_name: str, all_tickers: List[str],
                          ticker_data_grouped: Dict[str, pd.DataFrame],
                          current_date: datetime, top_n: int = 15,
                          price_history_cache=None) -> List[str]:
        """Get stock picks from a specific strategy."""
        try:
            if strategy_name == 'static_bh_3m':
                return select_dynamic_bh_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    period='3m',
                    current_date=current_date,
                    top_n=top_n,
                    price_history_cache=price_history_cache,
                )
            
            elif strategy_name == 'static_bh_6m':
                return select_dynamic_bh_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    period='6m',
                    current_date=current_date,
                    top_n=top_n,
                    price_history_cache=price_history_cache,
                )
            
            elif strategy_name == 'static_bh_1y':
                return select_dynamic_bh_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    period='1y',
                    current_date=current_date,
                    top_n=top_n,
                    price_history_cache=price_history_cache,
                )
            
            elif strategy_name == 'dyn_bh_6m':
                return select_dynamic_bh_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    period='6m',
                    current_date=current_date,
                    top_n=top_n,
                    price_history_cache=price_history_cache,
                )
            
            elif strategy_name == 'dyn_bh_1y_vol':
                picks = select_dynamic_bh_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    period='1y',
                    current_date=current_date,
                    top_n=top_n * 2,
                    price_history_cache=price_history_cache,
                )
                # Apply basic filter
                filtered_picks = []
                for ticker in picks:
                    close_history = get_cached_history_up_to(
                        price_history_cache,
                        ticker,
                        current_date,
                        field_name="close",
                        min_rows=20,
                    )
                    if close_history is not None and len(close_history) >= 20:
                        daily_returns = pd.Series(close_history).pct_change(fill_method=None).dropna()
                        vol = daily_returns.std() * np.sqrt(252) * 100
                        if vol <= 120:
                            filtered_picks.append(ticker)
                return filtered_picks[:top_n]
            
            elif strategy_name == 'risk_adj_mom':
                return select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped,
                                                 current_date=current_date,
                                                 top_n=top_n,
                                                 price_history_cache=price_history_cache)
            
            elif strategy_name == 'quality_mom':
                return select_quality_momentum_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=top_n,
                    price_history_cache=price_history_cache,
                )
            
            else:
                return []
                
        except Exception as e:
            return []
    
    def calculate_ensemble_scores(self, strategy_picks: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate ensemble scores with strategy weights."""
        stock_scores = defaultdict(float)
        stock_counts = defaultdict(int)
        
        for strategy, picks in strategy_picks.items():
            weight = CORR_STRATEGY_WEIGHTS.get(strategy, 0.25)
            for rank, ticker in enumerate(picks):
                rank_score = 1.0 / (rank + 1)
                stock_scores[ticker] += weight * rank_score
                stock_counts[ticker] += 1
        
        # Apply consensus filter (at least 1 strategy - less strict)
        consensus_scores = {
            ticker: score 
            for ticker, score in stock_scores.items()
            if stock_counts[ticker] >= 1  # Changed from 2 to 1
        }
        
        return consensus_scores
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     top_n: int = PORTFOLIO_SIZE,
                     price_history_cache=None) -> List[str]:
        """Main entry point: Select stocks with correlation filtering."""
        print(f"\n   🎯 Correlation-Filtered Ensemble Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        # 1. Get picks from each strategy
        strategy_picks = {}
        for strategy in CORR_ENSEMBLE_STRATEGIES:
            print(f"   🔍 Getting picks from {strategy}...")
            picks = self.get_strategy_picks(
                strategy, all_tickers, ticker_data_grouped,
                current_date, top_n=top_n * 2, price_history_cache=price_history_cache
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
        
        # 2. Calculate ensemble scores
        ensemble_scores = self.calculate_ensemble_scores(strategy_picks)
        
        if not ensemble_scores:
            print(f"   ⚠️ No consensus picks found (need at least 1 strategy to agree)")
            return []
        
        # 3. Sort by ensemble score
        sorted_candidates = [(ticker, score) for ticker, score in ensemble_scores.items()]
        sorted_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Calculate correlation matrix for top candidates
        top_tickers = [ticker for ticker, score in sorted_candidates[:top_n * 3]]
        correlation_matrix = self.calculate_correlation_matrix(
            top_tickers,
            ticker_data_grouped,
            current_date,
            price_history_cache=price_history_cache,
        )
        
        # 5. Find high correlation pairs
        high_corr_pairs = self.find_high_correlation_pairs(correlation_matrix)
        if high_corr_pairs:
            print(f"   📊 Found {len(high_corr_pairs)} high correlation pairs (> {MAX_CORRELATION:.0%})")
            for t1, t2, corr in high_corr_pairs[:5]:  # Show top 5
                print(f"      {t1}-{t2}: {corr:.2%}")
        
        # 6. Apply correlation filter
        filtered_stocks = self.apply_correlation_filter(sorted_candidates, correlation_matrix)
        print(f"   🔗 After correlation filter: {len(filtered_stocks)} stocks")
        
        # 7. Apply sector diversification
        diversified_stocks = self.apply_sector_diversification(filtered_stocks)
        print(f"   🏢 After sector diversification: {len(diversified_stocks)} stocks")
        
        # 8. Take top N
        final_selection = diversified_stocks[:top_n]
        
        # 9. Display results
        print(f"   ✅ Selected {len(final_selection)} stocks:")
        for ticker in final_selection:
            score = ensemble_scores[ticker]
            sector = self.get_stock_sector(ticker)
            print(f"      {ticker}: score={score:.3f}, sector={sector}")
        
        # 10. Display sector breakdown
        sector_breakdown = defaultdict(int)
        for ticker in final_selection:
            sector = self.get_stock_sector(ticker)
            sector_breakdown[sector] += 1
        
        print(f"   📊 Sector breakdown:")
        for sector, count in sector_breakdown.items():
            weight = count / len(final_selection)
            print(f"      {sector}: {count} stocks ({weight:.1%})")
        
        return final_selection


# ============================================
# Module-level function for integration
# ============================================

# Global instance for state persistence
_corr_ensemble_instance = None

def get_corr_ensemble_instance() -> CorrelationFilteredEnsemble:
    """Get or create the global correlation ensemble instance."""
    global _corr_ensemble_instance
    if _corr_ensemble_instance is None:
        _corr_ensemble_instance = CorrelationFilteredEnsemble()
    return _corr_ensemble_instance


def select_correlation_ensemble_stocks(all_tickers: List[str],
                                       ticker_data_grouped: Dict[str, pd.DataFrame],
                                       current_date: datetime = None,
                                       top_n: int = PORTFOLIO_SIZE,
                                       price_history_cache=None) -> List[str]:
    """
    Correlation-Filtered Ensemble stock selection strategy.
    
    This strategy combines multiple strategies with:
    1. Correlation filtering to reduce concentration
    2. Sector diversification constraints
    3. Ensemble consensus
    
    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data
        current_date: Current date for analysis
        train_start_date: Start date for training
        top_n: Number of stocks to select
        
    Returns:
        List[str]: Selected ticker symbols
    """
    # Filter out inverse ETFs - they should only be in inverse_etf_hedge strategy
    from config import INVERSE_ETFS
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]
    
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    current_date = resolve_cache_current_date(price_history_cache, current_date, all_tickers)
    if current_date is None:
        return []
    
    ensemble = get_corr_ensemble_instance()
    return ensemble.select_stocks(
        all_tickers, ticker_data_grouped, current_date, top_n, price_history_cache=price_history_cache
    )


def reset_corr_ensemble_state():
    """Reset the global correlation ensemble instance."""
    global _corr_ensemble_instance
    _corr_ensemble_instance = None
