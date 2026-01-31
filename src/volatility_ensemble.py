"""
Volatility-Adjusted Ensemble Strategy

Combines multiple strategies with volatility-based position sizing.
Features:
- Inverse volatility weighting for position sizes
- Dynamic volatility caps based on market regime
- Ensemble of top performing strategies
- Risk-adjusted rebalancing
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

# Strategies to include
VOL_ENSEMBLE_STRATEGIES = [
    'static_bh_3m',
    'dyn_bh_1y_vol', 
    'risk_adj_mom',
    'quality_mom',
]

# Volatility parameters
MAX_PORTFOLIO_VOLATILITY = 0.20  # 20% annualized max portfolio volatility
MAX_SINGLE_STOCK_VOLATILITY = 0.40  # 40% annualized max for any single stock
VOLATILITY_LOOKBACK_DAYS = 30  # Days to calculate volatility

# Position sizing
MIN_POSITION_WEIGHT = 0.05  # 5% minimum position weight


MAX_POSITION_WEIGHT = 0.30  # 30% maximum position weight

# Rebalancing thresholds
VOLATILITY_REBALANCE_THRESHOLD = 0.05  # 5% change in volatility triggers rebalance


class VolatilityAdjustedEnsemble:
    """
    Volatility-Adjusted Ensemble that manages risk through:
    1. Inverse volatility position sizing
    2. Dynamic volatility caps
    3. Ensemble consensus filtering
    """
    
    def __init__(self):
        self.strategy_weights = {
            'static_bh_3m': 0.25,
            'dyn_bh_1y_vol': 0.25,
            'risk_adj_mom': 0.25,
            'quality_mom': 0.25,
        }
        self.last_volatility_check = None
        self.current_portfolio_volatility = 0.0
        
    def calculate_stock_volatility(self, ticker: str, ticker_data_grouped: Dict[str, pd.DataFrame],
                                  current_date: datetime) -> float:
        """Calculate annualized volatility for a stock."""
        try:
            if ticker not in ticker_data_grouped:
                return 0.5  # Default high volatility if no data
            
            ticker_data = ticker_data_grouped[ticker]
            lookback_start = current_date - timedelta(days=VOLATILITY_LOOKBACK_DAYS)
            recent_data = ticker_data[(ticker_data.index >= lookback_start) & 
                                      (ticker_data.index <= current_date)]
            
            if len(recent_data) < 10:
                return 0.5  # Default high volatility
            
            # Calculate daily returns
            daily_returns = recent_data['Close'].pct_change().dropna()
            if len(daily_returns) < 5:
                return 0.5
            
            # Annualized volatility
            volatility = daily_returns.std() * np.sqrt(252)
            return min(volatility, 1.0)  # Cap at 100%
            
        except Exception as e:
            return 0.5  # Default high volatility on error
    
    def calculate_inverse_volatility_weights(self, tickers: List[str], 
                                            ticker_data_grouped: Dict[str, pd.DataFrame],
                                            current_date: datetime) -> Dict[str, float]:
        """Calculate inverse volatility weights for position sizing."""
        volatilities = {}
        
        # Calculate volatilities
        for ticker in tickers:
            vol = self.calculate_stock_volatility(ticker, ticker_data_grouped, current_date)
            # Apply maximum volatility cap
            vol = min(vol, MAX_SINGLE_STOCK_VOLATILITY)
            volatilities[ticker] = max(vol, 0.05)  # Minimum 5% volatility
        
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        inv_vol_sum = sum(1.0 / vol for vol in volatilities.values())
        
        for ticker, vol in volatilities.items():
            inv_vol_weights[ticker] = (1.0 / vol) / inv_vol_sum
        
        # Apply position size constraints
        for ticker in inv_vol_weights:
            inv_vol_weights[ticker] = max(MIN_POSITION_WEIGHT, 
                                         min(MAX_POSITION_WEIGHT, inv_vol_weights[ticker]))
        
        # Renormalize to sum to 1
        total_weight = sum(inv_vol_weights.values())
        inv_vol_weights = {t: w / total_weight for t, w in inv_vol_weights.items()}
        
        return inv_vol_weights
    
    def get_strategy_picks(self, strategy_name: str, all_tickers: List[str],
                          ticker_data_grouped: Dict[str, pd.DataFrame],
                          current_date: datetime, train_start_date: datetime = None,
                          top_n: int = 15) -> List[str]:
        """Get stock picks from a specific strategy."""
        try:
            if strategy_name == 'static_bh_3m':
                return select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                               period='3m', current_date=current_date, top_n=top_n)
            
            elif strategy_name == 'dyn_bh_1y_vol':
                picks = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                                period='1y', current_date=current_date, top_n=top_n)
                # Apply volatility filter
                filtered_picks = []
                for ticker in picks:
                    if ticker in ticker_data_grouped:
                        vol = self.calculate_stock_volatility(ticker, ticker_data_grouped, current_date)
                        if vol <= 0.60:  # 60% annualized volatility max
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
                return []
                
        except Exception as e:
            return []
    
    def calculate_ensemble_scores(self, strategy_picks: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate ensemble scores with strategy weights."""
        stock_scores = defaultdict(float)
        stock_counts = defaultdict(int)
        
        for strategy, picks in strategy_picks.items():
            weight = self.strategy_weights.get(strategy, 0.25)
            for rank, ticker in enumerate(picks):
                # Higher rank gets higher score
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
    
    def calculate_portfolio_volatility(self, tickers: List[str], weights: Dict[str, float],
                                       ticker_data_grouped: Dict[str, pd.DataFrame],
                                       current_date: datetime, debug: bool = False) -> float:
        """Calculate portfolio-level volatility."""
        try:
            # Get returns data for all stocks
            returns_data = {}
            lookback_start = current_date - timedelta(days=VOLATILITY_LOOKBACK_DAYS)
            
            # Debug: Check date types on first call
            if debug and tickers:
                first_ticker = tickers[0]
                if first_ticker in ticker_data_grouped:
                    idx = ticker_data_grouped[first_ticker].index
                    print(f"      🔍 DEBUG dates: current_date={current_date} (type={type(current_date).__name__})")
                    print(f"      🔍 DEBUG dates: lookback_start={lookback_start}")
                    print(f"      🔍 DEBUG dates: data index range={idx.min()} to {idx.max()} (type={type(idx[0]).__name__ if len(idx) > 0 else 'empty'})")
            
            missing_data_tickers = []
            insufficient_data_tickers = []
            
            for ticker in tickers:
                if ticker in ticker_data_grouped:
                    ticker_data = ticker_data_grouped[ticker]
                    recent_data = ticker_data[(ticker_data.index >= lookback_start) & 
                                              (ticker_data.index <= current_date)]
                    
                    if debug:
                        print(f"      🔍 DEBUG slice: {ticker} lookback={lookback_start}, current={current_date}, found {len(recent_data)} rows")
                        if len(recent_data) > 0:
                            print(f"      🔍 DEBUG {ticker} sample dates: {recent_data.index[0]} to {recent_data.index[-1]}")
                    
                    if len(recent_data) >= 10:
                        returns = recent_data['Close'].pct_change().dropna()
                        if len(returns) >= 5:
                            returns_data[ticker] = returns
                        else:
                            insufficient_data_tickers.append(f"{ticker}(returns={len(returns)})")
                    else:
                        insufficient_data_tickers.append(f"{ticker}(data={len(recent_data)})")
                else:
                    missing_data_tickers.append(ticker)
            
            if debug and (missing_data_tickers or insufficient_data_tickers):
                print(f"      🔍 DEBUG portfolio_vol: missing={missing_data_tickers}, insufficient={insufficient_data_tickers}")
            
            if len(returns_data) < 2:
                if debug:
                    print(f"      🔍 DEBUG portfolio_vol: Only {len(returns_data)} tickers with data, returning default 0.20")
                return 0.20  # Default 20% volatility
            
            # Create returns DataFrame - align on common dates
            returns_df = pd.DataFrame(returns_data)
            
            # Drop rows with any NaN (only keep dates where ALL tickers have data)
            returns_df_aligned = returns_df.dropna()
            
            if debug:
                print(f"      🔍 DEBUG portfolio_vol: returns_data has {len(returns_data)} tickers")
                print(f"      🔍 DEBUG portfolio_vol: returns_df shape={returns_df.shape}, aligned shape={returns_df_aligned.shape}")
                if len(returns_df_aligned) > 0:
                    print(f"      🔍 DEBUG portfolio_vol: aligned date range: {returns_df_aligned.index[0]} to {returns_df_aligned.index[-1]}")
            
            # Calculate covariance matrix using aligned data
            cov_matrix = returns_df_aligned.cov() * 252  # Annualized
            
            if debug:
                print(f"      🔍 DEBUG portfolio_vol: cov_matrix shape={cov_matrix.shape}")
                print(f"      🔍 DEBUG portfolio_vol: cov_matrix diagonal: {np.diag(cov_matrix)}")
            
            # Calculate portfolio variance
            portfolio_variance = 0.0
            for i, ticker1 in enumerate(tickers):
                if ticker1 in weights and ticker1 in cov_matrix:
                    weight1 = weights[ticker1]
                    for j, ticker2 in enumerate(tickers):
                        if ticker2 in weights and ticker2 in cov_matrix.columns:
                            weight2 = weights[ticker2]
                            portfolio_variance += weight1 * weight2 * cov_matrix.loc[ticker1, ticker2]
            
            portfolio_volatility = np.sqrt(portfolio_variance)
            return min(portfolio_volatility, 1.0)  # Cap at 100%
            
        except Exception as e:
            return 0.20  # Default 20% volatility on error
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Main entry point: Select stocks with volatility adjustment."""
        print(f"\n   🎯 Volatility-Adjusted Ensemble Strategy")
        print(f"   📅 Date: {current_date.date()}")
        print(f"   🔍 DEBUG: Input all_tickers count: {len(all_tickers)}")
        print(f"   🔍 DEBUG: Input ticker_data_grouped count: {len(ticker_data_grouped)}")
        print(f"   🔍 DEBUG: top_n: {top_n}")
        print(f"   🔍 DEBUG: MAX_PORTFOLIO_VOLATILITY: {MAX_PORTFOLIO_VOLATILITY}")
        
        # 1. Get picks from each strategy
        strategy_picks = {}
        for strategy in VOL_ENSEMBLE_STRATEGIES:
            print(f"   🔍 Getting picks from {strategy}...")
            picks = self.get_strategy_picks(
                strategy, all_tickers, ticker_data_grouped,
                current_date, train_start_date, top_n=top_n * 2
            )
            strategy_picks[strategy] = picks
            print(f"      → {len(picks)} picks: {picks[:5] if len(picks) > 5 else picks}")
        
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
        
        print(f"   🔍 DEBUG: Ensemble scores count: {len(ensemble_scores)}")
        if ensemble_scores:
            print(f"   🔍 DEBUG: Top 5 ensemble scores: {sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        if not ensemble_scores:
            print(f"   ⚠️ No consensus picks found (need at least 2 strategies to agree)")
            return []
        
        # 3. Sort by ensemble score
        sorted_candidates = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [ticker for ticker, score in sorted_candidates[:top_n * 2]]
        print(f"   🔍 DEBUG: Top candidates for volatility weighting: {len(top_candidates)}")
        
        # 4. Calculate inverse volatility weights
        vol_weights = self.calculate_inverse_volatility_weights(top_candidates, ticker_data_grouped, current_date)
        
        # 5. Select final portfolio with volatility constraint
        selected_stocks = []
        current_weights = {}
        
        # Greedy selection to maximize score while respecting volatility constraint
        print(f"   🔍 DEBUG: Starting greedy selection with MAX_PORTFOLIO_VOLATILITY={MAX_PORTFOLIO_VOLATILITY}")
        rejected_count = 0
        for ticker, score in sorted_candidates:
            if len(selected_stocks) >= top_n:
                break
            
            # Calculate portfolio volatility with this stock added
            test_weights = current_weights.copy()
            test_weights[ticker] = vol_weights.get(ticker, 0.1)
            
            # Normalize weights
            total_weight = sum(test_weights.values())
            test_weights = {t: w / total_weight for t, w in test_weights.items()}
            
            # Check portfolio volatility
            portfolio_vol = self.calculate_portfolio_volatility(
                list(test_weights.keys()), test_weights, ticker_data_grouped, current_date, debug=True
            )
            
            if portfolio_vol <= MAX_PORTFOLIO_VOLATILITY:
                selected_stocks.append(ticker)
                current_weights[ticker] = vol_weights.get(ticker, 0.1)
                print(f"      ✅ {ticker}: portfolio_vol={portfolio_vol:.1%} <= {MAX_PORTFOLIO_VOLATILITY:.1%}")
            else:
                rejected_count += 1
                print(f"      ❌ {ticker}: portfolio_vol={portfolio_vol:.1%} > {MAX_PORTFOLIO_VOLATILITY:.1%} (REJECTED)")
        
        print(f"   🔍 DEBUG: Selected {len(selected_stocks)} stocks, rejected {rejected_count} due to volatility constraint")
        
        # 6. Display results
        print(f"   ✅ Selected {len(selected_stocks)} stocks:")
        for ticker in selected_stocks:
            vol = self.calculate_stock_volatility(ticker, ticker_data_grouped, current_date)
            weight = current_weights.get(ticker, 0.1)
            score = ensemble_scores[ticker]
            print(f"      {ticker}: vol={vol:.1%}, weight={weight:.1%}, score={score:.3f}")
        
        # 7. Calculate and display portfolio volatility
        if selected_stocks:
            portfolio_vol = self.calculate_portfolio_volatility(
                selected_stocks, current_weights, ticker_data_grouped, current_date
            )
            print(f"   📊 Portfolio volatility: {portfolio_vol:.1%}")
        
        return selected_stocks


# ============================================
# Module-level function for integration
# ============================================

# Global instance for state persistence
_vol_ensemble_instance = None

def get_vol_ensemble_instance() -> VolatilityAdjustedEnsemble:
    """Get or create the global volatility ensemble instance."""
    global _vol_ensemble_instance
    if _vol_ensemble_instance is None:
        _vol_ensemble_instance = VolatilityAdjustedEnsemble()
    return _vol_ensemble_instance


def select_volatility_ensemble_stocks(all_tickers: List[str],
                                      ticker_data_grouped: Dict[str, pd.DataFrame],
                                      current_date: datetime = None,
                                      train_start_date: datetime = None,
                                      top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Volatility-Adjusted Ensemble stock selection strategy.
    
    This strategy combines multiple strategies with:
    1. Inverse volatility position sizing
    2. Portfolio volatility constraints
    3. Consensus filtering
    
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
    
    ensemble = get_vol_ensemble_instance()
    return ensemble.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_vol_ensemble_state():
    """Reset the global volatility ensemble instance."""
    global _vol_ensemble_instance
    _vol_ensemble_instance = None
