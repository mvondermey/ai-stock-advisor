"""
AI-Enhanced Volatility Ensemble Strategy

Builds on the successful Volatility Ensemble (+106% returns) with AI enhancements:
- AI-predicted optimal strategy weights
- Dynamic volatility caps based on market regime
- AI-enhanced consensus scoring with confidence weights
- Market regime detection for strategy adaptation
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

# Import prediction for AI enhancements
from prediction import predict_target_return

# ============================================
# Configuration Parameters
# ============================================

# Base strategies (same as volatility_ensemble)
AI_VOL_ENSEMBLE_STRATEGIES = [
    'static_bh_3m',
    'dyn_bh_1y_vol', 
    'risk_adj_mom',
    'quality_mom',
]

# AI Enhancement parameters
AI_WEIGHT_LEARNING_RATE = 0.1  # How fast AI adapts weights
AI_CONFIDENCE_THRESHOLD = 0.6  # Minimum AI confidence to influence decisions
AI_REGIME_LOOKBACK_DAYS = 30  # Days to analyze for regime detection

# Enhanced volatility parameters (AI-adjusted)
BASE_MAX_PORTFOLIO_VOLATILITY = 0.20  # 20% annualized max portfolio volatility
BASE_MAX_SINGLE_STOCK_VOLATILITY = 0.40  # 40% annualized max for any single stock
VOLATILITY_LOOKBACK_DAYS = 30  # Days to calculate volatility

# Position sizing
MIN_POSITION_WEIGHT = 0.05  # 5% minimum position weight
MAX_POSITION_WEIGHT = 0.40  # 40% maximum position weight

# ============================================
# AI-Enhanced Volatility Ensemble Class
# ============================================

class AIVolatilityEnsemble:
    """AI-Enhanced Volatility Ensemble Strategy"""
    
    def __init__(self):
        self.strategy_weights = {
            'static_bh_3m': 0.25,
            'dyn_bh_1y_vol': 0.25,
            'risk_adj_mom': 0.25,
            'quality_mom': 0.25,
        }
        self.performance_history = []
        self.market_regime = "normal"
        
    def detect_market_regime(self, all_tickers_data: pd.DataFrame, current_date: datetime) -> str:
        """Detect current market regime using volatility and trend analysis"""
        try:
            # Get recent market data (use SPY or average of all stocks)
            recent_data = all_tickers_data[all_tickers_data['date'] <= current_date].tail(AI_REGIME_LOOKBACK_DAYS)
            
            if len(recent_data) < 20:
                return "normal"
            
            # Calculate market volatility
            daily_returns = recent_data.groupby('date')['Close'].mean().pct_change()
            market_volatility = daily_returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate market trend
            recent_return = (recent_data.groupby('date')['Close'].mean().iloc[-1] / 
                           recent_data.groupby('date')['Close'].mean().iloc[0] - 1)
            
            # Determine regime
            if market_volatility > 0.25:  # High volatility
                return "volatile"
            elif recent_return < -0.10:  # Strong downtrend
                return "bear"
            elif recent_return > 0.10:  # Strong uptrend
                return "bull"
            else:
                return "normal"
                
        except Exception as e:
            return "normal"
    
    def adjust_volatility_caps(self, market_regime: str) -> Tuple[float, float]:
        """Adjust volatility caps based on market regime"""
        regime_adjustments = {
            "normal": (1.0, 1.0),      # No adjustment
            "volatile": (0.8, 0.7),    # Reduce caps in volatile markets
            "bear": (0.7, 0.6),        # Further reduce in bear markets
            "bull": (1.2, 1.1),        # Can increase caps in bull markets
        }
        
        adjustment = regime_adjustments.get(market_regime, (1.0, 1.0))
        
        max_portfolio_vol = BASE_MAX_PORTFOLIO_VOLATILITY * adjustment[0]
        max_single_vol = BASE_MAX_SINGLE_STOCK_VOLATILITY * adjustment[1]
        
        return max_portfolio_vol, max_single_vol
    
    def predict_strategy_performance(self, strategy_picks: Dict[str, List[str]], 
                                   all_tickers_data: pd.DataFrame,
                                   current_date: datetime) -> Dict[str, float]:
        """Use AI to predict relative performance of each strategy"""
        strategy_predictions = {}
        
        for strategy, picks in strategy_picks:
            if not picks:
                strategy_predictions[strategy] = 0.0
                continue
            
            try:
                # Get AI predictions for top 3 picks
                total_confidence = 0.0
                valid_predictions = 0
                
                for ticker in picks[:3]:  # Check top 3 picks
                    try:
                        prediction, confidence = predict_target_return(ticker, all_tickers_data, current_date)
                        if prediction is not None and confidence > AI_CONFIDENCE_THRESHOLD:
                            total_confidence += prediction * confidence
                            valid_predictions += 1
                    except Exception:
                        continue
                
                # Average confidence-weighted prediction
                if valid_predictions > 0:
                    strategy_predictions[strategy] = total_confidence / valid_predictions
                else:
                    strategy_predictions[strategy] = 0.0
                    
            except Exception as e:
                strategy_predictions[strategy] = 0.0
        
        return strategy_predictions
    
    def update_strategy_weights(self, strategy_predictions: Dict[str, float]):
        """Update strategy weights based on AI predictions"""
        # Base weights
        new_weights = self.strategy_weights.copy()
        
        # AI influence on weights
        for strategy, prediction in strategy_predictions.items():
            if prediction > 0 and strategy in new_weights:
                # Increase weight for positive predictions
                weight_adjustment = AI_WEIGHT_LEARNING_RATE * prediction
                new_weights[strategy] += weight_adjustment
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v/total_weight for k, v in new_weights.items()}
        
        # Ensure minimum weights
        min_weight = 0.1  # 10% minimum for any strategy
        for strategy in new_weights:
            if new_weights[strategy] < min_weight:
                new_weights[strategy] = min_weight
        
        # Re-normalize after minimum weight adjustment
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v/total_weight for k, v in new_weights.items()}
        
        self.strategy_weights = new_weights
    
    def get_strategy_picks(self, strategy_name: str, all_tickers: List[str],
                          ticker_data_grouped: Dict[str, pd.DataFrame],
                          current_date: datetime, train_start_date: datetime,
                          top_n: int = 10) -> List[str]:
        """Get stock picks for a specific strategy"""
        try:
            if strategy_name == 'static_bh_3m':
                # Use 3-month momentum for static BH
                return select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                             current_date=current_date,
                                             lookback_days=90, top_n=top_n)
            
            elif strategy_name == 'dyn_bh_1y_vol':
                # Dynamic BH with volatility filter
                picks = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped,
                                                current_date=current_date,
                                                lookback_days=365, top_n=top_n*2)
                
                # Apply volatility filter
                filtered_picks = []
                for ticker in picks:
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
    
    def calculate_stock_volatility(self, ticker: str, ticker_data_grouped: Dict[str, pd.DataFrame],
                                 current_date: datetime) -> float:
        """Calculate historical volatility for a stock"""
        try:
            if ticker not in ticker_data_grouped:
                return 0.5  # Default high volatility
            
            ticker_data = ticker_data_grouped[ticker]
            historical_data = ticker_data[ticker_data['date'] <= current_date].tail(VOLATILITY_LOOKBACK_DAYS)
            
            if len(historical_data) < 10:
                return 0.5  # Default high volatility
            
            # Calculate daily returns
            daily_returns = historical_data['Close'].pct_change().dropna()
            
            # Annualized volatility
            volatility = daily_returns.std() * np.sqrt(252)
            
            return min(volatility, 1.0)  # Cap at 100%
            
        except Exception:
            return 0.5  # Default high volatility
    
    def calculate_ai_enhanced_scores(self, strategy_picks: Dict[str, List[str]], 
                                   strategy_predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate ensemble scores with AI-enhanced weighting"""
        stock_scores = defaultdict(float)
        stock_counts = defaultdict(int)
        
        for strategy, picks in strategy_picks.items():
            # Use AI-enhanced weights
            weight = self.strategy_weights.get(strategy, 0.25)
            
            # Boost weight based on AI prediction
            ai_boost = 1.0 + strategy_predictions.get(strategy, 0.0)
            enhanced_weight = weight * ai_boost
            
            for rank, ticker in enumerate(picks):
                # Higher rank gets higher score
                rank_score = 1.0 / (rank + 1)
                stock_scores[ticker] += enhanced_weight * rank_score
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
                                   current_date: datetime) -> float:
        """Calculate portfolio-level volatility"""
        try:
            # Get returns data for all stocks
            returns_data = {}
            
            for ticker in tickers:
                if ticker in ticker_data_grouped:
                    ticker_data = ticker_data_grouped[ticker]
                    historical_data = ticker_data[ticker_data['date'] <= current_date].tail(VOLATILITY_LOOKBACK_DAYS)
                    
                    if len(historical_data) >= 10:
                        daily_returns = historical_data['Close'].pct_change().dropna()
                        returns_data[ticker] = daily_returns
            
            if len(returns_data) < 2:
                return 0.15  # Default moderate volatility
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate portfolio returns
            portfolio_returns = sum(returns_df[ticker] * weights.get(ticker, 0) for ticker in tickers if ticker in returns_df.columns)
            
            # Calculate annualized volatility
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            
            return min(portfolio_volatility, 1.0)  # Cap at 100%
            
        except Exception:
            return 0.15  # Default moderate volatility
    
    def optimize_position_sizes(self, tickers: List[str], scores: Dict[str, float],
                             ticker_data_grouped: Dict[str, pd.DataFrame],
                             current_date: datetime, max_portfolio_vol: float,
                             max_single_vol: float) -> Dict[str, float]:
        """Optimize position sizes using AI-enhanced volatility management"""
        try:
            # Calculate individual stock volatilities
            volatilities = {}
            for ticker in tickers:
                vol = self.calculate_stock_volatility(ticker, ticker_data_grouped, current_date)
                volatilities[ticker] = min(vol, max_single_vol)
            
            # Start with equal weights
            weights = {ticker: 1.0/len(tickers) for ticker in tickers}
            
            # Adjust weights based on scores and volatility
            for ticker in tickers:
                score = scores.get(ticker, 0)
                vol = volatilities.get(ticker, 0.3)
                
                # Higher score = higher weight, higher volatility = lower weight
                score_factor = 1.0 + score
                vol_factor = 1.0 / (1.0 + vol)
                
                weights[ticker] *= score_factor * vol_factor
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # Apply position size constraints
            for ticker in weights:
                weights[ticker] = max(MIN_POSITION_WEIGHT, min(MAX_POSITION_WEIGHT, weights[ticker]))
            
            # Re-normalize after constraints
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # Check portfolio volatility constraint
            portfolio_vol = self.calculate_portfolio_volatility(tickers, weights, ticker_data_grouped, current_date)
            
            if portfolio_vol > max_portfolio_vol:
                # Scale down all weights proportionally
                scale_factor = max_portfolio_vol / portfolio_vol
                weights = {k: v*scale_factor for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            # Fallback to equal weights
            return {ticker: 1.0/len(tickers) for ticker in tickers}

# ============================================
# Main Selection Function
# ============================================

def select_ai_volatility_ensemble_stocks(all_tickers: List[str], 
                                        ticker_data_grouped: Dict[str, pd.DataFrame],
                                        current_date: datetime = None,
                                        train_start_date: datetime = None,
                                        top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """Select stocks using AI-Enhanced Volatility Ensemble Strategy"""
    
    if current_date is None:
        current_date = datetime.now()
    
    print(f"   🤖 AI Volatility Ensemble: Processing {len(all_tickers)} tickers")
    
    # Initialize AI ensemble
    ai_ensemble = AIVolatilityEnsemble()
    
    # Detect market regime
    all_tickers_df = pd.concat([df for df in ticker_data_grouped.values()], ignore_index=True)
    market_regime = ai_ensemble.detect_market_regime(all_tickers_df, current_date)
    print(f"   📊 Market regime: {market_regime}")
    
    # Adjust volatility caps based on regime
    max_portfolio_vol, max_single_vol = ai_ensemble.adjust_volatility_caps(market_regime)
    print(f"   📏 Volatility caps: Portfolio={max_portfolio_vol:.1%}, Single={max_single_vol:.1%}")
    
    # Get picks from each strategy
    strategy_picks = {}
    for strategy in AI_VOL_ENSEMBLE_STRATEGIES:
        picks = ai_ensemble.get_strategy_picks(
            strategy, all_tickers, ticker_data_grouped, 
            current_date, train_start_date, top_n*2
        )
        strategy_picks[strategy] = picks
        print(f"   📈 {strategy}: {len(picks)} picks")
    
    # Use AI to predict strategy performance
    strategy_predictions = ai_ensemble.predict_strategy_performance(
        strategy_picks, all_tickers_df, current_date
    )
    print(f"   🧠 AI predictions: {strategy_predictions}")
    
    # Update strategy weights based on AI predictions
    ai_ensemble.update_strategy_weights(strategy_predictions)
    print(f"   ⚖️ Updated weights: {ai_ensemble.strategy_weights}")
    
    # Calculate AI-enhanced ensemble scores
    ensemble_scores = ai_ensemble.calculate_ai_enhanced_scores(strategy_picks, strategy_predictions)
    
    if not ensemble_scores:
        print(f"   ⚠️ No consensus stocks found")
        return []
    
    # Sort by score and get top candidates
    sorted_candidates = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:top_n*2]
    
    # Optimize position sizes
    tickers = [ticker for ticker, score in top_candidates]
    scores = {ticker: score for ticker, score in top_candidates}
    
    optimized_weights = ai_ensemble.optimize_position_sizes(
        tickers, scores, ticker_data_grouped, current_date,
        max_portfolio_vol, max_single_vol
    )
    
    # Sort by final weights and return top N
    final_selection = sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
    selected_tickers = [ticker for ticker, weight in final_selection]
    
    print(f"   ✅ AI Volatility Ensemble selected {len(selected_tickers)} stocks")
    print(f"   📊 Final weights: {dict(final_selection)}")
    
    return selected_tickers
