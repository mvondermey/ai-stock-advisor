"""
Mom-Vol Hybrid 6M Sentiment Strategy

Combines Mom-Vol Hybrid 6M (best performer +48.8%) with price-derived sentiment analysis.
Features:
- Mom-Vol Hybrid 6M base selection (6-month momentum with volatility adjustment)
- Price-derived sentiment proxy (short-term reversal + volume sentiment)
- Sentiment filtering and score boosting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import re
import random

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

# Strategies to include - focused on best performer only
SENTIMENT_ENSEMBLE_STRATEGIES = [
    'mom_vol_hybrid_6m',  # Best performer: +48.8%
]

# Sentiment parameters
SENTIMENT_LOOKBACK_DAYS = 7  # Days to look back for news
MIN_SENTIMENT_SCORE = -0.3  # Minimum sentiment score
MAX_SENTIMENT_SCORE = 0.3   # Maximum sentiment score
SENTIMENT_WEIGHT = 0.30     # Weight of sentiment in final score (30%)

# Strategy weights - focused on best performer
SENTIMENT_STRATEGY_WEIGHTS = {
    'mom_vol_hybrid_6m': 1.0,  # 100% weight for best performer
}


class SentimentEnhancedEnsemble:
    """
    Sentiment-Enhanced Ensemble that combines:
    1. Traditional strategy picks
    2. Price-derived sentiment proxy (no API calls needed)
    3. Risk-adjusted sentiment weighting
    """
    
    def __init__(self):
        pass  # No API fetcher needed - sentiment is derived from price data
        
    def calculate_combined_sentiment(self, ticker: str, current_date: datetime,
                                     ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        Calculate sentiment from price data (no API calls).
        Uses short-term reversal and volume sentiment as proxy signals.
        """
        if ticker_data_grouped is None or ticker not in ticker_data_grouped:
            return {'combined': 0.0, 'confidence': 0.0, 'sources': ['none'],
                    'short_term_reversal': 0.0, 'volume_sentiment': 0.0}
        
        try:
            data = ticker_data_grouped[ticker]
            if data is None or len(data) < 20:
                return {'combined': 0.0, 'confidence': 0.0, 'sources': ['price'],
                        'short_term_reversal': 0.0, 'volume_sentiment': 0.0}
            
            # Ensure UTC
            current_ts = pd.Timestamp(current_date)
            if current_ts.tz is None:
                current_ts = current_ts.tz_localize('UTC')
            if data.index.tz is None:
                data = data.copy()
                data.index = data.index.tz_localize('UTC')
            
            filtered = data[data.index <= current_ts]
            if len(filtered) < 20:
                return {'combined': 0.0, 'confidence': 0.0, 'sources': ['price'],
                        'short_term_reversal': 0.0, 'volume_sentiment': 0.0}
            
            close = filtered['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) < 20:
                return {'combined': 0.0, 'confidence': 0.0, 'sources': ['price'],
                        'short_term_reversal': 0.0, 'volume_sentiment': 0.0}
            
            latest = close.iloc[-1]
            if latest <= 0:
                return {'combined': 0.0, 'confidence': 0.0, 'sources': ['price'],
                        'short_term_reversal': 0.0, 'volume_sentiment': 0.0}
            
            # Short-term reversal: 5d return minus 20d return
            p5 = close.iloc[-min(5, len(close))]
            p20 = close.iloc[-min(20, len(close))]
            ret_5d = ((latest - p5) / p5 * 100) if p5 > 0 else 0.0
            ret_20d = ((latest - p20) / p20 * 100) if p20 > 0 else 0.0
            short_term_reversal = ret_5d - ret_20d
            
            # Volume sentiment: volume surge signed by price direction
            volume_sentiment = 0.0
            if 'Volume' in filtered.columns:
                vol_series = filtered['Volume'].dropna()
                if len(vol_series) >= 20:
                    recent_5d_vol = vol_series.tail(5).mean()
                    avg_20d_vol = vol_series.tail(20).mean()
                    vol_surge = (recent_5d_vol / avg_20d_vol) - 1.0 if avg_20d_vol > 0 else 0.0
                    price_dir = 1.0 if ret_5d > 0 else (-1.0 if ret_5d < 0 else 0.0)
                    volume_sentiment = vol_surge * price_dir
            
            # Combined sentiment: normalize to roughly -1 to +1 range
            # short_term_reversal is in % (typically -10 to +10), scale by /10
            # volume_sentiment is typically -1 to +1
            combined = np.clip(short_term_reversal / 10.0 * 0.6 + volume_sentiment * 0.4, -1.0, 1.0)
            confidence = 0.8  # Price data is always reliable
            
            return {
                'combined': combined,
                'confidence': confidence,
                'sources': ['price'],
                'short_term_reversal': short_term_reversal,
                'volume_sentiment': volume_sentiment
            }
        except Exception:
            return {'combined': 0.0, 'confidence': 0.0, 'sources': ['price'],
                    'short_term_reversal': 0.0, 'volume_sentiment': 0.0}
    
    def get_strategy_picks(self, strategy_name: str, all_tickers: List[str],
                          ticker_data_grouped: Dict[str, pd.DataFrame],
                          current_date: datetime, train_start_date: datetime = None,
                          top_n: int = 15) -> List[str]:
        """Get stock picks from a specific strategy."""
        try:
            if strategy_name == 'mom_vol_hybrid_6m':
                # Momentum-Volatility Hybrid 6M: Best performer
                from shared_strategies import select_momentum_volatility_hybrid_6m_stocks
                return select_momentum_volatility_hybrid_6m_stocks(
                    all_tickers, ticker_data_grouped, 
                    current_date=current_date, 
                    top_n=top_n
                )
            
            elif strategy_name == 'mom_vol_hybrid_1y':
                # Momentum-Volatility Hybrid 1Y: Strong performer
                from shared_strategies import select_momentum_volatility_hybrid_1y_stocks
                return select_momentum_volatility_hybrid_1y_stocks(
                    all_tickers, ticker_data_grouped, 
                    current_date=current_date, 
                    top_n=top_n
                )
            
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
            print(f"   ⚠️ Error getting picks for {strategy_name}: {e}")
            return []
    
    def calculate_ensemble_scores(self, strategy_picks: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate ensemble scores with strategy weights."""
        stock_scores = defaultdict(float)
        stock_counts = defaultdict(int)
        
        for strategy, picks in strategy_picks.items():
            weight = SENTIMENT_STRATEGY_WEIGHTS.get(strategy, 0.5)
            for rank, ticker in enumerate(picks):
                rank_score = 1.0 / (rank + 1)
                stock_scores[ticker] += weight * rank_score
                stock_counts[ticker] += 1
        
        # Apply consensus filter (at least 1 strategy for 2-strategy ensemble)
        # Stocks picked by both strategies get higher scores naturally
        consensus_scores = {
            ticker: score 
            for ticker, score in stock_scores.items()
            if stock_counts[ticker] >= 1
        }
        
        return consensus_scores
    
    def apply_sentiment_filter(self, candidates: List[Tuple[str, float]],
                              ticker_data_grouped: Dict[str, pd.DataFrame],
                              current_date: datetime,
                              min_sentiment_threshold: float = -0.1) -> List[Tuple[str, float]]:
        """Apply sentiment filtering to candidates using price-derived sentiment."""
        filtered_candidates = []
        
        print(f"   💭 Applying price-derived sentiment filter to {len(candidates)} candidates...")
        
        for ticker, ensemble_score in candidates[:20]:  # Analyze top 20
            try:
                sentiment_data = self.calculate_combined_sentiment(ticker, current_date, ticker_data_grouped)
                
                # Check if sentiment data is valid
                if sentiment_data and 'combined' in sentiment_data and sentiment_data['combined'] is not None:
                    # Apply sentiment threshold
                    if sentiment_data['combined'] >= min_sentiment_threshold:
                        # Adjust score based on sentiment
                        sentiment_boost = sentiment_data['combined'] * SENTIMENT_WEIGHT
                        confidence_factor = sentiment_data.get('confidence', 0.5)
                        
                        # Final score = ensemble_score + sentiment_boost * confidence
                        final_score = ensemble_score + sentiment_boost * confidence_factor
                        filtered_candidates.append((ticker, final_score))
                        
                        print(f"      {ticker}: ensemble={ensemble_score:.3f}, sentiment={sentiment_data['combined']:.2f}, final={final_score:.3f}")
                    else:
                        print(f"      {ticker}: ensemble={ensemble_score:.3f}, sentiment={sentiment_data['combined']:.2f} (filtered)")
                else:
                    # No sentiment data available, use ensemble score as-is
                    print(f"      {ticker}: ensemble={ensemble_score:.3f}, sentiment=N/A (using ensemble)")
                    filtered_candidates.append((ticker, ensemble_score))
                    
            except Exception as e:
                # Error getting sentiment, use ensemble score as-is
                print(f"      {ticker}: ensemble={ensemble_score:.3f}, sentiment=ERROR (using ensemble)")
                filtered_candidates.append((ticker, ensemble_score))
        
        # If no candidates passed the filter but we have original candidates, use them
        if not filtered_candidates and candidates:
            print(f"   ⚠️ No candidates passed sentiment filter, using top {min(len(candidates), 10)} ensemble picks")
            return candidates[:10]
        
        return filtered_candidates
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime = None,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Main entry point: Select stocks with sentiment enhancement."""
        print(f"\n   🎯 Sentiment-Enhanced Ensemble Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        # 1. Get picks from each strategy
        strategy_picks = {}
        for strategy in SENTIMENT_ENSEMBLE_STRATEGIES:
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
        
        # 2. Calculate ensemble scores
        ensemble_scores = self.calculate_ensemble_scores(strategy_picks)
        
        if not ensemble_scores:
            print(f"   ⚠️ No consensus picks found")
            return []
        
        # 3. Sort by ensemble score
        sorted_candidates = [(ticker, score) for ticker, score in ensemble_scores.items()]
        sorted_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Apply sentiment filtering
        sentiment_filtered = self.apply_sentiment_filter(
            sorted_candidates, ticker_data_grouped, current_date
        )
        
        if not sentiment_filtered:
            print(f"   ⚠️ No candidates passed sentiment filter")
            return []
        
        # 5. Sort by final score and select top N
        sentiment_filtered.sort(key=lambda x: x[1], reverse=True)
        final_selection = [ticker for ticker, score in sentiment_filtered[:top_n]]
        
        # 6. Display results
        print(f"   ✅ Selected {len(final_selection)} stocks:")
        for ticker, final_score in sentiment_filtered[:top_n]:
            sentiment_data = self.calculate_combined_sentiment(ticker, current_date, ticker_data_grouped)
            print(f"      {ticker}: final_score={final_score:.3f}, sentiment={sentiment_data['combined']:.2f}")
        
        # 7. Display sentiment summary
        positive_count = sum(1 for t in final_selection 
                           if self.calculate_combined_sentiment(t, current_date, ticker_data_grouped)['combined'] > 0.1)
        negative_count = sum(1 for t in final_selection 
                           if self.calculate_combined_sentiment(t, current_date, ticker_data_grouped)['combined'] < -0.1)
        neutral_count = len(final_selection) - positive_count - negative_count
        
        print(f"   📊 Sentiment breakdown: {positive_count} positive, {neutral_count} neutral, {negative_count} negative")
        
        return final_selection


# ============================================
# Module-level function for integration
# ============================================

# Global instance for state persistence
_sentiment_ensemble_instance = None

def get_sentiment_ensemble_instance() -> SentimentEnhancedEnsemble:
    """Get or create the global sentiment ensemble instance."""
    global _sentiment_ensemble_instance
    if _sentiment_ensemble_instance is None:
        _sentiment_ensemble_instance = SentimentEnhancedEnsemble()
    return _sentiment_ensemble_instance


def select_sentiment_ensemble_stocks(all_tickers: List[str],
                                     ticker_data_grouped: Dict[str, pd.DataFrame],
                                     current_date: datetime = None,
                                     train_start_date: datetime = None,
                                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Sentiment-Enhanced Ensemble stock selection strategy.
    
    This strategy combines multiple strategies with:
    1. News sentiment analysis
    2. Social media sentiment
    3. Ensemble consensus with sentiment filtering
    
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
    
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    ensemble = get_sentiment_ensemble_instance()
    return ensemble.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_sentiment_ensemble_state():
    """Reset the global sentiment ensemble instance."""
    global _sentiment_ensemble_instance
    _sentiment_ensemble_instance = None
