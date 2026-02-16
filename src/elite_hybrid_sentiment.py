"""
Elite Hybrid Sentiment Strategy

Combines Elite Hybrid stock selection with real news sentiment analysis.
Features:
- Elite Hybrid base selection (Mom-Vol 6M + 1Y/3M Ratio + Dip Detection)
- Real news sentiment from Alpha Vantage API (FREE tier)
- Reddit sentiment from public API
- Sentiment filtering and score boosting
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

# Import real sentiment fetcher
from sentiment_fetcher import get_sentiment_fetcher

# Import Elite Hybrid strategy
from elite_hybrid_strategy import select_elite_hybrid_stocks

# ============================================
# Configuration Parameters
# ============================================

# Sentiment parameters
ELITE_SENTIMENT_LOOKBACK_DAYS = 7  # Days to look back for news
ELITE_MIN_SENTIMENT_SCORE = -0.2   # Minimum sentiment score (slightly stricter)
ELITE_SENTIMENT_WEIGHT = 0.25      # Weight of sentiment in final score (25%)


class EliteHybridSentiment:
    """
    Elite Hybrid Sentiment Strategy that combines:
    1. Elite Hybrid stock picks (Mom-Vol 6M + 1Y/3M Ratio + Dip Detection)
    2. Real news sentiment from Alpha Vantage API
    3. Reddit sentiment from public API
    4. Sentiment-adjusted scoring
    """
    
    def __init__(self):
        self.sentiment_fetcher = get_sentiment_fetcher()
        
    def calculate_combined_sentiment(self, ticker: str, current_date: datetime) -> Dict:
        """Get combined sentiment from real APIs."""
        return self.sentiment_fetcher.get_combined_sentiment(ticker, current_date)
    
    def get_elite_hybrid_picks(self, all_tickers: List[str],
                               ticker_data_grouped: Dict[str, pd.DataFrame],
                               current_date: datetime,
                               top_n: int = 20) -> List[str]:
        """Get stock picks from Elite Hybrid strategy."""
        try:
            return select_elite_hybrid_stocks(
                all_tickers, ticker_data_grouped, 
                current_date=current_date, 
                top_n=top_n
            )
        except Exception as e:
            print(f"   ⚠️ Error getting Elite Hybrid picks: {e}")
            return []
    
    def apply_sentiment_filter(self, candidates: List[str],
                              ticker_data_grouped: Dict[str, pd.DataFrame],
                              current_date: datetime,
                              min_sentiment_threshold: float = ELITE_MIN_SENTIMENT_SCORE) -> List[Tuple[str, float]]:
        """Apply sentiment filtering and scoring to candidates."""
        scored_candidates = []
        
        print(f"   📰 Applying sentiment analysis to {len(candidates)} Elite Hybrid candidates...")
        
        for rank, ticker in enumerate(candidates):
            try:
                # Base score from Elite Hybrid ranking (higher rank = higher score)
                base_score = 1.0 / (rank + 1)
                
                sentiment_data = self.calculate_combined_sentiment(ticker, current_date)
                
                # Check if sentiment data is valid
                if sentiment_data and 'combined' in sentiment_data and sentiment_data['combined'] is not None:
                    sentiment_score = sentiment_data['combined']
                    confidence = sentiment_data.get('confidence', 0.5)
                    
                    # Apply sentiment threshold
                    if sentiment_score >= min_sentiment_threshold:
                        # Boost score based on sentiment
                        sentiment_boost = sentiment_score * ELITE_SENTIMENT_WEIGHT * confidence
                        final_score = base_score + sentiment_boost
                        scored_candidates.append((ticker, final_score, sentiment_score))
                        
                        print(f"      {ticker}: rank={rank+1}, base={base_score:.3f}, sentiment={sentiment_score:.2f}, final={final_score:.3f}")
                    else:
                        print(f"      {ticker}: rank={rank+1}, sentiment={sentiment_score:.2f} (filtered - below {min_sentiment_threshold})")
                else:
                    # No sentiment data available, use base score
                    print(f"      {ticker}: rank={rank+1}, base={base_score:.3f}, sentiment=N/A (using base)")
                    scored_candidates.append((ticker, base_score, 0.0))
                    
            except Exception as e:
                # Error getting sentiment, use base score
                base_score = 1.0 / (rank + 1)
                print(f"      {ticker}: rank={rank+1}, base={base_score:.3f}, sentiment=ERROR (using base)")
                scored_candidates.append((ticker, base_score, 0.0))
        
        # If no candidates passed the filter but we have original candidates, use them
        if not scored_candidates and candidates:
            print(f"   ⚠️ No candidates passed sentiment filter, using top {min(len(candidates), 10)} Elite Hybrid picks")
            return [(t, 1.0/(i+1), 0.0) for i, t in enumerate(candidates[:10])]
        
        return scored_candidates
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime = None,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Main entry point: Select stocks with Elite Hybrid + Sentiment."""
        print(f"\n   🎯 Elite Hybrid Sentiment Strategy")
        print(f"   📅 Date: {current_date.date() if current_date else 'N/A'}")
        
        # 1. Get Elite Hybrid picks (get more candidates for sentiment filtering)
        print(f"   🔍 Getting Elite Hybrid picks...")
        elite_picks = self.get_elite_hybrid_picks(
            all_tickers, ticker_data_grouped, current_date, top_n=top_n * 2
        )
        print(f"      → {len(elite_picks)} Elite Hybrid candidates")
        
        # Safety check
        if not elite_picks:
            print(f"   ⚠️ No Elite Hybrid picks found!")
            return []
        
        # 2. Apply sentiment filtering and scoring
        sentiment_scored = self.apply_sentiment_filter(
            elite_picks, ticker_data_grouped, current_date
        )
        
        if not sentiment_scored:
            print(f"   ⚠️ No candidates after sentiment filtering")
            return elite_picks[:top_n]  # Fallback to Elite Hybrid picks
        
        # 3. Sort by final score and select top N
        sentiment_scored.sort(key=lambda x: x[1], reverse=True)
        final_selection = [ticker for ticker, score, sentiment in sentiment_scored[:top_n]]
        
        # 4. Display results
        print(f"   ✅ Selected {len(final_selection)} stocks:")
        for ticker, final_score, sentiment in sentiment_scored[:top_n]:
            print(f"      {ticker}: final_score={final_score:.3f}, sentiment={sentiment:.2f}")
        
        # 5. Display sentiment summary
        positive_count = sum(1 for _, _, s in sentiment_scored[:top_n] if s > 0.1)
        negative_count = sum(1 for _, _, s in sentiment_scored[:top_n] if s < -0.1)
        neutral_count = len(final_selection) - positive_count - negative_count
        
        print(f"   📊 Sentiment breakdown: {positive_count} positive, {neutral_count} neutral, {negative_count} negative")
        
        return final_selection


# ============================================
# Module-level function for integration
# ============================================

# Global instance for state persistence
_elite_hybrid_sentiment_instance = None

def get_elite_hybrid_sentiment_instance() -> EliteHybridSentiment:
    """Get or create the global Elite Hybrid Sentiment instance."""
    global _elite_hybrid_sentiment_instance
    if _elite_hybrid_sentiment_instance is None:
        _elite_hybrid_sentiment_instance = EliteHybridSentiment()
    return _elite_hybrid_sentiment_instance


def select_elite_hybrid_sentiment_stocks(all_tickers: List[str],
                                         ticker_data_grouped: Dict[str, pd.DataFrame],
                                         current_date: datetime = None,
                                         train_start_date: datetime = None,
                                         top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Elite Hybrid Sentiment stock selection strategy.
    
    This strategy combines Elite Hybrid with:
    1. News sentiment analysis
    2. Social media sentiment
    3. Sentiment-adjusted scoring
    
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
    
    ensemble = get_elite_hybrid_sentiment_instance()
    return ensemble.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_elite_hybrid_sentiment_state():
    """Reset the global Elite Hybrid Sentiment instance."""
    global _elite_hybrid_sentiment_instance
    _elite_hybrid_sentiment_instance = None
