"""
Risk-Adjusted Momentum + Sentiment Strategy
Combines Risk-Adjusted Momentum scoring with sentiment analysis for enhanced stock selection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared_strategies import select_risk_adj_mom_stocks
from sentiment_ensemble import get_sentiment_ensemble_instance


def select_risk_adj_mom_sentiment_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    price_history_cache=None,
) -> List[str]:
    """
    Select stocks using Risk-Adjusted Momentum + Sentiment analysis.
    
    This strategy combines:
    1. Risk-Adjusted Momentum scoring (volatility-adjusted returns)
    2. Sentiment analysis (news/social media sentiment)
    
    The goal is to get the best of both worlds:
    - Risk-adjusted performance from momentum analysis
    - Market sentiment insights for timing
    
    Args:
        all_tickers: List of all ticker symbols to analyze
        ticker_data_grouped: Dictionary of ticker -> DataFrame with price data
        current_date: Current date for analysis
        top_n: Number of stocks to select
        
    Returns:
        List of selected ticker symbols
    """
    
    print(f"   📊 Risk-Adj Mom Sentiment: Analyzing {len(all_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
    
    try:
        # Get Risk-Adjusted Momentum selections
        print(f"   🎯 Risk-Adj Mom Sentiment: Getting Risk-Adjusted Momentum picks...")
        risk_adj_mom_stocks = select_risk_adj_mom_stocks(
            all_tickers,
            ticker_data_grouped,
            current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        )
        
        print(f"   📈 Risk-Adj Mom Sentiment: Found {len(risk_adj_mom_stocks)} Risk-Adj Mom candidates")
        
        # Apply sentiment filtering directly to Risk-Adj Mom candidates
        # (NOT via select_sentiment_ensemble_stocks which runs its own Mom-Vol Hybrid 6M selection)
        print(f"   💭 Risk-Adj Mom Sentiment: Applying sentiment analysis...")
        ensemble = get_sentiment_ensemble_instance()
        
        # Convert stock list to (ticker, rank_score) tuples for sentiment filter
        candidates_with_scores = [(ticker, 1.0 / (i + 1)) for i, ticker in enumerate(risk_adj_mom_stocks)]
        
        sentiment_filtered = ensemble.apply_sentiment_filter(
            candidates_with_scores, ticker_data_grouped, current_date
        )
        
        # Sort by final score and extract tickers
        sentiment_filtered.sort(key=lambda x: x[1], reverse=True)
        sentiment_filtered_stocks = [ticker for ticker, score in sentiment_filtered[:top_n]]
        
        print(f"   ✅ Risk-Adj Mom Sentiment: Selected {len(sentiment_filtered_stocks)} stocks")
        
        # Show the selected stocks with their characteristics
        if sentiment_filtered_stocks:
            print(f"   🎯 Risk-Adj Mom Sentiment: Final selections:")
            for i, ticker in enumerate(sentiment_filtered_stocks[:5]):
                # Get basic metrics for display
                if ticker in ticker_data_grouped:
                    data = ticker_data_grouped[ticker]
                    if len(data) >= 126:  # 6 months of trading days
                        recent_data = data.tail(126)
                        momentum_6m = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100
                        volatility = recent_data['Close'].pct_change().std() * np.sqrt(252) * 100
                        print(f"      {i+1}. {ticker}: Risk-Adj Mom + Sentiment, 6M={momentum_6m:+.1f}%, Vol={volatility:.1f}%")
        
        return sentiment_filtered_stocks
        
    except Exception as e:
        print(f"   ❌ Risk-Adj Mom Sentiment strategy error: {e}")
        print("   ⚠️ Risk-Adj Mom Sentiment: Returning no selection")
        return []
